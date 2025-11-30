/* SimpleLICM.cpp
 *
 * This pass hoists loop-invariant code before the loop when it is safe to do so.
 *
 * Compatible with New Pass Manager
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

struct SimpleLICM : public PassInfoMixin<SimpleLICM> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR,
                        LPMUpdater &) {
    DominatorTree &DT = AR.DT;

    BasicBlock *Preheader = L.getLoopPreheader();
    if (!Preheader) {
      errs() << "No preheader, skipping loop\n";
      return PreservedAnalyses::all();
    }

    SmallPtrSet<Instruction *, 8> InvariantSet;
    // Worklist alogrithm to identify loop invariant instructions
    /*************************************/

    SmallVector<Instruction *, 32> Worklist;
    for (Loop::block_iterator BI = L.block_begin(), BE = L.block_end();
	BI != BE; ++BI){
      BasicBlock *BB = *BI;
      for (Instruction &I : *BB) {
        // ignores PHIs and any memory reading/writing instructions
        if (isa<PHINode>(&I))
          continue;
        if (I.mayReadOrWriteMemory())
          continue;

        bool AllOperandsOutsideLoop = true;
        for (Use &Op : I.operands()) {
          Value *V = Op.get();
          if (Instruction *OpInst = dyn_cast<Instruction>(V)) {
            if (L.contains(OpInst)) {
              AllOperandsOutsideLoop = false;
              break;
            }
          }
        }

        if (AllOperandsOutsideLoop) {
          InvariantSet.insert(&I);
          Worklist.push_back(&I);
        }
      }
    }

    while (!Worklist.empty()) {
      Instruction *Inv = Worklist.pop_back_val();

      for (User *U : Inv->users()) {
        Instruction *UserI = dyn_cast<Instruction>(U);
        if (!UserI)
          continue;
        if (!L.contains(UserI))
          continue;
        if (InvariantSet.contains(UserI))
          continue;
        if (isa<PHINode>(UserI))
          continue;
        if (UserI->mayReadOrWriteMemory())
          continue;                  // skip memory operations

        bool AllOperandsInvOrOutside = true;
        for (Use &Op : UserI->operands()) {
          Value *V = Op.get();
          if (Instruction *OpInst = dyn_cast<Instruction>(V)) {
            if (L.contains(OpInst) && !InvariantSet.contains(OpInst)) {
              AllOperandsInvOrOutside = false;
              break;
            }
          }
        }

        if (AllOperandsInvOrOutside) {
          InvariantSet.insert(UserI);
          Worklist.push_back(UserI);
        }
      }
    }

    /*************************************/
    
    //Actually hoist the instructions
    for (Instruction *I : InvariantSet) {
      if (isSafeToSpeculativelyExecute(I) && dominatesAllLoopExits(I, &L, DT)) {
        errs() << "Hoisting: " << *I << "\n";
        I->moveBefore(Preheader->getTerminator());
      }
    }

    return InvariantSet.empty() ? PreservedAnalyses::all()
                                : PreservedAnalyses::none();
  }

  bool dominatesAllLoopExits(Instruction *I, Loop *L, DominatorTree &DT) {
    SmallVector<BasicBlock *, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (BasicBlock *EB : ExitBlocks) {
      if (!DT.dominates(I, EB))
        return false;
    }
    return true;
  }
};

llvm::PassPluginLibraryInfo getSimpleLICMPluginInfo() {
  errs() << "SimpleLICM plugin: getSimpleLICMPluginInfo() called\n";
  return {LLVM_PLUGIN_API_VERSION, "simple-licm", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, LoopPassManager &LPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "simple-licm") {
                    LPM.addPass(SimpleLICM());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  errs() << "SimpleLICM plugin: llvmGetPassPluginInfo() called\n";
  return getSimpleLICMPluginInfo();
}
