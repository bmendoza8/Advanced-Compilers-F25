// Implemented using llvm 18 as explained in the other assignments. 

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct MemorySSADemoPass : PassInfoMixin<MemorySSADemoPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &MSSAResult = AM.getResult<MemorySSAAnalysis>(F);
    MemorySSA &MSSA = MSSAResult.getMSSA();

    errs() << "=== MemorySSADemo on function: " << F.getName() << " ===\n";

    // Iterate over basic blocks to show all MemoryAccesses
    for (auto &BB : F) {
      errs() << "BasicBlock: " << BB.getName() << "\n";

      // MemoryPHI nodes are found at block entries
      if (auto *PhiAcc = MSSA.getMemoryAccess(&BB)) {
        if (auto *MPhi = dyn_cast<MemoryPhi>(PhiAcc)) {
          errs() << "  MemoryPhi for block " << BB.getName() << ":\n";
          for (unsigned i = 0; i < MPhi->getNumIncomingValues(); ++i) {
            auto *IncomingAcc = MPhi->getIncomingValue(i);
            auto *Pred = MPhi->getIncomingBlock(i);
            errs() << "    from " << Pred->getName() << ": ";
            IncomingAcc->print(errs());
            errs() << "\n";
          }
        }
      }

      // Iterate over instructions for MemoryDEF/Use
      for (auto &I : BB) {
        if (auto *MA = MSSA.getMemoryAccess(&I)) {
          errs() << "  ";
          MA->print(errs());
          errs() << "\n";
        }
      }
    }

    // DOT-style graph
    DenseMap<const MemoryAccess *, unsigned> IDs;
    unsigned NextID = 0;

    auto getID = [&](const MemoryAccess *MA) -> unsigned {
      auto It = IDs.find(MA);
      if (It != IDs.end())
        return It->second;
      unsigned NewID = NextID++;
      IDs[MA] = NewID;
      return NewID;
    };

    errs() << "\n;
    errs() << "digraph \"MemorySSA_" << F.getName() << "\" {\n";
    for (auto &BB : F) {
      if (auto *PhiAcc = MSSA.getMemoryAccess(&BB)) {
        if (auto *MPhi = dyn_cast<MemoryPhi>(PhiAcc)) {
          unsigned ID = getID(MPhi);
          errs() << "  n" << ID << " [label=\"MemoryPhi in "
                 << BB.getName() << "\"];\n";
        }
      }
      for (auto &I : BB) {
        if (auto *MA = MSSA.getMemoryAccess(&I)) {
          unsigned ID = getID(MA);
          errs() << "  n" << ID << " [label=\"";
          MA->print(errs());
          errs() << "\"];\n";
        }
      }
    }
    for (auto &BB : F) {
      if (auto *PhiAcc = MSSA.getMemoryAccess(&BB)) {
        if (auto *MPhi = dyn_cast<MemoryPhi>(PhiAcc)) {
          unsigned ToID = getID(MPhi);
          for (unsigned i = 0; i < MPhi->getNumIncomingValues(); ++i) {
            auto *IncomingAcc = MPhi->getIncomingValue(i);
            unsigned FromID = getID(IncomingAcc);
            errs() << "  n" << FromID << " -> n" << ToID << ";\n";
          }
        }
      }
      for (auto &I : BB) {
        if (auto *MA = MSSA.getMemoryAccess(&I)) {
          if (auto *MD = dyn_cast<MemoryDef>(MA)) {
            MemoryAccess *Def = MD->getDefiningAccess();
            unsigned FromID = getID(Def);
            unsigned ToID = getID(MD);
            errs() << "  n" << FromID << " -> n" << ToID << ";\n";
          } else if (auto *MU = dyn_cast<MemoryUse>(MA)) {
            MemoryAccess *Def = MU->getDefiningAccess();
            unsigned FromID = getID(Def);
            unsigned ToID = getID(MU);
            errs() << "  n" << FromID << " -> n" << ToID << ";\n";
          }
        }
      }
    }

    errs() << "}\n\n";

    return PreservedAnalyses::all();
  }
};

// DSE
struct SimpleDSEPass : PassInfoMixin<SimpleDSEPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &MSSAResult = AM.getResult<MemorySSAAnalysis>(F);
    MemorySSA &MSSA = MSSAResult.getMSSA();

    SmallVector<Instruction *, 16> ToErase;

    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;
        if (SI->isVolatile())
          continue;

        MemoryAccess *MA = MSSA.getMemoryAccess(SI);
        if (!MA)
          continue;

        auto *MD = dyn_cast<MemoryDef>(MA);
        if (!MD)
          continue;

        bool HasRealUse = false;
        unsigned NumDefUsers = 0;

        for (User *U : MD->users()) {
          if (isa<MemoryUse>(U) || isa<MemoryPhi>(U)) {
            HasRealUse = true;
            break;
          }
          if (isa<MemoryDef>(U)) {
            NumDefUsers++;
          }
        }
        if (HasRealUse)
          continue;
        if (NumDefUsers == 0)
          continue;

        errs() << "DSE: removing dead store: " << *SI << "\n";
        ToErase.push_back(SI);
      }
    }
    for (Instruction *I : ToErase){
      I->eraseFromParent();
    }

    if (ToErase.empty())
      return PreservedAnalyses::all();
    return PreservedAnalyses::none();
  }
};


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MemorySSADemoPass", "v1.0",
          [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](FunctionAnalysisManager &FAM) {
                  FAM.registerPass([] { return MemorySSAAnalysis(); });
                });

            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "memssa-demo") {
                    FPM.addPass(MemorySSADemoPass());
                    return true;
                  }
                  if (Name == "simple-dse") {
                    FPM.addPass(SimpleDSEPass());
                    return true;
                  }
                  return false;
                });
          }};
}

