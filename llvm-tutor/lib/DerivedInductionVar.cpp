/* DerivedInductionVar.cpp 
 *
 * This pass detects derived induction variables using ScalarEvolution and
 * performs a simple transformation to eliminate them in inner loops of
 * loop nests.
 *
 * Compatible with New Pass Manager
*/

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

namespace {

class DerivedInductionVar : public PassInfoMixin<DerivedInductionVar> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

    const DataLayout &DL = F.getParent()->getDataLayout();
    SCEVExpander Exp(SE, DL, "derivediv");
    bool Changed = false;
    for (Loop *L : LI) {
      if (!L->getParentLoop())
        continue;

      errs() << "Analyzing INNER loop in function " << F.getName() << ":\n";

      BasicBlock *Header = L->getHeader();
      if (!Header)
        continue;

      SmallVector<PHINode *, 8> DerivedIVs;

      for (PHINode &PN : Header->phis()) {
        if (!PN.getType()->isIntegerTy())
          continue;

        const SCEV *S = SE.getSCEV(&PN);

        if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
          if (!AR->isAffine())
            continue;

          errs() << "  Derived induction variable (analysis): "
                 << PN.getName() << " = {" << *AR->getStart()
                 << ",+," << *AR->getStepRecurrence(SE) << "}<"
                 << L->getHeader()->getName() << ">\n";

          DerivedIVs.push_back(&PN);
        }
      }

      for (PHINode *PN : DerivedIVs) {
        const SCEV *S = SE.getSCEV(PN);

        SmallVector<Use *, 8> UsesInLoop;
        for (Use &U : PN->uses()) {
          if (Instruction *UserI = dyn_cast<Instruction>(U.getUser())) {
            if (L->contains(UserI)) {
              UsesInLoop.push_back(&U);
            }
          }
        }

        for (Use *UPtr : UsesInLoop) {
          Instruction *UserI = cast<Instruction>(UPtr->getUser());
          Value *NewV = Exp.expandCodeFor(S, PN->getType(), UserI);
          UPtr->set(NewV);
          Changed = true;
        }

        if (PN->use_empty()) {
          errs() << "  Eliminating induction variable PHI: "
                 << PN->getName() << "\n";
          PN->eraseFromParent();
          Changed = true;
        }
      }
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

// Register the pass
llvm::PassPluginLibraryInfo getDerivedInductionVarPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DerivedInductionVar", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "derived-iv") {
                    FPM.addPass(DerivedInductionVar());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getDerivedInductionVarPluginInfo();
}
