// LoopInfoExample.cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct LoopInfoExample : public PassInfoMixin<LoopInfoExample> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    errs() << "Analyzing function: " << F.getName() << "\n";

    // Get LoopInfo for this function
    LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

    // Top-level loops
    for (Loop *L : LI) {
      printLoopInfo(L, /*Depth=*/0);
    }

    return PreservedAnalyses::all();
  }

  // Recursive helper to print loop info w/ depth indentation 
  void printLoopInfo(Loop *L, unsigned Depth) {
    std::string Indent(Depth * 2, ' ');

    BasicBlock *Header    = L->getHeader();
    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Latch     = L->getLoopLatch();

    errs() << Indent << "Loop at depth " << L->getLoopDepth() << "\n";

    errs() << Indent << "  Header:    ";
    if (Header)
      Header->printAsOperand(errs(), false);
    else
      errs() << "<none>";
    errs() << "\n";

    errs() << Indent << "  Preheader: ";
    if (Preheader)
      Preheader->printAsOperand(errs(), false);
    else
      errs() << "<none>";
    errs() << "\n";

    errs() << Indent << "  Latch:     ";
    if (Latch)
      Latch->printAsOperand(errs(), false);
    else
      errs() << "<none>";
    errs() << "\n";

    // (c)
    errs() << Indent << "  Num blocks in loop: " << L->getNumBlocks() << "\n";
    errs() << Indent << "  Is innermost? "
           << (L->getSubLoops().empty() ? "yes" : "no") << "\n";

    // (c)
    errs() << Indent << "  Blocks:\n";
    for (BasicBlock *BB : L->blocks()) {
      errs() << Indent << "    ";
      BB->printAsOperand(errs(), false);
      errs() << "\n";
    }

    // (b)
    for (Loop *SubL : L->getSubLoops()) {
      errs() << Indent << "  Subloop:\n";
      printLoopInfo(SubL, Depth + 1);
    }
  }
};

} // namespace

// Register pass plugin
llvm::PassPluginLibraryInfo getLoopInfoExamplePluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "loop-info-example", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "loop-info-example") {
                FPM.addPass(LoopInfoExample());
                return true;
              }
              return false;
            });
      }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLoopInfoExamplePluginInfo();
}
