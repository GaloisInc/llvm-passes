#include "llvm/Pass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const int WORD_SIZE = 64;
const size_t TRACE_NUM_ARGS = 8;

namespace {
struct SetIntrinsicAttrs : public FunctionPass {
  static char ID;
  SetIntrinsicAttrs() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (!F.getName().startswith("__llvm__")) {
      return false;
    }

    // Set `optnone` to prevent the function body from being replaced with a
    // call to the `@llvm.*` intrinsic that it implements.
    F.removeFnAttr(Attribute::OptimizeForSize);
    F.addFnAttr(Attribute::OptimizeNone);

    // `optnone` requires `noinline`, and `noinline` conflicts with
    // `alwaysinline`.
    F.removeFnAttr(Attribute::AlwaysInline);
    F.addFnAttr(Attribute::NoInline);

    return true;
  }
}; // end of struct SetIntrinsicAttrs
}  // end of anonymous namespace

char SetIntrinsicAttrs::ID = 0;
static RegisterPass<SetIntrinsicAttrs> X(
        "cc-set-intrinsic-attrs",
        "Cheesecloth: set attributes for __llvm__* intrinsics",
        false /* Only looks at CFG */,
        false /* Analysis Pass */);
