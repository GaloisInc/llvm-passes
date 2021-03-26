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
struct CcInstrument : public FunctionPass {
  static char ID;
  CcInstrument() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    IntegerType* ty_word = IntegerType::get(F.getContext(), WORD_SIZE);
    IntegerType* ty_byte = IntegerType::get(F.getContext(), 8);
    Type* ty_string = ty_byte->getPointerTo();
    Type* trace_arg_tys[1 + TRACE_NUM_ARGS] = {ty_string, 0};
    for (size_t i = 0; i < TRACE_NUM_ARGS; ++i) {
      trace_arg_tys[1 + i] = ty_word;
    }
    Type* ty_void = Type::getVoidTy(F.getContext());
    FunctionType* fty_trace = FunctionType::get(ty_void, trace_arg_tys, false);
    auto func_info = F.getParent()->getOrInsertFunction("__cc_trace_exec", fty_trace);
    Value* func_trace = func_info.getCallee();

    Value* zero = ConstantInt::get(ty_word, 0);

    // Instrument the entry block with a call to `__cc_trace_exec(args...)`
    BasicBlock& entry = F.getEntryBlock();
    Instruction* first_inst = entry.getFirstNonPHI();

    Constant* name_str_array_init = ConstantDataArray::getString(F.getContext(), F.getName());
    Value* name_str_array = new GlobalVariable(*F.getParent(),
        name_str_array_init->getType(), true, GlobalValue::PrivateLinkage,
        name_str_array_init, "trace_name");
    errs() << "name_str_array: "; name_str_array->getType()->print(errs()); errs() << '\n';
    Value* gep_idxs[2] = { zero, zero };
    Value* name_str = GetElementPtrInst::CreateInBounds(
        name_str_array, gep_idxs, "trace_name", first_inst);
    errs() << "name_str: "; name_str->getType()->print(errs()); errs() << '\n';

    Value* trace_args[1 + TRACE_NUM_ARGS] = {name_str, 0};
    for (size_t i = 0; i < TRACE_NUM_ARGS ; ++i) {
      size_t j = 1 + i;
      if (i >= F.arg_size()) {
        trace_args[j] = zero;
        continue;
      }

      Argument* arg = &F.arg_begin()[i];
      Type* ty = arg->getType();
      if (IntegerType* int_ty = dyn_cast<IntegerType>(ty)) {
        if (int_ty->getBitWidth() == WORD_SIZE) {
          trace_args[j] = arg;
        } else if (int_ty->getBitWidth() < WORD_SIZE) {
          trace_args[j] = new ZExtInst(arg, ty_word, "trace_val", first_inst);
        } else {
          trace_args[j] = new TruncInst(arg, ty_word, "trace_val", first_inst);
        }
      } else if (PointerType* ptr_ty = dyn_cast<PointerType>(ty)) {
        trace_args[j] = new PtrToIntInst(arg, ty_word, "trace_val", first_inst);
      } else {
        trace_args[j] = zero;
      }
    }

    errs() << "fty_trace: "; fty_trace->print(errs()); errs() << '\n';

    CallInst::Create(fty_trace, func_trace, trace_args, "", first_inst);

    return true;
  }
}; // end of struct CcInstrument
}  // end of anonymous namespace

char CcInstrument::ID = 0;
static RegisterPass<CcInstrument> X(
        "cc-instrument",
        "Cheesecloth: insert instrumentation",
        false /* Only looks at CFG */,
        false /* Analysis Pass */);
