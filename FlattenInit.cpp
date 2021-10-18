#include "llvm/Pass.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct LinearTerm {
  Value* Ptr;
  uint64_t Coeff;

  LinearTerm(Value* Ptr, uint64_t Coeff) : Ptr(Ptr), Coeff(Coeff) {}
};

void addTerm(LinearTerm Term, SmallVector<LinearTerm, 1>& Terms) {
  for (auto& OldTerm : Terms) {
    if (OldTerm.Ptr == Term.Ptr) {
      OldTerm.Coeff += Term.Coeff;
      if (OldTerm.Coeff == 0) {
        std::swap(OldTerm, Terms.back());
        Terms.pop_back();
      }
      return;
    }
  }

  // No term with matching `Ptr` was found in `Terms`.
  Terms.push_back(Term);
}

void mergeTerms(SmallVector<LinearTerm, 1>& Dest, SmallVector<LinearTerm, 1> const& Src) {
  DenseMap<Value*, unsigned> PtrIdx;
  for (unsigned I = 0; I < Dest.size(); ++I) {
    PtrIdx.insert(std::make_pair(Dest[I].Ptr, I));
  }
  for (auto const& Term : Src) {
    auto It = PtrIdx.find(Term.Ptr);
    if (It != PtrIdx.end()) {
      unsigned I = It->second;
      Dest[I].Coeff += Term.Coeff;
      if (Dest[I].Coeff == 0) {
        std::swap(Dest[I], Dest.back());
        Dest.pop_back();
        PtrIdx.erase(It);
        PtrIdx.insert(std::make_pair(Dest[I].Ptr, I));
      }
    } else {
      Dest.push_back(Term);
    }
  }
}

/// A linear combination of pointers.
struct LinearPtr {
  SmallVector<LinearTerm, 1> Terms;
  uint64_t Offset;

  LinearPtr() : Terms(), Offset(0) {}
  explicit LinearPtr(Value* Ptr) : Terms { LinearTerm(Ptr, 1) }, Offset(0) {}
  explicit LinearPtr(uint64_t Const) : Terms {}, Offset(Const) {}

  LinearPtr neg() const {
    LinearPtr Out = *this;
    Out.Offset = -Out.Offset;
    for (auto& Term : Out.Terms) {
      Term.Coeff = -Term.Coeff;
    }
    return std::move(Out);
  }

  LinearPtr add(const LinearPtr& Other) const {
    LinearPtr Out;
    Out.Offset = Offset + Other.Offset;

    if (Terms.size() == 1) {
      Out.Terms = Other.Terms;
      addTerm(Terms[0], Out.Terms);
    } else if (Other.Terms.size() == 1) {
      Out.Terms = Terms;
      addTerm(Other.Terms[0], Out.Terms);
    } else {
      Out.Terms = Terms;
      mergeTerms(Out.Terms, Other.Terms);
    }

    return std::move(Out);
  }

  LinearPtr sub(const LinearPtr& Other) const {
    return std::move(add(Other.neg()));
  }

  LinearPtr mulConst(uint64_t Factor) const {
    LinearPtr Out = *this;
    Out.Offset *= Factor;
    for (auto& Term : Out.Terms) {
      Term.Coeff *= Factor;
    }
    return std::move(Out);
  }

  Optional<LinearPtr> mul(const LinearPtr& Other) const {
    if (Terms.size() > 0 && Other.Terms.size() > 0) {
      // Non-linear multiplication.
      return None;
    }

    if (Terms.size() == 0) {
      return Other.mulConst(Offset);
    } else {
      return mulConst(Other.Offset);
    }
  }

  /// Try to interpret this as a base-plus-offset expression, and return the
  /// base.  A base of `nullptr` indicates that no pointers are involved in
  /// this expression (it's an absolute memory address instead).
  Optional<Value*> getBase() const {
    if (Terms.size() == 0) {
      return nullptr;
    } else if (Terms.size() == 1 && Terms[0].Coeff == 1) {
      return Terms[0].Ptr;
    } else {
      return None;
    }
  }
};


enum MemStoreKind {
  /// Store the value `Val` at this location.
  OpStore,
  /// Set `Len` bytes to zero, starting from `Offset`.  This represents a
  /// `memset` call.
  OpZero,
  /// Set `Len` bytes to unknown values, starting from `Offset`.  `memcpy` sets
  /// the affected range to unknown, then copies over all known values from the
  /// source; this represents the fact that copying unknown parts of the source
  /// writes unknown data to the dest.
  OpUnknown,
};

struct MemStore {
  MemStoreKind Kind;
  uint64_t Offset;
  union {
    /// The value stored, for `OpStore`.
    Value* Val;
    /// The number of bytes affected, for `OpZero` and `OpUnknown`.
    uint64_t Len;
  };

  static MemStore CreateStore(uint64_t Offset, Value* Val) {
    MemStore MS = { OpStore, Offset };
    MS.Val = Val;
    return MS;
  }

  static MemStore CreateZero(uint64_t Offset, uint64_t Len) {
    MemStore MS = { OpZero, Offset };
    MS.Len = Len;
    return MS;
  }

  static MemStore CreateUnknown(uint64_t Offset, uint64_t Len) {
    MemStore MS = { OpUnknown, Offset };
    MS.Len = Len;
    return MS;
  }

  uint64_t getStoreSize(DataLayout const& DL) const {
    switch (Kind) {
      case OpStore:
        return DL.getTypeStoreSize(Val->getType());
      case OpZero:
      case OpUnknown:
        return Len;
      default:
        assert(0 && "bad mem op kind");
        return 0;
    }
  }

  uint64_t getEndOffset(DataLayout const& DL) const {
    return Offset + getStoreSize(DL);
  }

  bool overlaps(MemStore const& Other, DataLayout const& DL) const {
    return overlapsRange(Other.Offset, Other.getEndOffset(DL), DL);
  }

  bool overlapsRange(uint64_t Start, uint64_t End, DataLayout const& DL) const {
    return Offset < End && Start < getEndOffset(DL);
  }

  bool containsRange(uint64_t Start, uint64_t End, DataLayout const& DL) const {
    return Offset <= Start && getEndOffset(DL) <= End;
  }
};

struct MemRegion {
  std::vector<MemStore> Ops;
  /// Smallest offset written within this region.
  uint64_t MinOffset;
  /// Largest offset (inclusive) written within this region.  We use an
  /// inclusive range to avoid problems with wraparound.
  uint64_t MaxOffset;

  void pushOp(MemStore Op, DataLayout const& DL) {
    uint64_t End = Op.getEndOffset(DL);
    if (Ops.size() == 0) {
      MinOffset = Op.Offset;
      MaxOffset = End - 1;
    } else {
      if (Op.Offset < MinOffset) {
        MinOffset = Op.Offset;
      }
      if (End - 1 > MaxOffset) {
        MaxOffset = End - 1;
      }
    }
    Ops.push_back(Op);
  }
};

struct Memory {
  DataLayout const& DL;
  DenseMap<Value*, MemRegion> Regions;

  Memory(DataLayout const& DL) : DL(DL) {}

  MemRegion& getRegion(Value* V);
  void initRegion(MemRegion& Region, Value* V);
  void storeConstant(MemRegion& Region, uint64_t Offset, Constant* C);

  /// Attempt to load a value of type `T` from `Offset` within region `Base`.
  /// If the stored value has a different type, a cast will be created in block
  /// `BB` and the cast result will be returned instead.  If the value is
  /// unknown, this method returns null.
  Value* load(Value* Base, uint64_t Offset, Type* T, BasicBlock* BB);
  void store(Value* Base, uint64_t Offset, Value* V);
  void zero(Value* Base, uint64_t Offset, uint64_t Len);
  void setUnknown(Value* Base, uint64_t Offset, uint64_t Len);
  void clear() {
    Regions.clear();
  }
};

MemRegion& Memory::getRegion(Value* V) {
  auto It = Regions.find(V);
  if (It == Regions.end()) {
    MemRegion Region;
    initRegion(Region, V);
    It = Regions.try_emplace(V, std::move(Region)).first;
  }
  return It->second;
}

void Memory::initRegion(MemRegion& Region, Value* V) {
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    if (!GV->hasInitializer()) {
      return;
    }
    Constant* C = GV->getInitializer();
    storeConstant(Region, 0, C);
  }
}

void Memory::storeConstant(MemRegion& Region, uint64_t Offset, Constant* C) {
  if (auto Null = dyn_cast<ConstantPointerNull>(C)) {
    Region.pushOp(MemStore::CreateStore(Offset, C), DL);
  }
  // All other constants are unsupported for now, and initialize the region
  // with unknown values.
}

Value* Memory::load(Value* Base, uint64_t Offset, Type* T, BasicBlock* BB) {
  uint64_t End = Offset + DL.getTypeStoreSize(T);

  auto& Region = getRegion(Base);
  for (auto& Store : make_range(Region.Ops.rbegin(), Region.Ops.rend())) {
    if (Store.overlapsRange(Offset, End, DL)) {
      switch (Store.Kind) {
        case OpStore:
          if (Store.Offset == Offset) {
            Type* SrcTy = Store.Val->getType();
            if (SrcTy == T) {
              return Store.Val;
            } else if (CastInst::isBitOrNoopPointerCastable(SrcTy, T, DL)) {
              Instruction* Inst = CastInst::CreateBitOrPointerCast(Store.Val, T, "loadcast");
              BB->getInstList().push_back(Inst);
              return Inst;
            }
          }
          break;
        case OpZero:
          if (Store.containsRange(Offset, End, DL)) {
            return Constant::getNullValue(T);
          }
          break;
        case OpUnknown:
          break;
      }

      // `Store` overlaps this load, but we failed to obtain an appropriate
      // value, so the result is unknown.
      return nullptr;
    }
  }

  // No store to this region overlaps this value.
  return nullptr;
}

void Memory::store(Value* Base, uint64_t Offset, Value* V) {
  uint64_t End = Offset + DL.getTypeStoreSize(V->getType());

  auto& Region = getRegion(Base);
  // Look for an existing `OpStore` that we can reuse.
  for (auto& Store : make_range(Region.Ops.rbegin(), Region.Ops.rend())) {
    if (Store.overlapsRange(Offset, End, DL)) {
      if (Store.Kind == OpStore && Store.Offset == Offset &&
          Store.getStoreSize(DL) == DL.getTypeStoreSize(V->getType())) {
        Store.Val = V;
        return;
      }

      // `Store` overlaps the new store but can't be reused.
      break;
    }
  }

  Region.pushOp(MemStore::CreateStore(Offset, V), DL);
}

void Memory::zero(Value* Base, uint64_t Offset, uint64_t Len) {
  uint64_t End = Offset + Len;

  auto& Region = getRegion(Base);
  if (Region.Ops.size() > 0 && Offset <= Region.MinOffset && End - 1 >= Region.MaxOffset) {
    // We're overwriting all data currently in the region.
    Region.Ops.clear();
  }
  Region.pushOp(MemStore::CreateZero(Offset, Len), DL);
}

void Memory::setUnknown(Value* Base, uint64_t Offset, uint64_t Len) {
  uint64_t End = Offset + Len;

  auto& Region = getRegion(Base);
  if (Region.Ops.size() > 0 && Offset <= Region.MinOffset && End - 1 >= Region.MaxOffset) {
    // We're overwriting all data currently in the region.  Afterward, the
    // entire region is unknown, so we don't need to add an explicit op.
    Region.Ops.clear();
    return;
  }
  Region.pushOp(MemStore::CreateUnknown(Offset, Len), DL);
}


struct StackFrame {
  Function& Func;
  /// The basic block we were in before the current one.  Used when handling
  /// phi nodes.
  BasicBlock* PrevBB;
  BasicBlock* CurBB;
  BasicBlock::iterator Iter;

  DenseMap<Value*, Value*> Locals;

  /// When this frame is performing a call, this is the old `Value` for the
  /// return value of that call.  When the call retuns, a mapping from
  /// `ReturnValue` to the new return value will be added to `Locals`.
  Value* ReturnValue;
  /// When this frame is performing an `invoke`, this is the block to jump to
  /// if the call throws an exception.
  BasicBlock* UnwindDest;

  /// Create a new stack frame for the given function.  The caller is
  /// responsible for adding argument values to `Locals`.
  StackFrame(Function& Func)
    : Func(Func), PrevBB(nullptr), CurBB(&Func.getEntryBlock()), Iter(CurBB->begin()),
      ReturnValue(nullptr), UnwindDest(nullptr) {}

  void addArgument(unsigned I, Value* V) {
    Argument* Arg = std::next(Func.arg_begin(), I);
    Locals[Arg] = V;
  }

  void enterBlock(BasicBlock* BB);

  Value* mapValue(Value* OldVal);
};

/// Information about the calling context, used for unwinding.
struct UnwindContext {
  /// When the callee returns, it jumps to this new block and (if the return
  /// type is non-void) adds the new return value to the phi node in this
  /// block.
  BasicBlock* ReturnDest;

  /// When the callee `resume`s, it jumps to this new block and adds the resume
  /// value to the phi node in this block.
  BasicBlock* UnwindDest;

  /// List of (old) landingpad instructions of enclosing `invoke` instructions.
  /// All the clauses of these landingpads will be added to landingpads in the
  /// callee.
  SmallVector<LandingPadInst*, 2> LandingPads;

  UnwindContext() : ReturnDest(nullptr), UnwindDest(nullptr), LandingPads() {}
};

struct State {
  Function* NewFunc;
  BasicBlock* NewBB;
  TargetLibraryInfo* TLI;

  DenseMap<Value*, Optional<LinearPtr>> EvalCache;
  Memory Mem;
  std::vector<StackFrame> Stack;

  SimplifyQuery SQ;

  State(Function& OldFunc, StringRef NewName, TargetLibraryInfo* TLI)
    : TLI(TLI), SQ(OldFunc.getParent()->getDataLayout(), TLI),
      Mem(OldFunc.getParent()->getDataLayout()) {
    NewFunc = Function::Create(
        OldFunc.getFunctionType(),
        OldFunc.getLinkage(),
        OldFunc.getAddressSpace(),
        NewName,
        OldFunc.getParent());
    NewFunc->copyAttributesFrom(&OldFunc);

    NewBB = BasicBlock::Create(NewFunc->getContext(), "flatinit", NewFunc);

    StackFrame SF(OldFunc);
    for (unsigned I = 0; I < OldFunc.arg_size(); ++I) {
      SF.addArgument(I, std::next(NewFunc->arg_begin(), I));
    }
    Stack.emplace_back(std::move(SF));
  }

  void run();
  /// Process the next instruction.  Returns `true` on success, or `false` if
  /// the next instruction can't be processed.
  bool step();
  /// Process a call instruction.  Control resumes at `NormalDest` on success,
  /// or at the instruction after the call if `NormalDest` is nullptr.  For
  /// invoke instructions, `UnwindDest` gives the landing pad (for non-invoke
  /// calls, `UnwindDest` is nullptr).
  bool stepCall(CallBase* Call, CallBase* OldCall, BasicBlock* NormalDest, BasicBlock* UnwindDest);

  void unwind();
  void unwindFrame(StackFrame& SF, UnwindContext* PrevUC, UnwindContext* UC);

  Optional<std::pair<Value*, uint64_t>> evalBaseOffset(Value* V);
  LinearPtr* evalPtr(Value* V);
  Optional<LinearPtr> evalPtrImpl(Value* V);
  Optional<LinearPtr> evalPtrConstant(Constant* C);
  Optional<LinearPtr> evalPtrInstruction(Instruction* Inst);
  Optional<LinearPtr> evalPtrOpcode(unsigned Opcode, User* U);
  Optional<LinearPtr> evalPtrGEP(User* U);

  /// Wrapper around llvm::ConstantFoldConstant.
  Constant* constantFoldConstant(Constant* C);
  Constant* constantFoldExtra(Constant* C);
  Constant* constantFoldAlignmentCheck(Constant* C);

  /// Constant fold, plus some extra cases.  Returns nullptr if it was unable
  /// to reduce `Inst` to a constant.
  Constant* constantFoldInstructionExtra(Instruction* Inst);
  Constant* constantFoldNullCheckInst(Instruction* Inst);
};


// Flattening

void State::run() {
  while (step()) {
    // No-op
  }
  errs() << "\n\n\n ===== UNWINDING =====\n\n\n";
  if (Stack.size() > 0) {
    unwind();
  }
}

bool State::step() {
  if (Stack.size() == 0) {
    return false;
  }
  StackFrame& SF = Stack.back();
  Instruction* OldInst = &*SF.Iter;
  errs() << "step: " << *OldInst << "\n";

  // Special case: PHINode operands for the edges not taken may refer to values
  // that have no mapping in SF.Locals, so we must handle it before mapping
  // operands.
  if (auto PHI = dyn_cast<PHINode>(OldInst)) {
    assert(SF.PrevBB != nullptr);
    Value* OldVal = PHI->getIncomingValueForBlock(SF.PrevBB);
    Value* NewVal = SF.mapValue(OldVal);
    SF.Locals[PHI] = NewVal;
    ++SF.Iter;
    return true;
  }

  Instruction* Inst(OldInst->clone());
  Inst->setName(OldInst->getName());
  for (unsigned I = 0; I < Inst->getNumOperands(); ++I) {
    Value* OldVal = Inst->getOperand(I);
    if (!isa<BasicBlock>(OldVal)) {
      Value* NewVal = SF.mapValue(OldVal);
      Inst->setOperand(I, NewVal);
    }
  }


  // Try simplification and/or constant folding.

  Value* Simplified = SimplifyInstruction(Inst, SQ);
  Constant* C = nullptr;
  if (Simplified != nullptr) {
    if (auto SimpleConst = dyn_cast<Constant>(Simplified)) {
      // Constants are handled below, for both simplify and constant folding.
      C = SimpleConst;
    } else if (auto SimpleInst = dyn_cast<Instruction>(Simplified)) {
      // Simplify never creates an instruction; it only ever returns existing
      // ones.
      errs() << "step(simplify): map " << *OldInst << " -> " << *SimpleInst << "\n";
      SF.Locals[OldInst] = SimpleInst;
      Inst->deleteValue();
      ++SF.Iter;
      return true;
    } else {
      errs() << "bad value kind after simplify: " << *Simplified << "\n";
      assert(0 && "bad value kind after simplify");
    }
  }

  if (C == nullptr) {
    if (auto FoldedConst = constantFoldInstructionExtra(Inst)) {
      C = FoldedConst;
    }
  }

  if (C != nullptr) {
    C = constantFoldExtra(C);
    errs() << "step(constant): map " << *OldInst << " -> " << *C << "\n";
    SF.Locals[OldInst] = C;
    Inst->deleteValue();
    ++SF.Iter;
    return true;
  }


  // Instructions with special handling.  If these special cases fail, we fall
  // through and (usually) emit the instruction unchanged.

  // Try to handle function calls.  This can fail if the callee is unknown or
  // not defined, in which case we pass it through as an unknown instruction.
  if (auto Call = dyn_cast<CallInst>(Inst)) {
    if (stepCall(Call, cast<CallInst>(OldInst), nullptr, nullptr)) {
      return true;
    }
  }
  if (auto Invoke = dyn_cast<InvokeInst>(Inst)) {
    if (stepCall(Invoke, cast<InvokeInst>(OldInst), 
          Invoke->getNormalDest(), Invoke->getUnwindDest())) {
      return true;
    }
  }

  if (auto Return = dyn_cast<ReturnInst>(Inst)) {
    if (Stack.size() == 1) {
      // Returning from the top-level function should just be passed through.
      return false;
    }

    // `Return` is the new instruction (with operands already mapped), so it's
    // not invalidated when we deallocate `SF`.
    Value* RetVal = Return->getReturnValue();

    Stack.pop_back();
    StackFrame& PrevSF = Stack.back();
    if (PrevSF.ReturnValue != nullptr && RetVal != nullptr) {
      errs() << "step(return): map " << *PrevSF.ReturnValue << " -> " << *RetVal << "\n";
      PrevSF.Locals[PrevSF.ReturnValue] = RetVal;
    }
    PrevSF.ReturnValue = nullptr;
    PrevSF.UnwindDest = nullptr;

    Inst->deleteValue();
    // `PrevSF.Iter` was already updated when the call was processed.
    return true;
  }

  if (auto Branch = dyn_cast<BranchInst>(Inst)) {
    if (Branch->isUnconditional()) {
      SF.enterBlock(Branch->getSuccessor(0));
      Inst->deleteValue();
      return true;
    } else {
      if (auto ConstCond = dyn_cast<Constant>(Branch->getCondition())) {
        if (ConstCond->isOneValue()) {
          SF.enterBlock(Branch->getSuccessor(0));
        } else {
          SF.enterBlock(Branch->getSuccessor(1));
        }
        Inst->deleteValue();
        return true;
      }
    }
  }

  if (auto Load = dyn_cast<LoadInst>(Inst)) {
    if (auto Ptr = evalBaseOffset(Load->getPointerOperand())) {
      if (Value* V = Mem.load(Ptr->first, Ptr->second, Load->getType(), NewBB)) {
        SF.Locals[OldInst] = V;
        Inst->deleteValue();
        ++SF.Iter;
        return true;
      }
    }
  }


  // We can't pass through unknown terminators, because we don't know what to
  // execute next.
  if (Inst->isTerminator()) {
    errs() << "unwinding due to (old) " << *OldInst << "\n";
    errs() << "unwinding due to (new) " << *Inst << "\n";
    Inst->deleteValue();
    return false;
  }


  // Instructions that pass through, but with some special effect.
  if (auto Alloca = dyn_cast<AllocaInst>(Inst)) {
    EvalCache.try_emplace(Alloca, LinearPtr(Alloca));
  } else if (auto Store = dyn_cast<StoreInst>(Inst)) {
    if (auto Ptr = evalBaseOffset(Store->getPointerOperand())) {
      Mem.store(Ptr->first, Ptr->second, Store->getValueOperand());
    } else {
      errs() << "clear mem: unknown store dest in " << *Store << "\n";
      Mem.clear();
    }
  } else if (auto Call = dyn_cast<CallInst>(Inst)) {
    bool IsAlloc = isAllocationFn(Call, TLI);
    bool IsFree = isFreeCall(Call, TLI);
    if (IsAlloc || IsFree) {
      if (IsAlloc) {
        if (isReallocLikeFn(Call, TLI)) {
          // TODO: return the same pointer as the input, so reads through the
          // new pointer return writes through the old one?
          // TODO: handle realloc(NULL, size) as an alias for malloc(size)?
          assert(0 && "realloc NYI");
        } else {
          EvalCache.try_emplace(Call, LinearPtr(Call));
        }
      }
    }
  } else {
    // TODO: for unknown instructions with Inst->mayWriteToMemory(), clear all known memory
  }

  SF.Locals[OldInst] = Inst;
  NewBB->getInstList().push_back(Inst);
  errs() << "step(unknown): map " << *OldInst << " -> " << *Inst << "\n";
  ++SF.Iter;
  return true;
}

void StackFrame::enterBlock(BasicBlock* BB) {
  PrevBB = CurBB;
  CurBB = BB;
  Iter = BB->begin();
}

Value* StackFrame::mapValue(Value* OldVal) {
  if (isa<Constant>(OldVal)) {
    return OldVal;
  }

  auto It = Locals.find(OldVal);
  if (It == Locals.end()) {
    errs() << "error: no local mapping for value " << *OldVal << "\n";
    assert(0 && "no local mapping for value");
  }
  return It->second;
}

bool State::stepCall(
    CallBase* Call, CallBase* OldCall, BasicBlock* NormalDest, BasicBlock* UnwindDest) {
  // Make sure the callee is known.
  // TODO: handle bitcasted constants here
  Function* Callee = Call->getCalledFunction();
  if (Callee == nullptr) {
    return false;
  }
  if (Callee->isVarArg()) {
    // TODO: handle vararg calls
    return false;
  }
  if (isAllocationFn(Call, TLI) || isFreeCall(Call, TLI)) {
    // Never step into malloc or free functions.  They get special handling in
    // `step` instead.
    return false;
  }
  if (Callee->isDeclaration()) {
    // Function body is not available.
    return false;
  }

  // Advance the current frame past the call.
  StackFrame& SF = Stack.back();
  if (NormalDest == nullptr) {
    ++SF.Iter;
  } else {
    SF.enterBlock(NormalDest);
  }

  SF.ReturnValue = OldCall;
  SF.UnwindDest = UnwindDest;

  // Push a new frame 
  errs() << "enter function " << Callee->getName() << "\n";
  StackFrame NewSF(*Callee);
  for (unsigned I = 0; I < Call->arg_size(); ++I) {
    NewSF.addArgument(I, Call->getArgOperand(I));
  }
  Stack.emplace_back(std::move(NewSF));

  Call->deleteValue();
  return true;
}

Constant* State::constantFoldConstant(Constant* C) {
  Constant* Folded = ConstantFoldConstant(C, NewFunc->getParent()->getDataLayout(), TLI);
  if (Folded != nullptr) {
    return Folded;
  } else {
    return C;
  }
}

/// Apply extra constant folding for certain special cases.
Constant* State::constantFoldExtra(Constant* C) {
  C = constantFoldAlignmentCheck(C);
  C = constantFoldConstant(C);
  return C;
}

/// Fold alignment checks, like `(uintptr_t)ptr & 7`.  These appear in
/// functions like `memcpy` and `strcmp`.  We handle these by increasing the
/// alignment of the declaration of `ptr` so the result has a known value.
Constant* State::constantFoldAlignmentCheck(Constant* C) {
  auto And = dyn_cast<ConstantExpr>(C);
  if (And == nullptr || And->getOpcode() != Instruction::And) {
    return C;
  }

  Constant* Val = nullptr;
  ConstantInt* Mask = nullptr;
  if (Mask = dyn_cast<ConstantInt>(And->getOperand(0))) {
    Val = cast<Constant>(And->getOperand(1));
  } else if (Mask = dyn_cast<ConstantInt>(And->getOperand(1))) {
    Val = cast<Constant>(And->getOperand(0));
  } else {
    return C;
  }

  if (Mask->uge(4096)) {
    return C;
  }
  uint64_t MaskInt = Mask->getZExtValue();
  // Check if MaskInt is one less than a power of two.
  if ((MaskInt & (MaskInt + 1)) != 0) {
    return C;
  }
  uint64_t Align = MaskInt + 1;

  // strcmp tries to be clever, and does two alignment checks at once via
  // `((ptr1 | ptr2) & 7) == 0`.  We handle this by reassociating the
  // expression as `(ptr1 & 7) | (ptr2 & 7)`, then fold it recursively.
  auto ValOr = dyn_cast<ConstantExpr>(Val);
  if (ValOr != nullptr && ValOr->getOpcode() == Instruction::Or) {
    Constant* C0 = constantFoldExtra(ConstantExpr::getAnd(ValOr->getOperand(0), Mask));
    Constant* C1 = constantFoldExtra(ConstantExpr::getAnd(ValOr->getOperand(1), Mask));
    errs() << "found or: " << *ValOr << "\n";
    errs() << "  C0: " << *C0 << "\n";
    errs() << "  C1: " << *C1 << "\n";
    return constantFoldExtra(ConstantExpr::getOr(C0, C1));
  }

  // Evaluate `Val` as a base and offset pair.
  LinearPtr* Ptr = evalPtr(Val);
  if (Ptr == nullptr) {
    return C;
  }
  Optional<Value*> OptBase = Ptr->getBase();
  if (!OptBase.hasValue()) {
    return C;
  }
  Value* Base = OptBase.getValue();

  // If the base is a global variable or function, adjust its alignment to at
  // least `Align`.
  auto Global = dyn_cast<GlobalObject>(Base);
  if (Global == nullptr) {
    return C;
  }

  unsigned OldAlign = Global->getAlignment();
  if (OldAlign < Align) {
    Global->setAlignment(Align);
  }

  // We know `base & mask` is zero, so the result of `(base + offset) & mask`
  // is just `offset & mask`.
  return ConstantInt::get(Mask->getType(), Ptr->Offset & MaskInt);
}

Constant* State::constantFoldInstructionExtra(Instruction* Inst) {
  Constant* C = ConstantFoldInstruction(Inst, NewFunc->getParent()->getDataLayout(), TLI);
  if (C != nullptr) {
    return C;
  }

  C = constantFoldNullCheckInst(Inst);
  if (C != nullptr) {
    return C;
  }

  return nullptr;
}

/// Null checks.  We handle `ptr == NULL` by evaluating a `ptr` to a
/// `LinearPtr` and checking if it's null.
Constant* State::constantFoldNullCheckInst(Instruction* Inst) {
  auto ICmp = dyn_cast<ICmpInst>(Inst);
  if (ICmp == nullptr || !ICmp->isEquality()) {
    return nullptr;
  }

  Value* PtrVal = nullptr;
  Constant* NullConst;
  if ((NullConst = dyn_cast<Constant>(ICmp->getOperand(0))) && NullConst->isNullValue()) {
    PtrVal = ICmp->getOperand(1);
  } else if ((NullConst = dyn_cast<Constant>(ICmp->getOperand(1))) && NullConst->isNullValue()) {
    PtrVal = ICmp->getOperand(0);
  } else {
    return nullptr;
  }

  LinearPtr* Ptr = evalPtr(PtrVal);
  if (Ptr == nullptr) {
    return nullptr;
  }
  // Convert to base and offset form.  For now, we don't try to handle cases
  // where two pointers have been added together.
  Optional<Value*> OptBase = Ptr->getBase();
  if (!OptBase.hasValue()) {
    return nullptr;
  }
  Value* Base = OptBase.getValue();

  bool PtrIsNull = Base == nullptr && Ptr->Offset == 0;
  bool Result;
  if (ICmp->getPredicate() == CmpInst::ICMP_EQ) {
    Result = PtrIsNull;
  } else {
    assert(ICmp->getPredicate() == CmpInst::ICMP_NE);
    Result = !PtrIsNull;
  }

  if (Result) {
    return ConstantInt::getTrue(Inst->getContext());
  } else {
    return ConstantInt::getFalse(Inst->getContext());
  }
}

Optional<std::pair<Value*, uint64_t>> State::evalBaseOffset(Value* V) {
  LinearPtr* LP = evalPtr(V);
  if (LP == nullptr) {
    return None;
  }
  Optional<Value*> Base = LP->getBase();
  if (!Base.hasValue()) {
    return None;
  }
  return std::make_pair(Base.getValue(), LP->Offset);
}

LinearPtr* State::evalPtr(Value* V) {
  auto It = EvalCache.find(V);
  if (It == EvalCache.end()) {
    Optional<LinearPtr> Ptr = evalPtrImpl(V);
    It = EvalCache.try_emplace(V, std::move(Ptr)).first;
  }
  if (It->second.hasValue()) {
    return &It->second.getValue();
  } else {
    return nullptr;
  }
}

Optional<LinearPtr> State::evalPtrImpl(Value* V) {
  if (auto C = dyn_cast<Constant>(V)) {
    return evalPtrConstant(C);
  } else if (auto Inst = dyn_cast<Instruction>(V)) {
    return evalPtrInstruction(Inst);
  } else if (isa<ConstantPointerNull>(V)) {
    return LinearPtr((uint64_t)0);
  } else {
    return None;
  }
}

Optional<LinearPtr> State::evalPtrConstant(Constant* C) {
  if (auto Global = dyn_cast<GlobalObject>(C)) {
    return LinearPtr(C);
  } else if (auto Alias = dyn_cast<GlobalAlias>(C)) {
    return evalPtrConstant(Alias->getAliasee());
  } else if (auto Expr = dyn_cast<ConstantExpr>(C)) {
    return evalPtrOpcode(Expr->getOpcode(), Expr);
  } else {
    return None;
  }
}

Optional<LinearPtr> State::evalPtrInstruction(Instruction* Inst) {
  return evalPtrOpcode(Inst->getOpcode(), Inst);
}

Optional<LinearPtr> State::evalPtrOpcode(unsigned Opcode, User* U) {
  LinearPtr* A = nullptr;
  LinearPtr* B = nullptr;
  switch (Opcode) {
    case Instruction::Add:
      A = evalPtr(U->getOperand(0));
      B = evalPtr(U->getOperand(1));
      if (A == nullptr || B == nullptr) {
        return None;
      }
      return A->add(*B);

    case Instruction::Sub:
      A = evalPtr(U->getOperand(0));
      B = evalPtr(U->getOperand(1));
      if (A == nullptr || B == nullptr) {
        return None;
      }
      return A->sub(*B);

    case Instruction::Mul:
      A = evalPtr(U->getOperand(0));
      B = evalPtr(U->getOperand(1));
      if (A == nullptr || B == nullptr) {
        return None;
      }
      return A->mul(*B);

    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::BitCast:
      A = evalPtr(U->getOperand(0));
      if (A == nullptr) {
        return None;
      }
      return *A;

    case Instruction::GetElementPtr:
      return std::move(evalPtrGEP(U));

    default:
      return None;
  }
}

Optional<LinearPtr> State::evalPtrGEP(User* U) {
  // Get the pointee type for the base of the GEP.  Note the cast can fail,
  // since GEP works on vectors of pointers as well as ordinary pointers.
  auto BasePtrTy = dyn_cast<PointerType>(U->getOperand(0)->getType());
  if (BasePtrTy == nullptr) {
    return None;
  }
  Type* BaseTy = BasePtrTy->getElementType();

  // Get the base pointer as a `LinearPtr`.
  LinearPtr* BaseLP = evalPtr(U->getOperand(0));
  if (BaseLP == nullptr) {
    return None;
  }
  LinearPtr Result = *BaseLP;

  DataLayout const& DL = NewFunc->getParent()->getDataLayout();

  // Apply the first offset, which does pointer arithmeon to the base pointer.
  auto Idx0 = dyn_cast<ConstantInt>(U->getOperand(1));
  if (Idx0 == nullptr) {
    return None;
  }
  Result.Offset += DL.getTypeAllocSize(BaseTy) * Idx0->getSExtValue();

  Type* CurTy = BaseTy;
  for (unsigned I = 2; I < U->getNumOperands(); ++I) {
    auto Idx = dyn_cast<ConstantInt>(U->getOperand(I));
    if (Idx == nullptr) {
      return None;
    }
    int64_t IdxVal = Idx->getSExtValue();

    if (auto StructTy = dyn_cast<StructType>(CurTy)) {
      Result.Offset += DL.getStructLayout(StructTy)->getElementOffset(IdxVal);
      CurTy = StructTy->getElementType(IdxVal);
    } else if (auto ArrayTy = dyn_cast<ArrayType>(CurTy)) {
      Result.Offset += DL.getTypeAllocSize(ArrayTy->getElementType()) * IdxVal;
      CurTy = ArrayTy->getElementType();
    } else {
      return None;
    }
  }

  return std::move(Result);
}


// Unwinding

void State::unwind() {
  assert(Stack.size() > 0);

  // Stack of calling contexts.  There's an `UnwindContext` for every frame
  // except the innermost one.
  std::vector<UnwindContext> UnwindStack;
  for (unsigned I = 0; I < Stack.size() - 1; ++I) {
    StackFrame& SF = Stack[I];
    UnwindContext UC;
    if (I > 0) {
      UC = UnwindStack.back();
    }

    errs() << "build unwind context for " << SF.Func.getName() << "\n";

    UC.ReturnDest = BasicBlock::Create(NewFunc->getContext(), "returndest", NewFunc);
    Type* ReturnType = Stack[I + 1].Func.getReturnType();
    if (!ReturnType->isVoidTy()) {
      Value* PHI = PHINode::Create(ReturnType, 0, "returnval", UC.ReturnDest);
      assert(SF.ReturnValue != nullptr);
      SF.Locals[SF.ReturnValue] = PHI;
      errs() << "  old return value = " << *SF.ReturnValue << "\n";
      errs() << "  new return value = " << *PHI << "\n";
    }

    if (SF.UnwindDest != nullptr) {
      UC.UnwindDest = BasicBlock::Create(NewFunc->getContext(), "unwinddest", NewFunc);

      LandingPadInst* Pad = cast<LandingPadInst>(&SF.UnwindDest->front());
      UC.LandingPads.push_back(Pad);
      Value* PHI = PHINode::Create(Pad->getType(), 0, "unwindval", UC.UnwindDest);

      SF.Locals[Pad] = PHI;
    }

    UnwindStack.push_back(std::move(UC));
  }

  while (Stack.size() > 0) {
    unsigned I = Stack.size() - 1;

    UnwindContext* PrevUC = nullptr;
    if (I > 0 && I - 1 < UnwindStack.size()) {
      PrevUC = &UnwindStack[I - 1];
    }

    UnwindContext* UC = nullptr;
    if (I < UnwindStack.size()) {
      UC = &UnwindStack[I];
    }

    unwindFrame(Stack[I], PrevUC, UC);

    Stack.pop_back();
    if (UC != nullptr) {
      UnwindStack.pop_back();
    }
  }
}

struct UnwindFrameState {
  State& S;
  StackFrame& SF;
  UnwindContext* PrevUC;
  DenseMap<BasicBlock*, BasicBlock*> BlockMap;
  /// List of old blocks that need to be converted.  Every block in this list
  /// will also be present as a key in BlockMap.
  std::vector<BasicBlock*> Pending;
  /// List of all predecessors, as pairs of `OldPred` and `NewPred`.  For each
  /// predecessor, `handlePHINodes` will update the phi nodes of all its
  /// possible successors.
  std::vector<std::pair<BasicBlock*, BasicBlock*>> AllPreds;

  /// Trampolines used for exiting partial blocks.  Partial blocks need special
  /// handling because a separate full clone of the block might also exist if
  /// the original block is inside a loop.  We handle this by splitting the
  /// partial block just before the terminator and placing adding phi nodes for
  /// everything it defines into the trampoline block.
  ///
  /// The key in this map is the old block, and the value is the trampoline
  /// block.  The trampoline block consists of a sequence of PHINodes (one for
  /// each value defined in the old block, in order) followed by a clone of the
  /// old terminator.
  DenseMap<BasicBlock*, BasicBlock*> ExitTrampolines;

  UnwindFrameState(State& S, StackFrame& SF, UnwindContext* PrevUC)
    : S(S), SF(SF), PrevUC(PrevUC) {}

  void emitInst(Instruction* Inst, BasicBlock* Out);
  void emitFullBlock(BasicBlock* BB, BasicBlock* Out);
  void emitPartialBlock(BasicBlock* BB, BasicBlock::iterator Iter, BasicBlock* Out);
  void emitAllBlocks();

  void handlePHINodes();

  /// Map an old value to the corresponding new one.  This returns constants
  /// unchanged, and otherwise maps values through `SF.Locals`.  It aborts if
  /// `OldVal` is not a constant and isn't present in SF.Locals` either.
  Value* mapValue(Value* OldVal);
  /// Map an old basic block to the corresponding new one.  This creates the
  /// new block if needed.
  BasicBlock* mapBlock(BasicBlock* OldBB);
};

void State::unwindFrame(StackFrame& SF, UnwindContext* PrevUC, UnwindContext* UC) {
  errs() << "unwinding out of " << SF.Func.getName() << "\n";
  UnwindFrameState UFS(*this, SF, PrevUC);

  if (UC == nullptr) {
    // This is the innermost frame.  Emit the remaining instructions of the
    // current block directly into `NewBB`.  Other blocks will be generated as
    // needed.
    UFS.emitPartialBlock(SF.CurBB, SF.Iter, NewBB);
  } else {
    UFS.emitPartialBlock(SF.CurBB, SF.Iter, UC->ReturnDest);
  }

  if (SF.UnwindDest != nullptr) {
    errs() << "unwind dest = " << (void*)SF.UnwindDest << "\n";
    errs() << "unwind dest = " << SF.UnwindDest->getName() << "\n";
    SF.UnwindDest->print(errs());
    BasicBlock::iterator Iter = std::next(SF.UnwindDest->begin(), 1);
    UFS.emitPartialBlock(SF.UnwindDest, Iter, UC->UnwindDest);
  }

  UFS.emitAllBlocks();

  UFS.handlePHINodes();
}

void UnwindFrameState::emitInst(Instruction* Inst, BasicBlock* Out) {
  if (auto PHI = dyn_cast<PHINode>(Inst)) {
    // Add a new empty PHI node of the same type.  Incoming values will be
    // populated in a postprocessing pass, based on the edges actually
    // traversed.  (E.g. if we unwind in the middle of one side of a
    // conditional, the new merge block may have only one predecessor instead
    // of two.)
    PHINode* NewPHI = PHINode::Create(
        PHI->getType(), PHI->getNumIncomingValues(), PHI->getName(), Out);
    SF.Locals[PHI] = NewPHI;
    return;
  }

  if (auto Return = dyn_cast<ReturnInst>(Inst)) {
    if (PrevUC != nullptr) {
      BranchInst::Create(PrevUC->ReturnDest, Out);
      Value* OldVal = Return->getReturnValue();
      if (OldVal != nullptr && !OldVal->getType()->isVoidTy()) {
        PHINode* PHI = cast<PHINode>(&*PrevUC->ReturnDest->begin());
        Value* NewVal = mapValue(Return->getReturnValue());
        PHI->addIncoming(NewVal, Out);
      }
      return;
    }
  }

  if (auto Resume = dyn_cast<ResumeInst>(Inst)) {
    if (PrevUC != nullptr && PrevUC->UnwindDest != nullptr) {
      BranchInst::Create(PrevUC->UnwindDest, Out);
      PHINode* PHI = cast<PHINode>(&*PrevUC->UnwindDest->begin());
      Value* NewVal = mapValue(Resume->getValue());
      PHI->addIncoming(NewVal, Out);
      return;
    }
  }

  Instruction* NewInst = Inst->clone();
  NewInst->setName(Inst->getName());
  Out->getInstList().push_back(NewInst);

  errs() << "emitInst " << *Inst << "\n";
  for (unsigned I = 0; I < Inst->getNumOperands(); ++I) {
    Value* OldVal = Inst->getOperand(I);
    if (auto OldBB = dyn_cast<BasicBlock>(OldVal)) {
      BasicBlock* NewBB = mapBlock(OldBB);
      NewInst->setOperand(I, NewBB);
    } else {
      Value* NewVal = mapValue(OldVal);
      errs() << "  operand " << I << ": " << OldVal << " " << *OldVal << " -> " << NewVal << " " << *NewVal << "\n";
      NewInst->setOperand(I, NewVal);
    }
  }

  if (auto LandingPad = dyn_cast<LandingPadInst>(Inst)) {
    if (PrevUC != nullptr) {
      // Accumulate clauses from enclosing landing pads.  According to the LLVM
      // exception handling docs, this is roughly what LLVM does when inlining.
      unsigned TotalNewClauses = 0;
      for (auto EnclosingPad : PrevUC->LandingPads) {
        TotalNewClauses += EnclosingPad->getNumClauses();
      }
      LandingPad->reserveClauses(TotalNewClauses);
      for (auto EnclosingPad : PrevUC->LandingPads) {
        for (unsigned I = 0; I < EnclosingPad->getNumClauses(); ++I) {
          LandingPad->addClause(EnclosingPad->getClause(I));
        }
        if (EnclosingPad->isCleanup()) {
          LandingPad->setCleanup(true);
        }
      }
    }
  }

  if (Inst->isEHPad() || Inst->isExceptionalTerminator()) {
    // Exception-related instructions can only appear inside a function with an
    // EH personality.
    Function* OldFunc = Inst->getFunction();
    assert(OldFunc->hasPersonalityFn());
    Constant* Personality = OldFunc->getPersonalityFn();
    if (!S.NewFunc->hasPersonalityFn()) {
      S.NewFunc->setPersonalityFn(Personality);
    } else {
      assert(S.NewFunc->getPersonalityFn() == Personality &&
          "mismatch between different personality functions");
    }
  }

  SF.Locals[Inst] = NewInst;
}

void UnwindFrameState::emitFullBlock(BasicBlock* BB, BasicBlock* Out) {
  errs() << "emit " << BB->getName() << "\n";
  auto TrampIter = ExitTrampolines.find(BB);
  BasicBlock* TrampBB = nullptr;
  if (TrampIter != ExitTrampolines.end()) {
    TrampBB = TrampIter->second;
  }

  if (TrampBB == nullptr) {
    AllPreds.emplace_back(BB, Out);
  }
  // If TrampBB is non-null, the code that created the trampoline already added
  // it to AllPreds.

  for (Instruction& Inst : *BB) {
    if (TrampBB != nullptr && Inst.isTerminator()) {
      auto Iter = TrampBB->begin();
      for (Instruction& Inst : *BB) {
        if (Inst.isTerminator() || Inst.getType()->isVoidTy()) {
          continue;
        }
        PHINode* PHI = cast<PHINode>(&*Iter);
        ++Iter;
        // In processing this block, all the `SF.Locals` entries from `BB` have
        // been overwritten, replacing the trampoline PHINode with the computed
        // value in the new full block.  We add the computed value to the
        // PHINode, then set the `SF.Locals` entry back to the PHINode so that
        // any later blocks will use that.
        PHI->addIncoming(mapValue(&Inst), Out);
        SF.Locals[&Inst] = PHI;
      }
      BranchInst::Create(TrampBB, Out);
      return;
    }
    emitInst(&Inst, Out);
  }
}

void UnwindFrameState::emitPartialBlock(
    BasicBlock* BB, BasicBlock::iterator Iter, BasicBlock* Out) {
  errs() << "emit (partial) " << BB->getName() << "\n";
  for (Instruction& Inst : iterator_range<BasicBlock::iterator>(Iter, BB->end())) {
    if (Inst.isTerminator()) {
      break;
    }
    emitInst(&Inst, Out);
  }

  // Build trampoline block.
  BasicBlock* TrampBB = BasicBlock::Create(
      S.NewFunc->getContext(), BB->getName() + "_exit", S.NewFunc);
  AllPreds.emplace_back(BB, TrampBB);

  for (Instruction& Inst : *BB) {
    if (Inst.isTerminator() || Inst.getType()->isVoidTy()) {
      continue;
    }
    PHINode* PHI = PHINode::Create(Inst.getType(), 1, Inst.getName(), TrampBB);
    PHI->addIncoming(mapValue(&Inst), Out);
    SF.Locals[&Inst] = PHI;
    errs() << "trampoline: inserted mapping " << &Inst << " " << Inst << " -> " << PHI << " " << *PHI << "\n";
  }

  BranchInst::Create(TrampBB, Out);
  emitInst(BB->getTerminator(), TrampBB);
  ExitTrampolines.insert(std::make_pair(BB, TrampBB));
}

void UnwindFrameState::emitAllBlocks() {
  while (Pending.size() > 0) {
    BasicBlock* BB = Pending.back();
    Pending.pop_back();

    BasicBlock* Out = mapBlock(BB);
    emitFullBlock(BB, Out);
  }
}

void UnwindFrameState::handlePHINodes() {
  for (auto& Entry : AllPreds) {
    BasicBlock* OldPred = Entry.first;
    BasicBlock* NewPred = Entry.second;

    Instruction* Term = OldPred->getTerminator();
    for (unsigned I = 0; I < Term->getNumSuccessors(); ++I) {
      BasicBlock* OldSucc = Term->getSuccessor(I);
      for (PHINode& OldPHI : OldSucc->phis()) {
        PHINode* NewPHI = cast<PHINode>(mapValue(&OldPHI));
        Value* OldVal = OldPHI.getIncomingValueForBlock(OldPred);
        Value* NewVal = mapValue(OldVal);
        NewPHI->addIncoming(NewVal, NewPred);
      }
    }
  }
}

Value* UnwindFrameState::mapValue(Value* OldVal) {
  return SF.mapValue(OldVal);
}

BasicBlock* UnwindFrameState::mapBlock(BasicBlock* OldBB) {
  auto It = BlockMap.find(OldBB);
  if (It != BlockMap.end()) {
    return It->second;
  }

  BasicBlock* NewBB = BasicBlock::Create(S.NewFunc->getContext(), OldBB->getName(), S.NewFunc);
  BlockMap.insert(std::make_pair(OldBB, NewBB));
  Pending.push_back(OldBB);
  return NewBB;
}


// LLVM pass

struct FlattenInit : public ModulePass {
  static char ID;
  FlattenInit() : ModulePass(ID) {}

  bool runOnModule(Module& M) override {
    Function* MainFunc = M.getFunction("main");
    if (MainFunc == nullptr) {
      return false;
    }

    TargetLibraryInfo* TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

    MainFunc->setName("__cc_old_main");
    State S(*MainFunc, "main", TLI);
    S.run();

    for (auto& BB : *S.NewFunc) {
      for (auto& Inst : BB) {
        for (Value* V : Inst.operand_values()) {
          if (auto OpBB = dyn_cast<BasicBlock>(V)) {
            if (OpBB->getParent() != S.NewFunc) {
              errs() << "INVALID: inst " << Inst << " references basic block " << OpBB << " " << OpBB->getName() << " of function " << OpBB->getParent()->getName() << "\n";
            }
          } else if (auto OpInst = dyn_cast<Instruction>(V)) {
            if (OpInst->getFunction() != S.NewFunc) {
              errs() << "INVALID: inst " << Inst << " references instruction " << *OpInst << " of function " << OpInst->getFunction()->getName() << "\n";
            }
          }
        }
      }
    }

    MainFunc->eraseFromParent();

    return true;
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
}; // end of struct FlattenInit
}  // end of anonymous namespace

char FlattenInit::ID = 0;
static RegisterPass<FlattenInit> X(
        "flatten-init",
        "Inline and unroll constant-valued initialization code",
        false /* Only looks at CFG */,
        false /* Analysis Pass */);
