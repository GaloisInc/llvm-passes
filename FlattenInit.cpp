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
    return Offset <= Start && End <= getEndOffset(DL);
  }

  bool containedByRange(uint64_t Start, uint64_t End, DataLayout const& DL) const {
    return Start <= Offset && getEndOffset(DL) <= End;
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

  /// Load a value of type `T` from `Offset` within region `Base`.  If the
  /// range `Offset .. Offset + Len` (where `Len` is the size in bytes of `T`)
  /// is covered entirely by a single `OpStore`, this returns the value stored
  /// (which may or may not have type `T`); if it's covered entirely by a
  /// single `OpZero`, it returns a zero/null constant of type T.  The second
  /// return value is the byte offset where the requested range begins within
  /// the returned `Value`.  For example, if you store an 8-byte value at
  /// offset 16, then load a 1-byte / value from offset 19 the result of the
  /// `load` will be the stored value and the relative offset 3.  (In the
  /// `OpZero` case, the relative offset is always 0.)
  std::pair<Value*, uint64_t> load(Value* Base, uint64_t Offset, Type* T);
  void store(Value* Base, uint64_t Offset, Value* V);
  void zero(Value* Base, uint64_t Offset, uint64_t Len);
  void setUnknown(Value* Base, uint64_t Offset, uint64_t Len);
  void copy(Value* DestBase, uint64_t DestOffset,
      Value* SrcBase, uint64_t SrcOffset, uint64_t Len);
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
  if (auto Array = dyn_cast<ConstantArray>(C)) {
    Type* ElemTy = Array->getType()->getElementType();
    uint64_t Stride = DL.getTypeAllocSize(ElemTy);
    for (unsigned I = 0; I < Array->getNumOperands(); ++I) {
      storeConstant(Region, Offset + I * Stride, Array->getOperand(I));
    }
  } else if (auto DataArray = dyn_cast<ConstantDataArray>(C)) {
    Type* ElemTy = DataArray->getElementType();
    uint64_t Stride = DL.getTypeAllocSize(ElemTy);
    for (unsigned I = 0; I < DataArray->getNumElements(); ++I) {
      storeConstant(Region, Offset + I * Stride, DataArray->getElementAsConstant(I));
    }
  } else if (auto Struct = dyn_cast<ConstantStruct>(C)) {
    const StructLayout* Layout = DL.getStructLayout(Struct->getType());
    for (unsigned I = 0; I < Struct->getNumOperands(); ++I) {
      uint64_t FieldOffset = Offset + Layout->getElementOffset(I);
      storeConstant(Region, FieldOffset, Struct->getOperand(I));
    }
  } else if (auto AggZero = dyn_cast<ConstantAggregateZero>(C)) {
    uint64_t Len = DL.getTypeStoreSize(AggZero->getType());
    Region.pushOp(MemStore::CreateZero(Offset, Len), DL);
  } else if (C->getType()->isIntOrPtrTy() || C->getType()->isFloatingPointTy()) {
    // Primitive values can be stored directly.
    Region.pushOp(MemStore::CreateStore(Offset, C), DL);
  } else if (isa<UndefValue>(C)) {
    // Do nothing - there are no defined values to store.
  } else {
    // All other constants are unsupported for now, and initialize the region
    // with unknown values.
    errs() << "don't know how to store constant " << *C << "\n";
  }
}

std::pair<Value*, uint64_t> Memory::load(Value* Base, uint64_t Offset, Type* T) {
  uint64_t End = Offset + DL.getTypeStoreSize(T);

  auto& Region = getRegion(Base);
  for (auto& Store : make_range(Region.Ops.rbegin(), Region.Ops.rend())) {
    if (Store.overlapsRange(Offset, End, DL)) {
      switch (Store.Kind) {
        case OpStore:
          if (Store.containsRange(Offset, End, DL)) {
            return std::make_pair(Store.Val, Offset - Store.Offset);
          }
          errs() << "load failed: overlapping store starts at " << Store.Offset <<
            ", at offset " << Offset << " in " << *Base << "\n";
          break;
        case OpZero:
          if (Store.containsRange(Offset, End, DL)) {
            return std::make_pair(Constant::getNullValue(T), 0);
          }
          errs() << "load failed: overlapping zero covers " << Store.Offset << " .. " <<
            Store.getEndOffset(DL) << ", at offset " << Offset << " in " << *Base << "\n";
          break;
        case OpUnknown:
          errs() << "load failed: overlapping unknown, at offset " << Offset <<
            " in " << *Base << "\n";
          break;
      }

      // `Store` overlaps this load, but we failed to obtain an appropriate
      // value, so the result is unknown.
      return std::make_pair(nullptr, 0);
    }
  }

  // No store to this region overlaps this value.
  errs() << "load failed: no overlapping write, at offset " << Offset << " in " << *Base << "\n";
  return std::make_pair(nullptr, 0);
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

void Memory::copy(Value* DestBase, uint64_t DestOffset,
    Value* SrcBase, uint64_t SrcOffset, uint64_t Len) {
  errs() << "copy: dest " << *DestBase << " +" << DestOffset <<
    ", src " << *SrcBase << " +" << SrcOffset <<
    ", length " << Len << "\n";
  assert(SrcBase != DestBase && "copy within a single region is NYI");

  setUnknown(DestBase, DestOffset, Len);

  auto& DestRegion = getRegion(DestBase);
  auto& SrcRegion = getRegion(SrcBase);
  uint64_t SrcEnd = SrcOffset + Len;
  for (auto& Store : SrcRegion.Ops) {
    if (Store.containedByRange(SrcOffset, SrcEnd, DL)) {
      MemStore NewStore = Store;
      NewStore.Offset = Store.Offset - SrcOffset + DestOffset;
      DestRegion.pushOp(std::move(NewStore), DL);
    } else if (Store.overlapsRange(SrcOffset, SrcEnd, DL)) {
      if (Store.Kind == OpZero || Store.Kind == OpUnknown) {
        uint64_t Start = std::max(SrcOffset, Store.Offset);
        uint64_t End = std::min(SrcEnd, Store.getEndOffset(DL));

        MemStore NewStore = Store;
        NewStore.Offset = Start - SrcOffset + DestOffset;
        NewStore.Len = End - Start;
        DestRegion.pushOp(std::move(NewStore), DL);
      } else {
        errs() << "copy partially failed: store at " << Store.Offset <<
            " only partially overlaps source region " << SrcOffset << " .. " << SrcEnd <<
            ", when copying from " << *SrcBase << " +" << SrcOffset <<
            " to " << *DestBase << " +" << DestOffset << "\n";
      }
    }
  }
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
  /// When this frame is performing a call, this may be non-null to indicate
  /// that the return value should be cast to a different type after returning.
  /// This is used to handle casts through bitcasted function pointers.
  Type* CastReturnType;
  /// When this frame is performing an `invoke`, this is the block to jump to
  /// if the call throws an exception.
  BasicBlock* UnwindDest;

  /// Create a new stack frame for the given function.  The caller is
  /// responsible for adding argument values to `Locals`.
  StackFrame(Function& Func)
    : Func(Func), PrevBB(nullptr), CurBB(&Func.getEntryBlock()), Iter(CurBB->begin()),
      ReturnValue(nullptr), CastReturnType(nullptr), UnwindDest(nullptr) {}

  void addArgument(unsigned I, Value* V) {
    Argument* Arg = std::next(Func.arg_begin(), I);
    Locals[Arg] = V;
  }

  void enterBlock(BasicBlock* BB);
  /// Enter `BB`, or if `BB` is null, advance to the next instruction.
  ///
  /// This is mostly used for call handling, where some kinds of calls are
  /// terminators with an explicit next block, while others are ordinary
  /// instructions.
  void advance(BasicBlock* BB);

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
  DataLayout const& DL;
  Function* NewFunc;
  BasicBlock* NewBB;
  TargetLibraryInfo* TLI;

  DenseMap<Value*, Optional<LinearPtr>> EvalCache;
  Memory Mem;
  std::vector<StackFrame> Stack;

  SimplifyQuery SQ;

  State(Function& OldFunc, StringRef NewName, TargetLibraryInfo* TLI)
    : DL(OldFunc.getParent()->getDataLayout()), TLI(TLI), SQ(DL, TLI), Mem(DL) {
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
  bool stepMemset(CallBase* Call);
  bool stepMemcpy(CallBase* Call);
  bool stepMalloc(CallBase* Call, CallBase* OldCall);
  bool stepRealloc(CallBase* Call, CallBase* OldCall);
  bool stepFree(CallBase* Call, CallBase* OldCall);
  Value* stepAlloca(AllocaInst* Alloca);

  // Try to convert the result of `Memory::load` to a value of type `T`.
  // Returns null if the conversion isn't possible.
  Value* convertLoadResult(Value* V, uint64_t Offset, Type* T);
  // Allocate a new `GlobalVariable` containing an array.  This is a helper
  // function for `stepMalloc` and such.
  std::pair<GlobalVariable*, Constant*> allocateGlobal(
      Type* ElemTy, uint64_t Count, uint64_t Align, bool Zero);

  void unwind();
  void unwindFrame(StackFrame& SF, UnwindContext* PrevUC, UnwindContext* UC);

  Optional<std::pair<Value*, uint64_t>> evalBaseOffset(Value* V);
  LinearPtr* evalPtr(Value* V);
  Optional<LinearPtr> evalPtrImpl(Value* V);
  Optional<LinearPtr> evalPtrConstant(Constant* C);
  Optional<LinearPtr> evalPtrInstruction(Instruction* Inst);
  Optional<LinearPtr> evalPtrOpcode(unsigned Opcode, User* U);
  Optional<LinearPtr> evalPtrGEP(User* U);

  /// General-purpose instruction simplification and constant folding.  This
  /// may return `Inst` itself, some other existing instruction, or a constant.
  Value* foldInst(Instruction* Inst);
  /// Apply `foldInst` to `Inst`, emit `Inst` into `NewBB` if needed, and
  /// return the resulting value.  `Inst` will be deleted automatically if this
  /// is appropriate (so `Inst` should not be used afterward - but the returned
  /// `Value*` is guaranteed to evaluate to the same value as `Inst`).  This
  /// method should be used when synthesizing a new instruction from scratch.
  Value* foldAndEmitInst(Instruction* Inst);

  /// Wrapper around llvm::ConstantFoldConstant.
  Constant* constantFoldConstant(Constant* C);
  Constant* constantFoldExtra(Constant* C);
  Constant* constantFoldAlignmentCheckAnd(Constant* C);
  Constant* constantFoldAlignmentCheckURem(Constant* C);
  Constant* constantFoldAlignmentCheckPtr(Constant* C, LinearPtr* LP, uint64_t Align);

  /// Constant fold, plus some extra cases.  Returns nullptr if it was unable
  /// to reduce `Inst` to a constant.
  Constant* constantFoldInstructionExtra(Instruction* Inst);
  Constant* constantFoldNullCheckInst(Instruction* Inst);
  Constant* constantFoldPointerCompare(Instruction* Inst);
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

  Value* Folded = foldInst(Inst);
  if (Folded != Inst) {
    SF.Locals[OldInst] = Folded;
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

    errs() << "exit function " << SF.Func.getName() << "\n";

    // `Return` is the new instruction (with operands already mapped), so it's
    // not invalidated when we deallocate `SF`.
    Value* RetVal = Return->getReturnValue();

    Stack.pop_back();
    StackFrame& PrevSF = Stack.back();
    if (PrevSF.ReturnValue != nullptr && RetVal != nullptr) {
      if (PrevSF.CastReturnType != nullptr) {
        RetVal = foldAndEmitInst(CastInst::CreateBitOrPointerCast(
              RetVal, PrevSF.CastReturnType, "returncast"));
      }
      PrevSF.Locals[PrevSF.ReturnValue] = RetVal;
    }
    PrevSF.ReturnValue = nullptr;
    PrevSF.CastReturnType = nullptr;
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
  if (auto Switch = dyn_cast<SwitchInst>(Inst)) {
    if (auto ConstCond = dyn_cast<ConstantInt>(Switch->getCondition())) {
      SF.enterBlock(Switch->findCaseValue(ConstCond)->getCaseSuccessor());
      Inst->deleteValue();
      return true;
    }
  }

  if (auto Load = dyn_cast<LoadInst>(Inst)) {
    if (auto Ptr = evalBaseOffset(Load->getPointerOperand())) {
      auto Result = Mem.load(Ptr->first, Ptr->second, Load->getType());
      if (Result.first != nullptr) {
        if (auto V = convertLoadResult(Result.first, Result.second, Load->getType())) {
          SF.Locals[OldInst] = V;
          Inst->deleteValue();
          ++SF.Iter;
          return true;
        }
      }
    } else {
      errs() << "failed to evaluate to a pointer expression: " << *Load->getPointerOperand() << "\n";
    }
  }


  // Allocation instructions.
  if (auto Alloca = dyn_cast<AllocaInst>(Inst)) {
    if (Value* V = stepAlloca(Alloca)) {
      SF.Locals[OldInst] = V;
      Inst->deleteValue();
      ++SF.Iter;
      return true;
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
  if (auto Store = dyn_cast<StoreInst>(Inst)) {
    if (auto Ptr = evalBaseOffset(Store->getPointerOperand())) {
      Mem.store(Ptr->first, Ptr->second, Store->getValueOperand());
    } else {
      errs() << "clear mem: unknown store dest in " << *Store << "\n";
      Mem.clear();
    }
  } else {
    // TODO: for unknown instructions with Inst->mayWriteToMemory(), clear all known memory
  }

  SF.Locals[OldInst] = Inst;
  NewBB->getInstList().push_back(Inst);
  ++SF.Iter;
  return true;
}

Value* State::convertLoadResult(Value* V, uint64_t Offset, Type* T) {
  Type* SrcTy = V->getType();
  if (SrcTy == T) {
    return V;
  }

  // Check if we can just do a simple cast.
  if (CastInst::isBitOrNoopPointerCastable(SrcTy, T, DL)) {
    return foldAndEmitInst(CastInst::CreateBitOrPointerCast(V, T, "loadcast"));
  }

  // Complex case: extracting a byte from a larger value.
  if (auto IntTy = dyn_cast<IntegerType>(T)) {
    if (SrcTy->isIntOrPtrTy()) {
      Value* Src = V;
      if (auto SrcPtrTy = dyn_cast<PointerType>(SrcTy)) {
        SrcTy = DL.getIntPtrType(SrcTy);
        Src = foldAndEmitInst(new PtrToIntInst(Src, SrcTy, "loadcast"));
      }
      if (Offset > 0) {
        auto ShiftAmount = ConstantInt::get(SrcTy, 8 * Offset);
        Src = foldAndEmitInst(BinaryOperator::Create(
              Instruction::LShr, Src, ShiftAmount, "loadshift"));
      }
      return foldAndEmitInst(CastInst::Create(Instruction::Trunc, Src, T, "loadtrunc"));
    }
  }

  errs() << "load failed: can't extract " << *T << " from offset " << Offset <<
    " of " << *V << "\n";
  return nullptr;
}

void StackFrame::enterBlock(BasicBlock* BB) {
  PrevBB = CurBB;
  CurBB = BB;
  Iter = BB->begin();
}

void StackFrame::advance(BasicBlock* BB) {
  if (BB == nullptr) {
    ++Iter;
  } else {
    enterBlock(BB);
  }
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

Function* getCallee(Value* V) {
  if (auto Func = dyn_cast<Function>(V)) {
    return Func;
  } else if (auto Expr = dyn_cast<ConstantExpr>(V)) {
    if (Expr->getOpcode() == Instruction::BitCast) {
      return getCallee(Expr->getOperand(0));
    }
  } else if (auto Cast = dyn_cast<CastInst>(V)) {
    if (Cast->getOpcode() == Instruction::BitCast) {
      return getCallee(Cast->getOperand(0));
    }
  }
  return nullptr;
}

bool State::stepCall(
    CallBase* Call, CallBase* OldCall, BasicBlock* NormalDest, BasicBlock* UnwindDest) {
  // Never step into malloc or free functions.  If the special handler fails to
  // handle it, bail out.
  if (isReallocLikeFn(Call, TLI)) {
    if (stepRealloc(Call, OldCall)) {
      Stack.back().advance(NormalDest);
      return true;
    }
    return false;
  }
  if (isAllocationFn(Call, TLI)) {
    if (stepMalloc(Call, OldCall)) {
      Stack.back().advance(NormalDest);
      return true;
    }
    return false;
  }
  if (isFreeCall(Call, TLI)) {
    if (stepFree(Call, OldCall)) {
      Stack.back().advance(NormalDest);
      return true;
    }
    return false;
  }

  // Get the callee and the signature used for the call (which may differ from
  // the callee's actual signature, if the callee has been bitcasted).
  Function* Callee = getCallee(Call->getCalledOperand());
  FunctionType* CallSig = Call->getFunctionType();

  if (Callee->isIntrinsic() && CallSig == Callee->getFunctionType()) {
    // Each of `stepFoo` functions called here (e.g. `stepMemset`) is expected
    // to dispose of `Call` appropriately (either deleting it or adding it to
    // `NewBB`) if it returns `true`.
    switch (Callee->getIntrinsicID()) {
      case Intrinsic::memset:
        if (stepMemset(Call)) {
          Stack.back().advance(NormalDest);
          return true;
        }
        break;
      case Intrinsic::memcpy:
        if (stepMemcpy(Call)) {
          Stack.back().advance(NormalDest);
          return true;
        }
        break;
    }
  }

  if (Callee->isDeclaration()) {
    // Function body is not available.
    return false;
  }

  if (Callee->getFunctionType() != CallSig) {
    // The callee has been bitcasted to another signature.  Check for
    // compatibility.
    FunctionType* DefSig = Callee->getFunctionType();
    if (Call->arg_size() < DefSig->getNumParams()) {
      // Calling with too few arguments is not supported.
      return false;
    }
    for (unsigned I = 0; I < DefSig->getNumParams(); ++I) {
      Type* CallTy = Call->getArgOperand(I)->getType();
      Type* DefTy = DefSig->getParamType(I);
      if (CallTy != DefTy && !CastInst::isBitOrNoopPointerCastable(CallTy, DefTy, DL)) {
        // Argument types are not convertible.
        return false;
      }
    }

    Type* CallReturnTy = Call->getType();
    Type* DefReturnTy = DefSig->getReturnType();
    if (CallReturnTy != DefReturnTy && !CallReturnTy->isVoidTy() &&
        !CastInst::isBitOrNoopPointerCastable(DefReturnTy, CallReturnTy, DL)) {
      // Return type is non-void and not convertible.
      return false;
    }
  }

  // Advance the current frame past the call.
  StackFrame& SF = Stack.back();
  SF.advance(NormalDest);

  SF.ReturnValue = OldCall;
  if (Callee->getFunctionType()->getReturnType() != Call->getType()) {
    SF.CastReturnType = Call->getType();
  }
  SF.UnwindDest = UnwindDest;

  // Push a new frame 
  errs() << "enter function " << Callee->getName() << "\n";
  StackFrame NewSF(*Callee);
  for (unsigned I = 0; I < Call->arg_size(); ++I) {
    Value* V = Call->getArgOperand(I);
    Type* ExpectTy = Callee->getFunctionType()->getParamType(I);
    if (V->getType() != ExpectTy) {
      V = foldAndEmitInst(CastInst::CreateBitOrPointerCast(V, ExpectTy, "callcast"));
    }
    NewSF.addArgument(I, V);
  }
  Stack.emplace_back(std::move(NewSF));

  Call->deleteValue();
  return true;
}

bool State::stepMemset(CallBase* Call) {
  auto DestPtr = evalBaseOffset(Call->getOperand(0));
  if (!DestPtr) {
    return false;
  }

  auto ValConst = dyn_cast<Constant>(Call->getOperand(1));
  if (ValConst == nullptr) {
    return false;
  }
  if (!ValConst->isZeroValue()) {
    return false;
  }

  auto LenConst = dyn_cast<ConstantInt>(Call->getOperand(2));
  if (LenConst == nullptr) {
    return false;
  }
  uint64_t Len = LenConst->getZExtValue();

  // We ignore operand 3, the `isvolatile` flag.

  Mem.zero(DestPtr->first, DestPtr->second, Len);

  NewBB->getInstList().push_back(Call);
  return true;
}

bool State::stepMemcpy(CallBase* Call) {
  auto DestPtr = evalBaseOffset(Call->getOperand(0));
  if (!DestPtr) {
    return false;
  }

  auto SrcPtr = evalBaseOffset(Call->getOperand(1));
  if (!SrcPtr) {
    return false;
  }

  auto LenConst = dyn_cast<ConstantInt>(Call->getOperand(2));
  if (LenConst == nullptr) {
    return false;
  }
  uint64_t Len = LenConst->getZExtValue();

  // We ignore operand 3, the `isvolatile` flag.

  if (SrcPtr->first == DestPtr->first) {
    errs() << "memcpy failed: copy within a single region is NYI, in " << *SrcPtr->first << "\n";
  }

  Mem.copy(DestPtr->first, DestPtr->second, SrcPtr->first, SrcPtr->second, Len);

  NewBB->getInstList().push_back(Call);
  return true;
}

std::pair<GlobalVariable*, Constant*> State::allocateGlobal(
    Type* ElemTy, uint64_t Count, uint64_t Align, bool Zero) {
  Type* Ty = ArrayType::get(ElemTy, Count);
  Constant* Init;
  if (Zero) {
    Init = ConstantAggregateZero::get(Ty);
  } else {
    Init = UndefValue::get(Ty);
  }
  GlobalVariable* GV = new GlobalVariable(
      *NewFunc->getParent(),
      Ty,
      false,  // not constant
      GlobalValue::InternalLinkage,
      Init,
      "alloc");
  GV->setAlignment(Align);
  Constant* GEPIdxs[2] = {
    ConstantInt::get(IntegerType::get(NewFunc->getContext(), 64), 0),
    ConstantInt::get(IntegerType::get(NewFunc->getContext(), 32), 0),
  };
  Constant* Ptr = ConstantExpr::getInBoundsGetElementPtr(Ty, GV, GEPIdxs);
  return std::make_pair(GV, Ptr);
}

bool State::stepMalloc(CallBase* Call, CallBase* OldCall) {
  Function* Callee = Call->getCalledFunction();
  if (Callee == nullptr) {
    return false;
  }
  LibFunc LF;
  if (!TLI->getLibFunc(*Callee, LF)) {
    return false;
  }

  // The size of the allocated region.  This is always set.
  Value* SizeV;
  // If set, the size of the allocated region is the product of `Size` and
  // `Size2`, instead of just `Size`.
  Value* Size2V = nullptr;
  // If set, the allocated region is aligned to a multiple of this amount.
  Value* AlignV = nullptr;
  // If set, the allocated region is zeroed.
  bool Zero = false;
  // Out-pointer argument.  If set, the function stores the allocated pointer
  // here and returns zero, instead of returning the pointer directly.
  Value* OutPtrV = nullptr;

  switch (LF) {
    case LibFunc_malloc:
      SizeV = Call->getOperand(0);
      break;
    case LibFunc_calloc:
      SizeV = Call->getOperand(0);
      Size2V = Call->getOperand(1);
      Zero = true;
      break;
    case LibFunc_posix_memalign:
      OutPtrV = Call->getOperand(0);
      AlignV = Call->getOperand(1);
      SizeV = Call->getOperand(2);
      break;
    default:
      errs() << "failed to handle malloc: unsupported callee in " << *Call << "\n";
      return false;
  }

  // Convert the arguments.
  uint64_t Size;
  if (auto Int = dyn_cast<ConstantInt>(SizeV)) {
    Size = Int->getZExtValue();
  } else {
    errs() << "malloc failed: non-constant size, in " << *Call << "\n";
    return false;
  }

  uint64_t Size2 = 1;
  if (Size2V != nullptr) {
    if (auto Int = dyn_cast<ConstantInt>(Size2V)) {
      Size2 = Int->getZExtValue();
    } else {
      errs() << "malloc failed: non-constant size2, in " << *Call << "\n";
      return false;
    }
  }

  // If `Size2` is set, set `Size = Size * Size2`, but check for overflow.
  if (Size2 != 1) {
    uint64_t Product;
    if (!__builtin_mul_overflow(Size, Size2, &Product)) {
      errs() << "malloc failed: size product overflowed, in " << *Call << "\n";
      return false;
    }
    Size = Product;
  }

  uint64_t Align;
  if (AlignV != nullptr) {
    if (auto Int = dyn_cast<ConstantInt>(AlignV)) {
      Align = Int->getZExtValue();
    } else {
      errs() << "malloc failed: non-constant align, in " << *Call << "\n";
      return false;
    }
  } else {
    Align = 8;
  }

  Optional<std::pair<Value*, uint64_t>> OutPtr = None;
  if (OutPtrV != nullptr) {
    OutPtr = evalBaseOffset(OutPtrV);
    if (!OutPtr) {
      errs() << "malloc failed: failed to evaluate out ptr, in " << *Call << "\n";
      return false;
    }
  }

  // Create the allocation.
  auto Global = allocateGlobal(IntegerType::get(NewFunc->getContext(), 8), Size, Align, Zero);
  GlobalVariable* GV = Global.first;
  Value* Ptr = Global.second;

  errs() << "malloc: allocated " << *GV << " for " << *Call << "\n";

  Value* ReturnValue;
  if (OutPtr) {
    Mem.store(OutPtr->first, OutPtr->second, Ptr);
    ReturnValue = ConstantInt::get(Call->getType(), 0);
  } else {
    ReturnValue = Ptr;
  }

  // Add the allocation to memory.  Note `GV` is the base, not `Ptr` itself.
  EvalCache[GV] = LinearPtr(GV);

  Stack.back().Locals[OldCall] = ReturnValue;
  Call->deleteValue();
  return true;
}

bool State::stepRealloc(CallBase* Call, CallBase* OldCall) {
  Function* Callee = Call->getCalledFunction();
  if (Callee == nullptr) {
    return false;
  }
  LibFunc LF;
  if (!TLI->getLibFunc(*Callee, LF)) {
    return false;
  }

  Value* OldPtrV;
  Value* SizeV;

  switch (LF) {
    case LibFunc_realloc:
      OldPtrV = Call->getOperand(0);
      SizeV = Call->getOperand(1);
      break;
    default:
      errs() << "failed to handle realloc: unsupported callee in " << *Call << "\n";
      return false;
  }

  // Convert the arguments.
  auto OldPtr = evalBaseOffset(OldPtrV);
  if (!OldPtr) {
    errs() << "realloc failed: failed to evaluate ptr, in " << *Call << "\n";
    return false;
  }

  uint64_t Size;
  if (auto Int = dyn_cast<ConstantInt>(SizeV)) {
    Size = Int->getZExtValue();
  } else {
    errs() << "realloc failed: non-constant size, in " << *Call << "\n";
    return false;
  }

  uint64_t MemcpySize = Size;
  if (OldPtr->second != 0) {
    errs() << "realloc failed: old pointer has nonzero offset, in " << *Call << "\n";
    return false;
  } else if (OldPtr->first == nullptr) {
    // If the old pointer is NULL, there's nothing to copy.
    MemcpySize = 0;
  } else if (auto OldGV = dyn_cast<GlobalVariable>(OldPtr->first)) {
    uint64_t GVSize = DL.getTypeAllocSize(OldGV->getValueType());
    if (GVSize < MemcpySize) {
      MemcpySize = GVSize;
    }
  } else {
    errs() << "realloc failed: couldn't get size of old pointer, in " << *Call << "\n";
    errs() << "  old pointer is " << *OldPtr->first << "\n";
    return false;
  }

  uint64_t Align = 8;
  bool Zero = false;

  // Create the allocation.
  auto Global = allocateGlobal(IntegerType::get(NewFunc->getContext(), 8), Size, Align, Zero);
  GlobalVariable* GV = Global.first;
  Value* Ptr = Global.second;

  errs() << "realloc: reallocated " << *OldPtr->first << " as " << *GV << " for " << *Call << "\n";

  // Add the allocation to memory, and copy over the old contents.  Note `GV`
  // is the base, not `Ptr` itself.
  EvalCache[GV] = LinearPtr(GV);
  if (MemcpySize != 0) {
    Mem.copy(GV, 0, OldPtr->first, 0, MemcpySize);

    // Emit a call to `memcpy` to initialize the new global from the old.
    Type* SizeTy = IntegerType::get(NewFunc->getContext(), 64);
    Type* MemcpyTys[3] = { Ptr->getType(), OldPtrV->getType(), SizeTy };
    Function* MemcpyFunc = Intrinsic::getDeclaration(
        NewFunc->getParent(), Intrinsic::memcpy, MemcpyTys);
    Value* MemcpyArgs[4] = {
      Ptr,
      OldPtrV,
      ConstantInt::get(SizeTy, MemcpySize),
      ConstantInt::getFalse(NewFunc->getContext())
    };
    CallInst::Create(MemcpyFunc->getFunctionType(), MemcpyFunc, MemcpyArgs,
        "", NewBB);
  }

  Stack.back().Locals[OldCall] = Ptr;
  Call->deleteValue();
  return true;
}

bool State::stepFree(CallBase* Call, CallBase* OldCall) {
  Function* Callee = Call->getCalledFunction();
  if (Callee == nullptr) {
    return false;
  }
  LibFunc LF;
  if (!TLI->getLibFunc(*Callee, LF)) {
    return false;
  }

  switch (LF) {
    case LibFunc_free:
      break;
    default:
      errs() << "failed to handle free: unsupported callee in " << *Call << "\n";
      return false;
  }

  // free(ptr) is a no-op in this model.
  Call->deleteValue();
  return true;
}

Value* State::stepAlloca(AllocaInst* Alloca) {
  uint64_t Count;
  if (auto Int = dyn_cast<ConstantInt>(Alloca->getArraySize())) {
    Count = Int->getZExtValue();
  } else {
    errs() << "alloca failed: non-constant size, in " << *Alloca << "\n";
    return nullptr;
  }

  uint64_t Align = 8;
  bool Zero = false;

  auto Global = allocateGlobal(Alloca->getAllocatedType(), Count, Align, Zero);
  GlobalVariable* GV = Global.first;
  Value* Ptr = Global.second;

  EvalCache[GV] = LinearPtr(GV);
  return Ptr;
}

Value* State::foldInst(Instruction* Inst) {
  Constant* C = nullptr;
  Value* Simplified = SimplifyInstruction(Inst, SQ);
  if (Simplified == nullptr) {
    // `Inst` can't be simplified any further.  Try folding it to a constant
    // instead.
    C = constantFoldInstructionExtra(Inst);
  } else if (isa<Instruction>(Simplified)) {
    // Simplify never creates an instruction; it only ever returns existing
    // ones.  The existing instruction was already processed by constant
    // folding, so we know it can't be folded further.
    return Simplified;
  } else if (auto SimpleConst = dyn_cast<Constant>(Simplified)) {
    // Constants are handled below, for both simplify and constant folding.
    C = SimpleConst;
  } else {
    errs() << "bad value kind after simplify: " << *Simplified << "\n";
    assert(0 && "bad value kind after simplify");
  }

  if (C != nullptr) {
    C = constantFoldExtra(C);
    return C;
  }

  return Inst;
}

Value* State::foldAndEmitInst(Instruction* Inst) {
  Value* V = foldInst(Inst);
  if (V == Inst) {
    NewBB->getInstList().push_back(Inst);
  } else {
    Inst->deleteValue();
  }
  return V;
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
  C = constantFoldAlignmentCheckAnd(C);
  C = constantFoldAlignmentCheckURem(C);
  C = constantFoldConstant(C);
  return C;
}

/// Fold alignment checks, like `(uintptr_t)ptr & 7`.  These appear in
/// functions like `memcpy` and `strcmp`.  We handle these by increasing the
/// alignment of the declaration of `ptr` so the result has a known value.
Constant* State::constantFoldAlignmentCheckAnd(Constant* C) {
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
    return constantFoldExtra(ConstantExpr::getOr(C0, C1));
  }

  // Evaluate `Val` as a pointer expression.
  LinearPtr* Ptr = evalPtr(Val);
  if (Ptr == nullptr) {
    return C;
  }
  return constantFoldAlignmentCheckPtr(C, Ptr, Align);
}

Constant* State::constantFoldAlignmentCheckURem(Constant* C) {
  auto URem = dyn_cast<ConstantExpr>(C);
  if (URem == nullptr || URem->getOpcode() != Instruction::URem) {
    return C;
  }

  errs() << "looking at urem: " << *C << "\n";

  Constant* Val = URem->getOperand(0);;
  ConstantInt* AlignConst = dyn_cast<ConstantInt>(URem->getOperand(1));
  if (AlignConst == nullptr) {
    return C;
  }

  if (AlignConst->uge(4096 + 1)) {
    return C;
  }
  uint64_t Align = AlignConst->getZExtValue();
  // Check if Align is a power of two.
  if ((Align & (Align - 1)) != 0) {
    return C;
  }

  // Evaluate `Val` as a pointer expression.
  LinearPtr* Ptr = evalPtr(Val);
  if (Ptr == nullptr) {
    errs() << "failed to evaluate to a pointer: " << *Val << "\n";
    return C;
  }
  return constantFoldAlignmentCheckPtr(C, Ptr, Align);
}

Constant* State::constantFoldAlignmentCheckPtr(Constant* C, LinearPtr* LP, uint64_t Align) {
  for (auto& Term : LP->Terms) {
    // If the base is a global variable or function, adjust its alignment to at
    // least `Align`.
    auto Global = dyn_cast<GlobalObject>(Term.Ptr);
    if (Global == nullptr) {
      return C;
    }

    unsigned OldAlign = Global->getAlignment();
    if (OldAlign < Align) {
      Global->setAlignment(Align);
    }
  }

  // Each term's base pointer is now equal to zero mod `Align`, so the sum of
  // all terms is also equal to zero mod `Align`.  The overall result is now
  // equal to `LP->Offset % Align`.
  return ConstantInt::get(C->getType(), LP->Offset & (Align - 1));
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

  C = constantFoldPointerCompare(Inst);
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

Constant* State::constantFoldPointerCompare(Instruction* Inst) {
  auto ICmp = dyn_cast<ICmpInst>(Inst);
  if (ICmp == nullptr || !ICmp->isEquality()) {
    return nullptr;
  }

  auto Ptr1 = evalBaseOffset(ICmp->getOperand(0));
  if (!Ptr1) {
    return nullptr;
  }
  auto Ptr2 = evalBaseOffset(ICmp->getOperand(1));
  if (!Ptr2) {
    return nullptr;
  }

  // Two pointers are equal if they have the same base and the same offset.
  bool Result = Ptr1->first == Ptr2->first && Ptr1->second == Ptr2->second;
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
  } else if (auto Int = dyn_cast<ConstantInt>(C)) {
    if (Int->getBitWidth() > 64) {
      // Don't convert if it will cause us to lose data.
      return None;
    }
    return LinearPtr(Int->getZExtValue());
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
      if (SF.CastReturnType != nullptr) {
        PHI = foldAndEmitInst(CastInst::CreateBitOrPointerCast(
              PHI, SF.CastReturnType, "returncast"));
      }
      assert(SF.ReturnValue != nullptr);
      SF.Locals[SF.ReturnValue] = PHI;
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

  /// Map from old phi nodes to corresponding new ones.  We can't use
  /// `SF.Locals` for this because some old phi nodes are mapped to trampoline
  /// block phi nodes in that map.
  DenseMap<PHINode*, PHINode*> PHINodeMap;

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
    PHINodeMap[PHI] = NewPHI;
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

  for (unsigned I = 0; I < Inst->getNumOperands(); ++I) {
    Value* OldVal = Inst->getOperand(I);
    if (auto OldBB = dyn_cast<BasicBlock>(OldVal)) {
      BasicBlock* NewBB = mapBlock(OldBB);
      NewInst->setOperand(I, NewBB);
    } else {
      Value* NewVal = mapValue(OldVal);
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
        PHINode* NewPHI = PHINodeMap[&OldPHI];
        assert(NewPHI != nullptr && "missing entry for phi node in PHINodeMap");
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
    if (MainFunc == nullptr || MainFunc->isDeclaration()) {
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
