#include "llvm/Pass.h"
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
  Optional<Value*> base() const {
    if (Terms.size() == 0) {
      return nullptr;
    } else if (Terms.size() == 1 && Terms[0].Coeff == 1) {
      return Terms[0].Ptr;
    } else {
      return None;
    }
  }
};


struct MemStore {
  uint64_t Addr;
  Value* Val;
};

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

  DenseMap<Value*, std::vector<MemStore>> Mem;
  std::vector<StackFrame> Stack;

  State(Function& OldFunc, StringRef NewName) {
    NewFunc = Function::Create(
        OldFunc.getFunctionType(),
        OldFunc.getLinkage(),
        OldFunc.getAddressSpace(),
        NewName,
        OldFunc.getParent());
    NewFunc->copyAttributesFrom(&OldFunc);

    NewBB = BasicBlock::Create(NewFunc->getContext(), "", NewFunc);

    StackFrame SF(OldFunc);
    for (unsigned I = 0; I < OldFunc.arg_size(); ++I) {
      SF.addArgument(I, std::next(NewFunc->arg_begin(), I));
    }
    Stack.emplace_back(std::move(SF));
  }

  void unwind();
  void unwindFrame(StackFrame& SF, UnwindContext* PrevUC, UnwindContext* UC);
};

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

    UC.ReturnDest = BasicBlock::Create(NewFunc->getContext(), "returndest", NewFunc);
    Type* ReturnType = Stack[I + 1].Func.getReturnType();
    if (!ReturnType->isVoidTy()) {
      Value* PHI = PHINode::Create(ReturnType, 0, "returnval", UC.ReturnDest);
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
      auto Iter = BB->begin();
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
  if (isa<Constant>(OldVal)) {
    return OldVal;
  }

  auto It = SF.Locals.find(OldVal);
  if (It == SF.Locals.end()) {
    errs() << "error: no local mapping for value " << *OldVal << "\n";
    assert(0 && "no local mapping for value");
  }
  errs() << "mapValue: found mapping" << OldVal << " " << *OldVal << " -> " << It->second << " " << *It->second << "\n";
  return It->second;
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


struct FlattenInit : public ModulePass {
  static char ID;
  FlattenInit() : ModulePass(ID) {}

  bool runOnModule(Module& M) override {
    Function* MainFunc = M.getFunction("main");
    if (MainFunc == nullptr) {
      return false;
    }

    MainFunc->setName("__cc_old_main");
    State S(*MainFunc, "main");
    S.unwind();

    return true;
  }
}; // end of struct FlattenInit
}  // end of anonymous namespace

char FlattenInit::ID = 0;
static RegisterPass<FlattenInit> X(
        "flatten-init",
        "Inline and unroll constant-valued initialization code",
        false /* Only looks at CFG */,
        false /* Analysis Pass */);
