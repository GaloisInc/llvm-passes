#include "llvm/Pass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const int WORD_SIZE = 64;
const size_t TRACE_NUM_ARGS = 8;

/// Downcast `Ty` to `VectorType*`, but only if it's a vector that we know how
/// to unroll.
static VectorType* unrollableVectorType(Type* Ty, Module* M) {
  if (auto VecTy = dyn_cast<VectorType>(Ty)) {
    if (VecTy->isScalable()) {
      return nullptr;
    }
    const DataLayout& DL = M->getDataLayout();
    if (DL.getTypeSizeInBits(VecTy->getElementType()) % 8 != 0) {
      return nullptr;
    }
    return VecTy;
  } else {
    return nullptr;
  }
}

namespace {
struct UnrollVectors : public FunctionPass {
  static char ID;
  UnrollVectors() : FunctionPass(ID) {}

  // TODO: handle ConstantVector

  bool runOnFunction(Function &F) override {
    DenseMap<Value*, SmallVector<Value*, 2>> UnrollMap;
    std::vector<PHINode*> UnrolledPHIs;
    Module* M = F.getParent();

    // Unroll vector-typed phi nodes.
    for (BasicBlock& BB : F) {
      for (PHINode& PHI : BB.phis()) {
        if (VectorType* VecTy = unrollableVectorType(PHI.getType(), M)) {
          unsigned Count = VecTy->getNumElements(); 
          Type* ElemTy = VecTy->getElementType();
          SmallVector<Value*, 2> Elems;
          for (unsigned i = 0; i < Count; ++i) {
            PHINode* ElemPHI = PHINode::Create(ElemTy, PHI.getNumIncomingValues(),
                Twine(PHI.getName()).concat(Twine(i)), &BB);
            Elems.push_back(ElemPHI);
          }
          UnrollMap.try_emplace(&PHI, std::move(Elems));
          UnrolledPHIs.push_back(&PHI);
        }
      }
    }

    // Walk all instructions in dominance order.
    DominatorTree Dom(F);
    std::vector<DomTreeNodeBase<BasicBlock>*> Pending;
    Pending.push_back(Dom.getRootNode());

    while (Pending.size() > 0) {
      DomTreeNodeBase<BasicBlock>* Node = Pending.back();
      Pending.pop_back();
      BasicBlock* BB = Node->getBlock();
      for (DomTreeNodeBase<BasicBlock>* Child : *Node) {
        Pending.push_back(Child);
      }

      for (Instruction& I : *BB) {
        if (!handleKnownInst(&I, UnrollMap)) {
          // Check if this instruction uses any unrolled value.
        }
      }
    }

    // Set incoming values for all vector-typed phi nodes.
    for (PHINode* PHI : UnrolledPHIs) {
      SmallVector<Value*, 2>& Unrolled = UnrollMap[PHI];
      unsigned Count = cast<VectorType>(PHI->getType())->getNumElements(); 
      for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
        BasicBlock* IncBB = PHI->getIncomingBlock(i);
        Value* IncVal = PHI->getIncomingValue(i);

        auto it = UnrollMap.find(IncVal);
        if (it == UnrollMap.end()) {
          llvm::errs() << "failed to find incoming value for unrolled phi node:\n";
          llvm::errs() << "  phi node: " << *PHI << "\n";
          llvm::errs() << "  incoming value: " << *IncVal << "\n";
          abort();
        }
        SmallVector<Value*, 2>& IncElems = it->second;

        for (unsigned j = 0; j < Count; ++j) {
          PHINode* ElemPHI = cast<PHINode>(Unrolled[j]);
          ElemPHI->setIncomingBlock(i, IncBB);
          ElemPHI->setIncomingValue(i, IncElems[j]);
        }
      }
    }

    return true;
  }

  bool handleKnownInst(Instruction* I, DenseMap<Value*, SmallVector<Value*, 2>>& UnrollMap) {
    Module* M = I->getModule();

    if (auto InsertElement = dyn_cast<InsertElementInst>(I)) {
      if (!unrollableVectorType(InsertElement->getType(), M)) {
        return false;
      }

      Value* Vector = InsertElement->getOperand(0);
      Value* Elem = InsertElement->getOperand(1);
      Value* IndexVal = InsertElement->getOperand(2);

      unsigned Index;
      if (auto ConstIndexVal = dyn_cast<ConstantInt>(IndexVal)) {
        Index = ConstIndexVal->getZExtValue();
      } else {
        return false;
      }

      auto it = UnrollMap.find(Vector);
      if (it == UnrollMap.end()) {
        return false;
      }
      SmallVector<Value*, 2> Elems = it->second;
      Elems[Index] = Elem;
      UnrollMap.try_emplace(InsertElement, std::move(Elems));
      return true;

    } else if (auto ExtractElement = dyn_cast<ExtractElementInst>(I)) {
      if (!unrollableVectorType(ExtractElement->getVectorOperandType(), M)) {
        return false;
      }

      Value* Vector = ExtractElement->getOperand(0);
      Value* IndexVal = ExtractElement->getOperand(1);

      unsigned Index;
      if (auto ConstIndexVal = dyn_cast<ConstantInt>(IndexVal)) {
        Index = ConstIndexVal->getZExtValue();
      } else {
        return false;
      }

      auto it = UnrollMap.find(Vector);
      if (it == UnrollMap.end()) {
        return false;
      }
      SmallVector<Value*, 2>& Elems = it->second;
      ExtractElement->replaceAllUsesWith(Elems[Index]);
      return true;

    } else if (auto Load = dyn_cast<LoadInst>(I)) {
      VectorType* VecTy = unrollableVectorType(Load->getType(), M);
      if (!VecTy) {
        return false;
      }

      if (!Load->isSimple()) {
        return false;
      }

      unsigned Count = VecTy->getNumElements(); 
      Type* ElemTy = VecTy->getElementType();
      unsigned AddrSpace = Load->getPointerAddressSpace();
      Value* Ptr = Load->getPointerOperand();
      Ptr = CastInst::Create(Instruction::BitCast, Ptr,
          PointerType::get(ElemTy, AddrSpace),
          Twine(Ptr->getName(), "elem"), Load);

      unsigned Align = Load->getAlignment();
      uint64_t ElemSize = M->getDataLayout().getTypeStoreSize(ElemTy);
      if (Align > ElemSize) {
        assert(Align % ElemSize == 0);
        Align = ElemSize;
      }

      SmallVector<Value*, 2> Elems;
      for (unsigned i = 0; i < Count; ++i) {
        Value* OffsetPtr;
        if (i == 0) {
          OffsetPtr = Ptr;
        } else {
          Value* Idxs[1] = { ConstantInt::get(IntegerType::get(M->getContext(), 32), i) };
          OffsetPtr = GetElementPtrInst::CreateInBounds(ElemTy, Ptr, Idxs,
              Twine(Ptr->getName()).concat(Twine(i)), Load);
        }
        Elems.push_back(new LoadInst(ElemTy, OffsetPtr,
              Twine(Load->getName()).concat(Twine(i)), false, Align, Load));
      }
      UnrollMap.try_emplace(Load, std::move(Elems));

      return true;

    } else if (auto Store = dyn_cast<StoreInst>(I)) {
      return false;  // TODO

    } else {
      return false;
    }
  }
}; // end of struct UnrollVectors
}  // end of anonymous namespace

char UnrollVectors::ID = 0;
static RegisterPass<UnrollVectors> X(
        "unroll-vectors",
        "Unroll vector ops into sequences of scalar ops",
        false /* Only looks at CFG */,
        false /* Analysis Pass */);
