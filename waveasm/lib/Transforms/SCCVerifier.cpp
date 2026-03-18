// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCC Verifier Pass
//
// Verifies that no SCC-clobbering SALU instruction is placed between an
// SCC-producing op and its consumer.  Uses isa<> checks instead of
// hasTrait<SCCDef>() because adding traits to existing op classes changes
// ODS-generated C++ and causes MLIR passes to produce different IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-scc-verifier"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCCVERIFIER
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

/// Returns true if the operation writes the SCC flag on hardware.
static bool writesSCC(Operation *op) {
  if (isa<S_ADD_U32, S_ADDC_U32, S_ADD_I32, S_SUB_U32, S_SUB_I32>(op))
    return true;
  if (isa<S_CMP_LT_U32, S_CMP_EQ_U32, S_CMP_LE_U32, S_CMP_GT_U32,
          S_CMP_GE_U32, S_CMP_NE_U32, S_CMP_LT_I32, S_CMP_EQ_I32,
          S_CMP_LE_I32, S_CMP_GT_I32, S_CMP_GE_I32, S_CMP_NE_I32>(op))
    return true;
  if (isa<S_AND_B32, S_AND_B64, S_OR_B32, S_OR_B64, S_XOR_B32, S_XOR_B64,
          S_ANDN2_B32, S_ANDN2_B64, S_ORN2_B32, S_ORN2_B64,
          S_NAND_B32, S_NAND_B64, S_NOR_B32, S_NOR_B64,
          S_XNOR_B32, S_XNOR_B64>(op))
    return true;
  if (isa<S_LSHL_B32, S_LSHL_B64, S_LSHR_B32, S_LSHR_B64,
          S_ASHR_I32, S_ASHR_I64>(op))
    return true;
  if (isa<S_MIN_I32, S_MIN_U32, S_MAX_I32, S_MAX_U32>(op))
    return true;
  if (isa<S_BFE_U32, S_BFE_I32, S_BFE_U64, S_BFE_I64>(op))
    return true;
  if (isa<S_NOT_B32, S_NOT_B64, S_BREV_B32, S_BREV_B64,
          S_BCNT0_I32_B32, S_BCNT0_I32_B64, S_BCNT1_I32_B32, S_BCNT1_I32_B64,
          S_FF0_I32_B32, S_FF0_I32_B64, S_FF1_I32_B32, S_FF1_I32_B64,
          S_FLBIT_I32_B32, S_FLBIT_I32_B64, S_ABS_I32>(op))
    return true;
  return false;
}

struct SCCVerifierPass
    : public waveasm::impl::WAVEASMSCCVerifierBase<SCCVerifierPass> {
  using WAVEASMSCCVerifierBase::WAVEASMSCCVerifierBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    unsigned errorCount = 0;
    module->walk([&](ProgramOp program) {
      for (Block &block : program.getBody())
        errorCount += verifyBlock(block);
    });
    if (errorCount > 0) {
      LLVM_DEBUG(llvm::dbgs() << "SCC verifier: found " << errorCount
                               << " SCC hazard(s)\n");
      signalPassFailure();
    }
  }

private:
  static SmallVector<Operation *> findSCCClobbersBetween(Operation *producer,
                                                          Operation *consumer) {
    SmallVector<Operation *> clobbers;
    if (!producer || !consumer || producer->getBlock() != consumer->getBlock())
      return clobbers;
    bool inRange = false;
    for (Operation &op : *producer->getBlock()) {
      if (&op == producer) { inRange = true; continue; }
      if (&op == consumer) break;
      if (inRange && writesSCC(&op))
        clobbers.push_back(&op);
    }
    return clobbers;
  }

  static void emitSCCClobberError(Operation *consumer, Operation *producer,
                                   ArrayRef<Operation *> clobbers) {
    auto diag = consumer->emitError()
                << "SCC hazard: " << clobbers.size()
                << " SCC-clobbering op(s) between SCC producer '"
                << producer->getName() << "' and consumer '"
                << consumer->getName() << "'";
    for (Operation *c : clobbers)
      diag.attachNote(c->getLoc())
          << "SCC clobbered here by '" << c->getName() << "'";
    diag.attachNote(producer->getLoc()) << "SCC defined here";
  }

  unsigned verifyBlock(Block &block) {
    unsigned errors = 0;
    Operation *lastSCCWriter = nullptr;
    for (Operation &op : block) {
      if (auto condOp = dyn_cast<ConditionOp>(&op)) {
        Value cond = condOp.getCondition();
        Operation *condDef = cond.getDefiningOp();
        if (condDef && lastSCCWriter && lastSCCWriter != condDef) {
          auto clobbers = findSCCClobbersBetween(condDef, &op);
          if (!clobbers.empty()) {
            emitSCCClobberError(&op, condDef, clobbers);
            ++errors;
          }
        }
      }
      if (auto ifOp = dyn_cast<IfOp>(&op)) {
        Value cond = ifOp.getCondition();
        Operation *condDef = cond.getDefiningOp();
        if (condDef && lastSCCWriter && lastSCCWriter != condDef) {
          auto clobbers = findSCCClobbersBetween(condDef, &op);
          if (!clobbers.empty()) {
            emitSCCClobberError(&op, condDef, clobbers);
            ++errors;
          }
        }
      }
      if (isa<S_CSELECT_B32>(&op) && !lastSCCWriter) {
        op.emitError()
            << "SCC hazard: s_cselect_b32 has no preceding SCC writer";
        ++errors;
      }
      if (isa<S_ADDC_U32>(&op) && !lastSCCWriter) {
        op.emitError()
            << "SCC hazard: s_addc_u32 has no preceding SCC writer";
        ++errors;
      }
      if (writesSCC(&op))
        lastSCCWriter = &op;
      for (Region &region : op.getRegions())
        for (Block &nestedBlock : region)
          errors += verifyBlock(nestedBlock);
    }
    return errors;
  }
};

} // namespace
