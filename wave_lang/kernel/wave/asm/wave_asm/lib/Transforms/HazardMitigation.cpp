// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass - Insert s_nop instructions for hardware hazards
//
// This pass handles hardware-specific hazards that require NOP insertion:
// - VALU → v_readfirstlane hazard (gfx940+)
// - s_barrier → MFMA/LDS-consumer hazard (gfx940+): the hardware needs
//   at least 2 wait states after s_barrier before an MFMA that consumes
//   data written to LDS by another wave.  We insert s_nop 0 (1 wait state
//   each) after each barrier to match the aiter kernel's pattern.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

/// Check if an operation is a VALU instruction (writes VGPR, not memory)
bool isVALUOp(Operation *op) {
  // Must produce at least one VGPR result
  bool writesVGPR = false;
  for (Value result : op->getResults()) {
    if (isa<VRegType, PVRegType>(result.getType())) {
      writesVGPR = true;
      break;
    }
  }
  if (!writesVGPR)
    return false;

  // Exclude memory operations (VMEM, LDS, SMEM)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, GLOBAL_LOAD_DWORD,
          GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
          GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT,
          GLOBAL_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
          FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4, DS_READ_B32, DS_READ_B64,
          DS_READ_B128, DS_READ2_B32, DS_READ2_B64, DS_READ_U8, DS_READ_I8,
          DS_READ_U16, DS_READ_I16>(op))
    return false;

  // Exclude non-ALU ops that produce VGPRs
  if (isa<PrecoloredVRegOp, PackOp, ExtractOp>(op))
    return false;

  // Exclude v_readfirstlane (it's the consumer in the hazard, not the producer)
  if (isa<V_READFIRSTLANE_B32>(op))
    return false;

  return true;
}

/// Check if an operation is v_readfirstlane
bool isReadfirstlaneOp(Operation *op) { return isa<V_READFIRSTLANE_B32>(op); }

/// Get the set of VGPRs written by an operation
llvm::DenseSet<Value> getVGPRDefs(Operation *op) {
  llvm::DenseSet<Value> defs;
  for (Value result : op->getResults()) {
    Type ty = result.getType();
    // Check if it's a VGPR type
    if (isa<VRegType, PVRegType>(ty)) {
      defs.insert(result);
    }
  }
  return defs;
}

/// Get the set of VGPRs read by an operation
llvm::DenseSet<Value> getVGPRUses(Operation *op) {
  llvm::DenseSet<Value> uses;
  for (Value operand : op->getOperands()) {
    Type ty = operand.getType();
    // Check if it's a VGPR type
    if (isa<VRegType, PVRegType>(ty)) {
      uses.insert(operand);
    }
  }
  return uses;
}

/// Check if two value sets have any intersection
bool hasIntersection(const llvm::DenseSet<Value> &a,
                     const llvm::DenseSet<Value> &b) {
  for (Value v : a) {
    if (b.contains(v))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Target-Specific Hazard Rules
//===----------------------------------------------------------------------===//

/// Check if target requires VALU → readfirstlane hazard mitigation
static bool needsVALUReadFirstLaneHazard(TargetAttrInterface target) {
  // gfx940+ (CDNA3/4) architectures need this hazard mitigation
  return isa<GFX942TargetAttr, GFX950TargetAttr, GFX1250TargetAttr>(target);
}

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass
//===----------------------------------------------------------------------===//

struct HazardMitigationPass
    : public PassWrapper<HazardMitigationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HazardMitigationPass)

  HazardMitigationPass() = default;
  HazardMitigationPass(StringRef target) : targetArch(target.str()) {
    std::optional<TargetKind> parsed = symbolizeTargetKind(target);
    if (parsed) {
      this->targetKindEnum = *parsed;
      this->validTarget = true;
    }
  }

  StringRef getArgument() const override { return "waveasm-hazard-mitigation"; }

  StringRef getDescription() const override {
    return "Insert s_nop instructions to mitigate hardware hazards";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Check for invalid target architecture
    if (!validTarget) {
      module.emitError() << "Invalid target architecture: '" << targetArch
                         << "'. Supported targets: gfx942, gfx950, gfx1250";
      return signalPassFailure();
    }

    // Process each program
    module.walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  TargetKind targetKindEnum = TargetKind::GFX942;
  std::string targetArch;
  bool validTarget = true; // Default constructor uses default target
  unsigned numNopsInserted = 0;

  void processProgram(ProgramOp program) {
    TargetAttrInterface targetKind;
    // Get target from program if available.
    if (auto targetAttr = program.getTarget()) {
      targetKind = targetAttr.getTargetKind();
    } else {
      targetKind = getTargetKindAttr(program.getContext(), targetKindEnum);
    }

    // Check if this target needs VALU → readfirstlane hazard mitigation
    bool needsVALUHazard = needsVALUReadFirstLaneHazard(targetKind);

    // gfx940+ also needs barrier-adjacent NOPs for the s_barrier → MFMA/LDS
    // consumer hazard.  The hardware requires at least 2 wait states after
    // s_barrier before an instruction that reads data written to LDS by
    // another wave (the common pattern in double-buffered GEMM kernels).
    bool needsBarrierNops = needsVALUReadFirstLaneHazard(targetKind);

    if (!needsVALUHazard && !needsBarrierNops)
      return;

    // Collect operations in order, recursively walking into while/if bodies
    llvm::SmallVector<Operation *> ops;
    collectOpsRecursive(program.getBodyBlock(), ops);

    // Scan for hazards and collect insertion points
    llvm::SmallVector<Operation *> varuReadfirstlanePoints;
    llvm::SmallVector<Operation *> barrierNopPoints;

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      Operation *current = ops[i];
      Operation *next = ops[i + 1];

      // Check for VALU → v_readfirstlane hazard
      if (needsVALUHazard && isVALUOp(current) && isReadfirstlaneOp(next)) {
        auto defs = getVGPRDefs(current);
        auto uses = getVGPRUses(next);
        if (hasIntersection(defs, uses)) {
          varuReadfirstlanePoints.push_back(next);
        }
      }
    }

    // Barrier-adjacent NOP insertion: insert 2 s_nop 0 after each s_barrier.
    // This ensures the hardware has enough wait states before the next
    // instruction (typically an MFMA or ds_read) can safely consume data
    // that was written to LDS by other waves before the barrier.
    if (needsBarrierNops) {
      for (Operation *op : ops) {
        if (isa<S_BARRIER>(op)) {
          // Count existing NOPs immediately after the barrier
          int existingNops = 0;
          Operation *scan = op->getNextNode();
          while (scan && isa<S_NOP>(scan)) {
            existingNops++;
            scan = scan->getNextNode();
          }

          // Insert enough NOPs to reach 2 total after barrier
          int nopsNeeded = 2 - existingNops;
          if (nopsNeeded > 0 && op->getNextNode()) {
            barrierNopPoints.push_back(op);
          }
        }
      }
    }

    // Insert VALU → readfirstlane NOPs
    for (Operation *insertBefore : varuReadfirstlanePoints) {
      OpBuilder builder(insertBefore);
      S_NOP::create(builder, insertBefore->getLoc(),
                    builder.getI32IntegerAttr(0));
      numNopsInserted++;
    }

    // Insert barrier-adjacent NOPs
    for (Operation *barrierOp : barrierNopPoints) {
      int existingNops = 0;
      Operation *scan = barrierOp->getNextNode();
      while (scan && isa<S_NOP>(scan)) {
        existingNops++;
        scan = scan->getNextNode();
      }

      int nopsNeeded = 2 - existingNops;
      if (nopsNeeded > 0 && barrierOp->getNextNode()) {
        Operation *insertBefore = barrierOp->getNextNode();
        while (insertBefore && isa<S_NOP>(insertBefore))
          insertBefore = insertBefore->getNextNode();
        if (!insertBefore)
          continue;
        OpBuilder builder(insertBefore);
        for (int n = 0; n < nopsNeeded; ++n) {
          S_NOP::create(builder, barrierOp->getLoc(),
                        builder.getI32IntegerAttr(0));
          numNopsInserted++;
        }
      }
    }

  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMHazardMitigationPass() {
  return std::make_unique<HazardMitigationPass>();
}

std::unique_ptr<mlir::Pass>
createWAVEASMHazardMitigationPass(llvm::StringRef targetArch) {
  return std::make_unique<HazardMitigationPass>(targetArch);
}

} // namespace waveasm
