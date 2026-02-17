// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Ticketing Pass - Insert s_waitcnt instructions for memory synchronization
//
// This pass implements a ticket-based tracking system for VMEM and LGKM
// memory operations. Each memory operation is assigned a monotonically
// increasing ticket. When a value is used, the pass computes the minimum
// wait threshold and inserts s_waitcnt if required.
//
// Key optimizations over a naive approach:
// 1. Incremental wait tracking: when new loads are issued after a wait,
//    we increment the last-wait threshold instead of resetting it. This
//    correctly recognizes that values loaded BEFORE the last wait are still
//    ready, avoiding unnecessary waitcnt insertions.
// 2. Latency-covering for LGKM: for ds_read (LDS) loads, if enough
//    instructions separate the load from its use, we skip fine-grained
//    lgkmcnt(N>0) waits since the data has certainly arrived. This matches
//    how hand-tuned kernels (e.g., AITER) rely on latency hiding.
// 3. Barrier handling respects schedule-placed waitcnts: if the schedule
//    already placed a partial wait (e.g., vmcnt(10)) before a barrier, the
//    pass trusts it and only supplements missing counters.  This matches
//    the aiter kernel's pattern of partial drains before barriers.
// 4. Loop back-edge (ConditionOp) does NOT reset wait tracking, allowing
//    waits from the end of one iteration to carry to the next, eliminating
//    redundant waitcnts at loop tops.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Ticketing System
//===----------------------------------------------------------------------===//

/// Tracks memory operation tickets for waitcnt insertion.
///
/// The ticket system maps each load operation to a monotonically increasing
/// ticket number. The hardware vmcnt/lgkmcnt counters track outstanding
/// loads in FIFO order, so vmcnt(N) means "wait until at most N loads are
/// outstanding" (the N most recent). To wait for ticket T, we need
/// vmcnt(lastTicket - T), which allows (lastTicket - T) newer loads to
/// remain outstanding while guaranteeing T (and older) have completed.
class Ticketing {
public:
  Ticketing() = default;

  /// Issue a new VMEM ticket, returns the ticket number.
  /// A new load shifts the meaning of previous waits: if we previously
  /// waited for vmcnt(N), that now effectively means vmcnt(N+1) since
  /// one more operation is outstanding. We increment (not reset) the
  /// last wait value to reflect this.
  int64_t nextVmemTicket() {
    ++lastVmemTicket;
    if (lastVmemWait.has_value())
      ++(*lastVmemWait);
    return lastVmemTicket;
  }

  /// Issue a new LGKM ticket, returns the ticket number.
  int64_t nextLgkmTicket() {
    ++lastLgkmTicket;
    if (lastLgkmWait.has_value())
      ++(*lastLgkmWait);
    return lastLgkmTicket;
  }

  /// Get current VMEM ticket (last issued)
  int64_t getLastVmemTicket() const { return lastVmemTicket; }

  /// Get current LGKM ticket (last issued)
  int64_t getLastLgkmTicket() const { return lastLgkmTicket; }

  /// Compute the vmcnt threshold needed to wait for a given ticket.
  /// Returns std::nullopt if no wait is needed (already satisfied).
  std::optional<int64_t> computeVmemWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastVmemTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastVmemTicket - requiredTicket);

    if (lastVmemWait.has_value() && *lastVmemWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Compute the lgkmcnt threshold needed to wait for a given ticket.
  std::optional<int64_t> computeLgkmWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastLgkmTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastLgkmTicket - requiredTicket);

    if (lastLgkmWait.has_value() && *lastLgkmWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Check if any VMEM operations are outstanding since the last full drain.
  /// Used for barrier pre-waits where we need vmcnt(0).
  bool hasOutstandingVmem() const {
    if (lastVmemTicket < 0)
      return false; // No VMEM ops issued at all
    return !lastVmemWait.has_value() || *lastVmemWait > 0;
  }

  /// Check if any LGKM operations are outstanding since the last full drain.
  bool hasOutstandingLgkm() const {
    if (lastLgkmTicket < 0)
      return false; // No LGKM ops issued at all
    return !lastLgkmWait.has_value() || *lastLgkmWait > 0;
  }

  /// Record that a vmem wait was emitted with the given threshold
  void observeVmemWait(int64_t threshold) {
    if (!lastVmemWait.has_value() || threshold < *lastVmemWait)
      lastVmemWait = threshold;
  }

  /// Record that a lgkm wait was emitted with the given threshold
  void observeLgkmWait(int64_t threshold) {
    if (!lastLgkmWait.has_value() || threshold < *lastLgkmWait)
      lastLgkmWait = threshold;
  }

  /// Reset wait tracking (e.g., at loop entry from outside)
  void resetWaits() {
    lastVmemWait = std::nullopt;
    lastLgkmWait = std::nullopt;
  }

private:
  int64_t lastVmemTicket = -1;
  int64_t lastLgkmTicket = -1;
  std::optional<int64_t> lastVmemWait;
  std::optional<int64_t> lastLgkmWait;
};

//===----------------------------------------------------------------------===//
// Memory Operation Classification
//===----------------------------------------------------------------------===//

enum class MemOpKind {
  None,
  VmemLoad,  // buffer_load, global_load, flat_load
  VmemStore, // buffer_store, global_store, flat_store
  LgkmLoad,  // ds_read, s_load
  LgkmStore  // ds_write
};

/// Classify an operation by its memory type using MLIR op types
MemOpKind classifyMemOp(Operation *op) {
  // VMEM loads (buffer, global, flat)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, GLOBAL_LOAD_DWORD,
          GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
          GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT,
          GLOBAL_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
          FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4,
          // Gather-to-LDS: VMEM load directly to LDS (still uses VMEM path)
          BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORDX4_LDS>(op))
    return MemOpKind::VmemLoad;

  // VMEM stores (buffer, global, flat)
  if (isa<BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX3,
          BUFFER_STORE_DWORDX4, BUFFER_STORE_BYTE, BUFFER_STORE_SHORT,
          GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX3,
          GLOBAL_STORE_DWORDX4, GLOBAL_STORE_BYTE, GLOBAL_STORE_SHORT,
          FLAT_STORE_DWORD, FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX3,
          FLAT_STORE_DWORDX4>(op))
    return MemOpKind::VmemStore;

  // LDS loads (ds_read)
  if (isa<DS_READ_B32, DS_READ_B64, DS_READ_B128, DS_READ2_B32, DS_READ2_B64,
          DS_READ_U8, DS_READ_I8, DS_READ_U16, DS_READ_I16>(op))
    return MemOpKind::LgkmLoad;

  // LDS stores (ds_write)
  if (isa<DS_WRITE_B32, DS_WRITE_B64, DS_WRITE_B128, DS_WRITE2_B32,
          DS_WRITE2_B64, DS_WRITE_B8, DS_WRITE_B16>(op))
    return MemOpKind::LgkmStore;

  // Scalar loads (s_load) - also use LGKM counter
  if (isa<S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4, S_LOAD_DWORDX8,
          S_LOAD_DWORDX16>(op))
    return MemOpKind::LgkmLoad;

  return MemOpKind::None;
}

/// Check if an operation is an MFMA (matrix fused multiply-add) instruction.
/// MFMAs are long-latency (32+ cycles) and read source VGPRs at issue time.
bool isMFMAOp(Operation *op) {
  return op->hasTrait<OpTrait::MFMAOp>();
}

//===----------------------------------------------------------------------===//
// InsertWaitcnt Pass
//===----------------------------------------------------------------------===//

struct InsertWaitcntPass
    : public PassWrapper<InsertWaitcntPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertWaitcntPass)

  InsertWaitcntPass() = default;
  InsertWaitcntPass(bool insertAfterLoads)
      : insertAfterLoads(insertAfterLoads) {}

  StringRef getArgument() const override { return "waveasm-insert-waitcnt"; }

  StringRef getDescription() const override {
    return "Insert s_waitcnt instructions for memory synchronization";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each program
    module.walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  bool insertAfterLoads = false;

  /// Minimum number of operations between an LGKM load and its use for
  /// latency covering. ds_read (LDS) latency is ~20-40 cycles on CDNA.
  /// With 40+ intervening instructions (each >= 1 cycle, MFMAs = 32 cycles),
  /// the load has certainly completed. This allows us to skip lgkmcnt(N>0)
  /// waits when enough instructions provide latency hiding.
  static constexpr int64_t kMinOpsForLgkmLatency = 40;

  // Statistics
  unsigned numVmemOps = 0;
  unsigned numLgkmOps = 0;
  unsigned numWaitcntInserted = 0;

  void processProgram(ProgramOp program) {
    Ticketing ticketing;

    // Get target-specific waitcnt limits. GFX9 uses 4-bit lgkmcnt (max 15).
    // When computed threshold exceeds the limit, we must use 0 (full drain).
    int64_t maxLgkmcnt = 15;
    int64_t maxVmcnt = 63;
    if (auto targetAttr = program.getTarget()) {
      TargetAttrInterface targetKind = targetAttr.getTargetKind();
      maxLgkmcnt = targetKind.getMaxLgkmcnt();
      maxVmcnt = targetKind.getMaxVmcnt();
    }

    auto capLgkmcnt = [maxLgkmcnt](int64_t v) {
      return v > maxLgkmcnt ? int64_t(0) : v;
    };
    auto capVmcnt = [maxVmcnt](int64_t v) {
      return v > maxVmcnt ? int64_t(0) : v;
    };

    // Map from Value to the ticket that produces it
    llvm::DenseMap<Value, std::pair<MemOpKind, int64_t>> valueTickets;

    // Map from Value to the sequential op index where the load was issued.
    // Used for latency-covering heuristic.
    llvm::DenseMap<Value, int64_t> loadDefIndex;

    // Sequential operation counter for latency tracking
    int64_t opIndex = 0;

    // Collect all ops first then iterate, to avoid iterator invalidation
    // when inserting new waitcnt ops during traversal.  program.walk
    // triggers double-free crashes on large triple-buffered kernels.
    llvm::SmallVector<Operation *> allOps;
    collectOpsRecursive(program.getBodyBlock(), allOps);

    auto processOp = [&](Operation *op) {

      // Increment sequential index for every real operation
      ++opIndex;

      // Observe pre-existing waitcnt instructions (from translation or prior
      // passes) This prevents emitting redundant waits
      if (auto waitcnt = dyn_cast<S_WAITCNT>(op)) {
        auto vmcntAttr = waitcnt.getVmcntAttr();
        auto lgkmcntAttr = waitcnt.getLgkmcntAttr();
        if (vmcntAttr) {
          ticketing.observeVmemWait(vmcntAttr.getValue().getSExtValue());
        }
        if (lgkmcntAttr) {
          ticketing.observeLgkmWait(lgkmcntAttr.getValue().getSExtValue());
        }
        return;
      }

      if (auto waitcnt = dyn_cast<S_WAITCNT_VMCNT>(op)) {
        auto countAttr = waitcnt.getCountAttr();
        if (countAttr) {
          ticketing.observeVmemWait(countAttr.getValue().getSExtValue());
        }
        return;
      }

      if (auto waitcnt = dyn_cast<S_WAITCNT_LGKMCNT>(op)) {
        auto countAttr = waitcnt.getCountAttr();
        if (countAttr) {
          ticketing.observeLgkmWait(countAttr.getValue().getSExtValue());
        }
        return;
      }

      // Barriers require memory to be synchronized before execution.
      //
      // If the schedule has already placed a waitcnt immediately before
      // the barrier (e.g., from MemoryCounterWaitBarrier), we trust it
      // and only add a supplementary wait for any counter the schedule
      // didn't cover.  This allows partial waits like vmcnt(10) from
      // the schedule to survive, matching the aiter kernel pattern.
      //
      // If no schedule-placed waitcnt precedes the barrier, fall back to
      // the conservative full drain (vmcnt(0) lgkmcnt(0)).
      if (isa<S_BARRIER>(op)) {
        // Check if the immediately preceding op is a waitcnt placed by the
        // schedule (i.e., already in the IR before this pass ran).
        // We detect this by checking the previous node.
        Operation *prev = op->getPrevNode();
        bool hasScheduleVmcnt = false;
        bool hasScheduleLgkmcnt = false;

        if (prev) {
          if (auto swc = dyn_cast<S_WAITCNT>(prev)) {
            hasScheduleVmcnt = swc.getVmcntAttr() != nullptr;
            hasScheduleLgkmcnt = swc.getLgkmcntAttr() != nullptr;
          } else if (isa<S_WAITCNT_VMCNT>(prev)) {
            hasScheduleVmcnt = true;
          } else if (isa<S_WAITCNT_LGKMCNT>(prev)) {
            hasScheduleLgkmcnt = true;
          }
        }

        // Only insert waits for counters not already covered by the
        // schedule-placed waitcnt.
        bool needVmem = !hasScheduleVmcnt && ticketing.hasOutstandingVmem();
        bool needLgkm = !hasScheduleLgkmcnt && ticketing.hasOutstandingLgkm();

        if (needVmem || needLgkm) {
          OpBuilder builder(op->getContext());
          builder.setInsertionPoint(op);

          if (needVmem && needLgkm) {
            S_WAITCNT::create(builder, op->getLoc(),
                              builder.getI32IntegerAttr(0),
                              builder.getI32IntegerAttr(0), IntegerAttr());
          } else if (needVmem) {
            S_WAITCNT_VMCNT::create(builder, op->getLoc(), 0);
          } else {
            S_WAITCNT_LGKMCNT::create(builder, op->getLoc(), 0);
          }
          numWaitcntInserted++;
        }

        // Observe the effective wait state after the barrier.
        // If the schedule placed a partial wait (e.g. vmcnt(10)), observe
        // that value instead of 0.
        if (hasScheduleVmcnt && prev) {
          if (auto swc = dyn_cast<S_WAITCNT>(prev)) {
            if (auto attr = swc.getVmcntAttr())
              ticketing.observeVmemWait(attr.getInt());
          } else if (auto swcv = dyn_cast<S_WAITCNT_VMCNT>(prev)) {
            if (auto attr = swcv.getCountAttr())
              ticketing.observeVmemWait(attr.getInt());
          }
        } else {
          ticketing.observeVmemWait(0);
        }

        if (hasScheduleLgkmcnt && prev) {
          if (auto swc = dyn_cast<S_WAITCNT>(prev)) {
            if (auto attr = swc.getLgkmcntAttr())
              ticketing.observeLgkmWait(attr.getInt());
          } else if (auto swcl = dyn_cast<S_WAITCNT_LGKMCNT>(prev)) {
            if (auto attr = swcl.getCountAttr())
              ticketing.observeLgkmWait(attr.getInt());
          }
        } else {
          ticketing.observeLgkmWait(0);
        }
        return;
      }

      // Check if this is a memory operation
      MemOpKind kind = classifyMemOp(op);

      if (kind != MemOpKind::None) {
        // For stores, check operands BEFORE incrementing ticket counter
        if (kind == MemOpKind::VmemStore || kind == MemOpKind::LgkmStore) {
          std::optional<int64_t> neededVmcnt;
          std::optional<int64_t> neededLgkmcnt;

          for (Value operand : op->getOperands()) {
            auto it = valueTickets.find(operand);
            if (it == valueTickets.end())
              continue;

            auto [opKind, opTicket] = it->second;

            if (opKind == MemOpKind::VmemLoad) {
              auto wait = ticketing.computeVmemWait(opTicket);
              if (wait.has_value()) {
                if (!neededVmcnt.has_value() || *wait < *neededVmcnt)
                  neededVmcnt = wait;
              }
            } else if (opKind == MemOpKind::LgkmLoad) {
              auto wait = ticketing.computeLgkmWait(opTicket);
              if (wait.has_value()) {
                if (!neededLgkmcnt.has_value() || *wait < *neededLgkmcnt)
                  neededLgkmcnt = wait;
              }
            }
          }

          // Insert waitcnt before the store if needed
          if (neededVmcnt.has_value() || neededLgkmcnt.has_value()) {
            OpBuilder builder(op->getContext());
            builder.setInsertionPoint(op);

            int64_t vmcntVal =
                neededVmcnt.has_value() ? capVmcnt(*neededVmcnt) : 0;
            int64_t lgkmcntVal =
                neededLgkmcnt.has_value() ? capLgkmcnt(*neededLgkmcnt) : 0;

            if (neededVmcnt.has_value() && neededLgkmcnt.has_value()) {
              S_WAITCNT::create(builder, op->getLoc(),
                                builder.getI32IntegerAttr(vmcntVal),
                                builder.getI32IntegerAttr(lgkmcntVal),
                                IntegerAttr());
              ticketing.observeVmemWait(vmcntVal);
              ticketing.observeLgkmWait(lgkmcntVal);
            } else if (neededVmcnt.has_value()) {
              S_WAITCNT_VMCNT::create(builder, op->getLoc(),
                                      builder.getI32IntegerAttr(vmcntVal));
              ticketing.observeVmemWait(vmcntVal);
            } else {
              S_WAITCNT_LGKMCNT::create(builder, op->getLoc(),
                                        builder.getI32IntegerAttr(lgkmcntVal));
              ticketing.observeLgkmWait(lgkmcntVal);
            }
            numWaitcntInserted++;
          }
        }

        // Now increment the ticket counter for this memory op
        // NOTE: Only LOADS increment the counter because vmcnt/lgkmcnt only
        // track outstanding loads, not stores.
        int64_t ticket = -1;
        if (kind == MemOpKind::VmemLoad) {
          ticket = ticketing.nextVmemTicket();
          numVmemOps++;
        } else if (kind == MemOpKind::LgkmLoad) {
          ticket = ticketing.nextLgkmTicket();
          numLgkmOps++;
        } else {
          numVmemOps++;
        }

        // Map results to tickets and record def index (for loads)
        for (Value result : op->getResults()) {
          valueTickets[result] = {kind, ticket};
          loadDefIndex[result] = opIndex;
        }

        // For conservative mode, insert wait immediately after loads
        if (insertAfterLoads &&
            (kind == MemOpKind::VmemLoad || kind == MemOpKind::LgkmLoad)) {
          OpBuilder builder(op->getContext());
          builder.setInsertionPointAfter(op);

          if (kind == MemOpKind::VmemLoad) {
            S_WAITCNT_VMCNT::create(builder, op->getLoc(),
                                    builder.getI32IntegerAttr(0));
            ticketing.observeVmemWait(0);
          } else {
            S_WAITCNT_LGKMCNT::create(builder, op->getLoc(),
                                      builder.getI32IntegerAttr(0));
            ticketing.observeLgkmWait(0);
          }
          numWaitcntInserted++;
        }
        return;
      }

      // For non-memory operations, check if any operands require waits
      std::optional<int64_t> neededVmcnt;
      std::optional<int64_t> neededLgkmcnt;

      for (Value operand : op->getOperands()) {
        auto it = valueTickets.find(operand);
        if (it == valueTickets.end())
          continue;

        auto [opKind, ticket] = it->second;

        if (opKind == MemOpKind::VmemLoad) {
          auto wait = ticketing.computeVmemWait(ticket);
          if (wait.has_value()) {
            if (!neededVmcnt.has_value() || *wait < *neededVmcnt)
              neededVmcnt = wait;
          }
        } else if (opKind == MemOpKind::LgkmLoad) {
          auto wait = ticketing.computeLgkmWait(ticket);
          if (wait.has_value()) {
            // Latency-covering heuristic for LGKM loads:
            // If the threshold > 0 (this is NOT the most recent LGKM op)
            // and enough instructions have elapsed since the load, skip
            // the wait. The data has certainly arrived due to latency hiding.
            // We only apply this when threshold > 0 because threshold == 0
            // means the load is the most recent LGKM op and might still be
            // in-flight.
            if (*wait > 0) {
              auto defIt = loadDefIndex.find(operand);
              if (defIt != loadDefIndex.end()) {
                int64_t gap = opIndex - defIt->second;
                if (gap >= kMinOpsForLgkmLatency) {
                  // Enough instructions for latency hiding; skip this wait
                  continue;
                }
              }
            }

            if (!neededLgkmcnt.has_value() || *wait < *neededLgkmcnt)
              neededLgkmcnt = wait;
          }
        }
      }

      // Insert waitcnt if needed
      if (neededVmcnt.has_value() || neededLgkmcnt.has_value()) {
        OpBuilder builder(op->getContext());
        builder.setInsertionPoint(op);

        int64_t vmcntVal =
            neededVmcnt.has_value() ? capVmcnt(*neededVmcnt) : 0;
        int64_t lgkmcntVal =
            neededLgkmcnt.has_value() ? capLgkmcnt(*neededLgkmcnt) : 0;

        // Use combined s_waitcnt if both are needed
        if (neededVmcnt.has_value() && neededLgkmcnt.has_value()) {
          S_WAITCNT::create(builder, op->getLoc(),
                            builder.getI32IntegerAttr(vmcntVal),
                            builder.getI32IntegerAttr(lgkmcntVal),
                            IntegerAttr());
          ticketing.observeVmemWait(vmcntVal);
          ticketing.observeLgkmWait(lgkmcntVal);
        } else if (neededVmcnt.has_value()) {
          S_WAITCNT_VMCNT::create(builder, op->getLoc(),
                                  builder.getI32IntegerAttr(vmcntVal));
          ticketing.observeVmemWait(vmcntVal);
        } else {
          S_WAITCNT_LGKMCNT::create(builder, op->getLoc(),
                                    builder.getI32IntegerAttr(lgkmcntVal));
          ticketing.observeLgkmWait(lgkmcntVal);
        }
        numWaitcntInserted++;
      }

      // Reset waits at loop ENTRY (LoopOp) since pending waits from before
      // the loop may not be valid inside. We do NOT reset at ConditionOp
      // (loop back-edge) because waits from the end of one iteration ARE
      // valid at the start of the next — the hardware state is continuous.
      // This avoids redundant lgkmcnt(0)/vmcnt(0) at the top of each
      // loop iteration.
      if (isa<LoopOp>(op) ||
          op->getName().getStringRef().contains("branch")) {
        ticketing.resetWaits();
      }
    };

    for (Operation *op : allOps)
      processOp(op);

    // Post-processing: combine adjacent separate s_waitcnt_vmcnt and
    // s_waitcnt_lgkmcnt into a single combined s_waitcnt instruction.
    // This happens when one waitcnt comes from the schedule (e.g.,
    // MemoryCounterWaitBarrier) and the other from the pass (e.g.,
    // barrier pre-wait), producing two adjacent separate waitcnts.
    combineAdjacentWaitcnts(program);
  }

  /// Merge adjacent S_WAITCNT_VMCNT + S_WAITCNT_LGKMCNT (in either order)
  /// into a single S_WAITCNT with both fields.
  void combineAdjacentWaitcnts(ProgramOp program) {
    llvm::SmallVector<std::pair<Operation *, Operation *>> toCombine;

    // Collect ops first to avoid walk-during-modification issues.
    llvm::SmallVector<Operation *> combineOps;
    collectOpsRecursive(program.getBodyBlock(), combineOps);

    for (Operation *op : combineOps) {
      Operation *next = op->getNextNode();
      if (!next)
        continue;

      auto vmcntOp = dyn_cast<S_WAITCNT_VMCNT>(op);
      auto lgkmcntOp = dyn_cast<S_WAITCNT_LGKMCNT>(next);
      if (vmcntOp && lgkmcntOp) {
        toCombine.push_back({op, next});
        continue;
      }

      auto lgkmcntOp2 = dyn_cast<S_WAITCNT_LGKMCNT>(op);
      auto vmcntOp2 = dyn_cast<S_WAITCNT_VMCNT>(next);
      if (lgkmcntOp2 && vmcntOp2) {
        toCombine.push_back({op, next});
        continue;
      }
    }

    // Track erased ops to avoid double-free when consecutive toCombine
    // entries share an operation (e.g. vmcnt/lgkmcnt/vmcnt triple).
    llvm::DenseSet<Operation *> erased;
    for (auto [first, second] : toCombine) {
      if (erased.contains(first) || erased.contains(second))
        continue;

      int64_t vmcntVal = 0;
      int64_t lgkmcntVal = 0;

      if (auto vmcnt = dyn_cast<S_WAITCNT_VMCNT>(first)) {
        vmcntVal = vmcnt.getCountAttr().getValue().getSExtValue();
        lgkmcntVal = dyn_cast<S_WAITCNT_LGKMCNT>(second)
                         .getCountAttr()
                         .getValue()
                         .getSExtValue();
      } else {
        lgkmcntVal = dyn_cast<S_WAITCNT_LGKMCNT>(first)
                         .getCountAttr()
                         .getValue()
                         .getSExtValue();
        vmcntVal = dyn_cast<S_WAITCNT_VMCNT>(second)
                       .getCountAttr()
                       .getValue()
                       .getSExtValue();
      }

      OpBuilder builder(first->getContext());
      builder.setInsertionPoint(first);
      S_WAITCNT::create(builder, first->getLoc(),
                        builder.getI32IntegerAttr(vmcntVal),
                        builder.getI32IntegerAttr(lgkmcntVal), IntegerAttr());

      erased.insert(first);
      erased.insert(second);
      second->erase();
      first->erase();
    }
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMInsertWaitcntPass() {
  return std::make_unique<InsertWaitcntPass>();
}

std::unique_ptr<mlir::Pass>
createWAVEASMInsertWaitcntPass(bool insertAfterLoads) {
  return std::make_unique<InsertWaitcntPass>(insertAfterLoads);
}

} // namespace waveasm
