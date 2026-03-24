# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Issue Tracking

This project uses **bd (beads)** for all issue tracking. Use `bd` exclusively — do NOT use TodoWrite, TaskCreate, or markdown files for task tracking.

Run `bd prime` for full workflow context. Key commands:

```bash (on host)
bd ready                                   # Find unblocked work to pick up
bd show <id>                               # View issue details and dependencies
bd create --title="..." --description="..." --type=task|bug|feature --priority=2
bd update <id> --status=in_progress        # Claim an issue before starting work
bd close <id1> <id2> ...                   # Mark issues complete
bd dep add <issue> <depends-on>            # Add dependency between issues
bd search <query>                          # Search issues by keyword
bd dolt push                               # Push beads to remote
```

Priority scale: 0=critical, 1=high, 2=medium, 3=low, 4=backlog. Do NOT use text labels like "high"/"medium".

**Workflow**: Create a beads issue BEFORE writing code. Mark `in_progress` when starting. Close when done. For persistent knowledge across sessions, use `bd remember "insight"` and `bd memories <keyword>` — not MEMORY.md files.

**WARNING**: Do NOT use `bd edit` — it opens `$EDITOR` which blocks agents.

## What This Project Is

Wave is a Python DSL for writing high-performance ML GPU kernels targeting AMD GPUs (MI250, MI300, MI350, RDNA4). It sits atop PyTorch FX graph tracing and IREE/MLIR compilation. The `@wave` decorator traces Python kernel functions into a computation graph, which flows through Wave's compiler to generate optimized GPU binaries.


## GOAL

Reduce VGPR pressure (primary) and SGPR pressure (secondary) in the WaveASM register allocator so that the schedule compiles and produces correct results.

**Context**: The goal is the same as before — reduce register pressure so WaveASM can allocate. Both `interleaved_mma_0` and `interleaved_mma_1` exist in both the current branch and main. The key difference is the **`base_offsets` and `base_intervals`** used for interleaving, which cause the correctness/pressure issues.

## Schedule Variants

In `wave_lang/kernel/wave/schedules/gemm_mxfp4_double_buffer.py`:

### Working baseline

```python
        base_offsets = [0, 3, 2, 0]
        base_intervals = [4, 4, 2, 4]
```

`interleaved_mma_1` **merges** A data and A scale loads into one group, leaving group 3 empty:
```python
        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                loop_g2s_a[0:4] + [loop_g2s_a_scale[0]] + loop_g2s_a[4:8] + [loop_g2s_a_scale[1]],
                loop_shared_load_a_0,      # group 1
                loop_shared_load_a_scale_0,# group 2
                [],
            ],
            ...
        )
```

### tighter interleaving — higher register pressure

```python
        base_offsets = [0, 1, 1, 0]
        base_intervals = [2, 2, 2, 4]
```

`interleaved_mma_1` **merges** A data and A scale loads into one group, leaving group 3 empty:
```python
        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                loop_g2s_a[0:4] + [loop_g2s_a_scale[0]] + loop_g2s_a[4:8] + [loop_g2s_a_scale[1]],
                loop_shared_load_a_0,
                loop_shared_load_a_scale_0,
                [],                        # group 3: empty
            ],
            ...
        )
```

The tighter `base_offsets` (`[0,1,1,0]` vs `[0,3,2,0]`) and smaller `base_intervals` (`[2,2,2,4]` vs `[4,4,2,4]`) pack memory ops closer together between MFMAs, increasing live range overlap and pushing VGPR pressure over the 256 limit. The WaveASM allocator needs pressure improvements to handle this tighter schedule.

### Toggling between schedules

In `gemm_mxfp4_double_buffer.py` around line 1765, there is a toggle to switch between the passing and failing schedule:

```python
        if True: # Tighter interleaving not passing because of register pressure
            base_offsets = [0, 1, 1, 0]
            base_intervals = [2, 2, 2, 4]
        else:
            base_offsets = [0, 3, 2, 0]
            base_intervals = [4, 4, 2, 4]
```

Set `True` → `False` to switch to the passing (main) schedule.

## Reproducing

```bash
# Enter the docker container
docker exec -it 27d65447d0a7fa35086019f0ca9d9019a11b6758178b6f491c93a414ded9a573 zsh
cd /workspace/wave_ref && source .venv/bin/activate

# Build waveasm backend after changes
cd /workspace/wave_ref/waveasm/build && ninja
```

### Running the tests

```bash
cd /workspace/wave_ref/examples/python

# WaveASM backend — NEW schedule (fails with register pressure)
HIP_VISIBLE_DEVICES=2 python 7.1_schedule.py --test test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm --block 256,192,256 --shape 13056,43392,2048

# WaveASM backend — OLD schedule (passes, must not regress)

```

### Inspecting Generated Assembly

| File | Description |
|------|-------------|
| `examples/python/build/intermediates/gemm.rocmasm` | **Live** — WaveASM output, overwritten every run |

## WaveASM Register Allocator Architecture

The allocator lives in `waveasm/` and has no spilling — compilation fails if pressure exceeds limits.

### Key Files

| File | Purpose |
|------|---------|
| `lib/Transforms/Liveness.cpp` | Liveness analysis, live range construction, pressure computation |
| `include/waveasm/Transforms/Liveness.h` | `LiveRange`, `TiedClass`, `LivenessInfo` types |
| `lib/Transforms/LinearScanRegAlloc.cpp` | Linear scan allocator (VGPR, SGPR, AGPR) |
| `include/waveasm/Transforms/RegAlloc.h` | `RegPool`, `PhysicalMapping`, `AllocationStats` |
| `lib/Transforms/LinearScanPass.cpp` | MLIR pass wrapper; includes `rematerializeCheapOps()` |
| `lib/Transforms/VGPRCompaction.cpp` | Post-alloc compaction (reduces fragmentation, not pressure) |
| `include/waveasm/Target/AMDGCN/ABI.h` | Hardware limits: `kMaxVGPRsGFX9 = 256`, `kMaxSGPRsGFX9 = 102` |

### How Allocation Works

1. **Liveness** (`computeLiveness`): Linearizes ops, builds def/use points, constructs `LiveRange` per SSA value, extends ranges for loop-carried values, builds tied equivalence classes (loop args, MFMA acc operands).
2. **Pressure** (`computeMaxPressure`): Sweep-line over start/end events. Tied classes counted once per envelope. Reports `maxVRegPressure`, `maxSRegPressure`.
3. **Rematerialization** (`rematerializeCheapOps`): Clones `V_MOV_B32` with immediate operands near each use site, shortening live ranges. Only targets ranges longer than `kMinRematRangeLength = 10`.
4. **Linear scan** (`allocateRegClass`): Ranges sorted by `(start, -size, -alignment, -end)`. Bidirectional allocation for multi-dword VGPRs: long-lived (top 25% by length) from top, short-lived from bottom. Ceiling is `maxPressure`. No spilling — fails with `emitOpError` if pool exhausted.
5. **Compaction** (`VGPRCompaction`): Reassigns physical regs to close gaps. Does not reduce peak pressure.

### Current Weaknesses (Opportunities for Improvement)

- **No spilling**: Any pressure above 256 VGPRs is a hard failure. Even modest improvements to live range management could make the difference.
- **Rematerialization is limited**: Only targets `V_MOV_B32` with immediate operands. Could be extended to other cheap-to-recompute values (e.g., `S_MOV_B32`, address computations, bitcasts).
- **No live range splitting**: A long-lived range that is only used at the start and end still occupies a register for its entire lifetime. Splitting at "holes" (gaps where the value is not used) could reduce pressure significantly.
- **Tied class envelopes inflate pressure**: When loop-carried values are tied, the envelope covers the entire loop body even if individual members are short-lived. Smarter coalescing (or breaking ties when beneficial) could help.
- **No priority-based eviction**: When allocation fails, there is no attempt to evict a less-important range and retry. A priority heuristic (e.g., evict the longest-lived range with the most remaining lifetime) could recover from pressure spikes.

### Known Bugs (Low Priority — Not Blocking, Keep in Mind)

- **`emitSub(imm, SGPR)` silently produces VGPR instead of SGPR**: In `Handlers.h`, `emitSub` requires `isSGPRType(a.getType())` for the SALU path. When the minuend `a` is an immediate and subtrahend `b` is SGPR, it falls through to `V_SUB_U32`, producing a VGPR result. This breaks callers that assume scalar inputs yield scalar output (e.g., `emitCeilFromFloorQuotient` in `AffineHandlers.cpp` passes the VGPR `rem` to `S_CMP_NE_U32` which expects SGPR operands). Fix: materialize the immediate into an SGPR before using `S_SUB_U32` when `a` is imm but `b` is SGPR. Not currently triggered by the active schedule but reachable via symbolic ceildiv expressions.

## AITER Reference Kernel

`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s` is a hand-tuned AITER kernel for the same MXFP4 GEMM shape. Study its register usage patterns for inspiration:

- **Pre-planned register layout**: AITER uses a fixed, hand-assigned register plan. VGPRs for MMA accumulators, load destinations, and address computations are pre-assigned to non-overlapping ranges. This avoids the fragmentation that a greedy allocator creates.
- **Register reuse discipline**: Load destinations are reused as soon as they are consumed by MMA. No unnecessary live range extension.
- **SGPR efficiency**: SRD (Shader Resource Descriptor) registers are shared across similar loads via base+offset patterns rather than separate SRDs per buffer.

## Current State of Changes

### What was done (in `LinearScanPass.cpp`)

1. **Generalized `isRematerializableOp`** — Trait-based: accepts any op with `ArithmeticOp` trait, no `SCCDef` trait, single result, and cheap operands. Covers `V_ADD_U32`, `V_LSHLREV_B32`, `V_AND_B32`, `S_MUL_I32`, `S_MOV_B32` (with SGPR source), etc.

2. **VGPR-covering operand check** — `allOperandsCheap(op, &liveness)` now accepts VGPR operands (including `BlockArgument`s) whose live ranges fully contain the result's range. Previously only constants and SGPRs were accepted.

3. **Removed loop-body remat restriction** — VGPR ops defined outside a loop can now be cloned inside. The old guard `if (defOutsideLoop && isVGPR && isInsideLoopBody(user)) continue` was overly conservative — it prevented freeing VGPRs whose cross-loop live range (extended by Pass 2b) spanned the entire body.

4. **Multi-pass remat** — Runs up to 4 iterations since each pass creates new short-lived values enabling further candidates.

5. **Transitive remat (IMPLEMENTED BUT BUGGY — DISABLED)** — `isTransitivelyRematerializableOp` + `cloneTransitiveChain` can clone entire dependency chains near use sites. When enabled, pressure dropped 261→241 (under 256!) but produces **incorrect numerical results**. Disabled via empty `transitiveCandidates` vector.

### Results without transitive remat

| Schedule | Before | After |
|----------|--------|-------|
| Old (passing) | 254 VGPRs | **253 VGPRs** ✅ passes |
| Tight (failing) | 262 VGPRs | **261 VGPRs** ❌ still > 256 |

### VGPR pressure breakdown at peak (tight schedule, 261 VGPRs)

```
block_arg:              85 regs (25 vals)  — loop-carried, unavoidable
ds_read_b128:           56 regs (14 vals)  — LDS reads consumed by MFMA
v_add_u32:              53 regs (53 vals)  — ADDRESS COMPUTATION, main target
buffer_load_dwordx4:    48 regs (12 vals)  — B-matrix prefetch
ds_read_b32:             4 regs  (4 vals)  — scale values
buffer_load_dword:       3 regs  (3 vals)  — scale loads
other arith:            12 regs (12 vals)
```

The 53 `v_add_u32` address VGPRs are the only soft target. They're defined early in the loop body (positions 1188–1383), used for buffer_load addresses later (ending ~2492). Their operands are **short-lived VGPR intermediates** (not constants/SGPRs), so direct remat can't touch them. Transitive remat is required.

### The transitive remat bug

The bug is in how `cloneTransitiveChain` decides whether to use the ORIGINAL value vs recursing. Consider:

```
%a = v_cvt_u32_f32 %sgpr      → range [132, 137]
%b = v_mul_hi_u32 %a, %const   → range [136, 137]
%c = v_add_u32 %a, %b          → range [137, 1156]
```

`%c` is used at position 1156. When cloning `%b` near 1156, the code checks:
"does `%a`'s range [132,137] cover `%b`'s result range [136,137]?" → **YES** → uses original `%a`.

But `%a` is **dead at position 1156**! The covering check uses the **intermediate's result range** (136–137), not the **clone's insertion point** (1156).

**Two bugs:**

1. **In `isTransitivelyRematerializableOp`** — The covering check for recursive calls uses the intermediate's result range instead of the root's result range. A VGPR that covers `%b`'s range [136,137] might not cover `%c`'s range [137,1156]. Fix: pass the root's result range through all recursive calls and use it for all covering checks.

2. **In `cloneTransitiveChain`** — Even after fixing validation, the cloning must check the `cloned` map FIRST (to reuse already-cloned intermediates from earlier in the chain), then only use originals for true terminals (block_args with no defOp, SGPRs). For VGPR operands with defOps, always recurse — never use the stale covering check.

### Fix approach for transitive remat

```cpp
// isTransitivelyRematerializableOp: add rootRange parameter
static bool isTransitivelyRematerializableOp(
    Operation *op, const LivenessInfo &liveness, int depth,
    llvm::DenseSet<Operation *> &visited,
    const LiveRange *rootRange) {   // ← NEW: root's range, not intermediate's
  ...
  for (Value operand : op->getOperands()) {
    // Covering check uses rootRange, not this op's result range
    if (isVGPRType(operand.getType())) {
      const LiveRange *opRange = liveness.getRange(operand);
      if (opRange && opRange->start <= rootRange->start &&
          opRange->end >= rootRange->end)
        continue;  // Terminal: alive at all user positions
    }
    // Recurse with same rootRange
    if (depth > 0 && defOp && isVGPRType(operand.getType())) {
      if (isTransitivelyRematerializableOp(defOp, liveness, depth - 1,
                                           visited, rootRange))
        continue;
    }
    return false;
  }
}

// cloneTransitiveChain: check cloned map first, no covering check
for (Value operand : op->getOperands()) {
  if (defOp && cloned.count(defOp))       { use cloned[defOp]; continue; }
  if (defOp && isa<ConstantOp>(defOp))    { clone constant;    continue; }
  if (isSGPRType(operand.getType()))       { use original;      continue; }
  if (defOp && isVGPRType(...))            { recurse;            continue; }
  // BlockArgument (no defOp): use original — it's a loop-carried
  // value alive at all user positions (validated by rootRange check above)
  use original;
}
```

### Important architecture note

The allocator pass is invoked with `--waveasm-linear-scan=max-vgprs=512 max-agprs=512` (in `compile.py` line 1329). The pool has 512 slots, NOT 256. The `VGPRCompaction` pass runs afterward to pack registers down. The `.vgpr_count` in the assembly metadata is computed from the highest physical register index, not from the pool. So VGPR pressure > 256 produces out-of-range register indices that the assembler rejects.

## Lessons Learned (Session 2024-03-24)

### DO NOT add VALU instructions inside the loop body

The single most important constraint: **ZERO new VALU ops in the main MFMA loop**. In MFMA-dominated GEMM loops, every VALU instruction competes with MFMAs for issue slots. The AITER reference kernel has ~425 VALU address ops; adding more is unacceptable.

Approaches that violated this rule and MUST NOT be repeated:
- **Transitive rematerialization inside loops**: Cloned entire address computation chains (thread-ID decomposition, swizzle, row/col) near each ds_read use site. Inflated VALU from 445 → 1530 instructions. Even the "fixed" version with rootRange correctness was terrible for performance.
- **Direct rematerialization of VGPR arithmetic inside loops**: Cloning ops like `v_add_u32`, `v_and_b32`, `v_lshrrev_b32` inside loops (even single-op clones) multiplies address computation. Only `V_MOV_B32` (from immediate) and `S_MOV_B32` should ever be cloned inside loops.
- **Any rematerialization that targets ops defined INSIDE the loop body**: These are the original address computations; cloning them creates duplicates.

### The 445 VALU address ops are from the original MLIR IR, not from remat

The address computation chains (12 VALU ops before each `ds_read`) are emitted by the MLIR translation. They compute LDS addresses from thread ID + swizzle. The `--waveasm-loop-address-promotion` pass hoists them out, but requires VGPR headroom (currently disabled because it pushes VGPRs over 256).

### MFMA reordering does NOT reduce register pressure

Tested N-major 2×2 micro-tile reordering (hold B/N fixed, sweep A/M in pairs). Result: **pressure INCREASED** from 256 → 268 VGPRs.

Root cause: A-data is pre-loaded before the scheduling barrier (same as AITER). ALL A-data is live when MFMAs start regardless of consumption order. Reordering MFMAs to N-major means both M:0 and M:1 A-data must be live within each micro-tile, extending A-data live ranges.

AITER's MFMA ordering is about **latency hiding** (keeping the MFMA pipeline fed), not about register pressure. Their A-data is also fully pre-loaded in a burst of 16 ds_reads.

### Our schedule structure already matches AITER

Both use the same pattern:
1. **Prologue**: Burst-load A-data (ds_read) for the first phase
2. **Phase 0**: Pre-loaded A-data + MFMAs, interleaved with next-phase A-data prefetch
3. **Phase 1**: Pre-loaded A-data + MFMAs, interleaved with next-iteration A-data prefetch

This is the 2-partition cross-prefetching in our `interleave_operations` setup.

### The real pressure gap vs AITER is address computation

| Component | AITER (256×256) | Ours (256×192) |
|-----------|-----------------|----------------|
| A-data (ds_read from LDS) | 128 VGPRs | ~56 VGPRs |
| B-data (buffer_load from global) | 64 VGPRs | ~48 VGPRs |
| Accumulators | 256 AGPRs | 192 AGPRs |
| Address/misc VGPRs | **~10 VGPRs** | **~50 VGPRs** |
| Total VGPRs | 251 | 256 |

AITER precomputes all addresses with loop-invariant VGPRs + `offset:` immediates on ds_read. We recompute 12 VALU ops per ds_read inline. The ~40 VGPR gap is entirely from address computation overhead.

### Dynamic scratch VGPR and result-as-scratch

The v15 scratch VGPR reservation costs 1 usable VGPR slot. Solutions explored:
- **Dynamic placement** (scratchVGPR = maxAllocated): Works but if pressure = 256, scratch goes to v256 (out of range).
- **Result-as-scratch** for `v_cndmask_b32`, `v_add_u32`, VOP3+ literal materialization: Use the instruction's result register as the temporary for `v_mov_b32` literal load. Eliminates the need for a separate scratch VGPR in most cases. E.g., `v_mov_b32 v240, 128; v_add_u32 v240, v249, v240` instead of `v_mov_b32 v_scratch, 128; v_add_u32 v240, v249, v_scratch`.
- **VOP2 instructions** (`v_mul_f32`, `v_add_f32`, etc.) can accept literals in src0 directly, avoiding scratch entirely.

### Transitive remat rootRange fix is correct but should NEVER run inside loops

The bug in `isTransitivelyRematerializableOp` was real: covering checks used the intermediate's result range instead of the root's range. An operand that covers `[136,137]` was accepted as a terminal but dead at the root's use site at position 1156. The fix (pass `rootRange` through all recursive calls) is correct. But transitive remat must be disabled inside loop bodies to avoid VALU bloat.

## Strategy

The goal is general-purpose improvements to the WaveASM allocator — not special-casing this schedule. Focus areas:

1. ~~**Extend rematerialization**~~ ✅ Done (generalized to all Pure ArithmeticOp).
2. ~~**Fix transitive rematerialization**~~ ✅ rootRange fix is correct, but MUST NOT be used inside loop bodies.
3. **Reduce address computation VGPRs** — The ~50 VGPRs used for inline address computation vs AITER's ~10 is the key gap. Path: enable `--waveasm-loop-address-promotion` which requires VGPR headroom.
4. **Live range splitting** at holes — if a value has a gap between its last use in one region and next use in another, split the range and free the register during the gap.
5. **Smarter tied class handling** — avoid tying values when it would extend the envelope far beyond the individual member lifetimes.
6. **SGPR pressure** — look for opportunities to hoist scalar computations out of loops or share SRD registers.
7. **Improve bidirectional allocation heuristic** — the current 75% threshold for "long-lived" is arbitrary.

## Architecture

### Compilation Pipeline

```
Python kernel (@wave decorator)
  -> FX graph tracing (PyTorch)
  -> Wave IR (expansion, constraint resolution, scheduling)
  -> MLIR emission (via fx_emitter or water_emitter)
  -> IREE compilation or WaveASM (direct AMDGCN assembly)
  -> GPU binary
```

### Key Concepts

- **`@wave` decorator** (`wave_lang/kernel/wave/wave.py`) traces kernel functions and handles compilation/dispatch.
- **Constraints** (`wave_lang/kernel/wave/constraints.py`) define how computation maps to GPU hardware — workgroups, waves, and thread layouts are expressed as constraints rather than imperative code.
- **Symbolic dimensions**: Kernel authors use symbolic SymPy variables for tensor shapes; the compiler resolves them at specialization time.
- **Dual emission paths**: Kernels can be emitted via IREE/MLIR (default) or via WaveASM for direct AMDGCN assembly generation.
- **Software pipelining**: The scheduling pass (`wave_lang/kernel/wave/scheduling/`) applies modulo scheduling with the APLP Rust solver to maximize ILP.

### Directory Map

| Directory | Purpose |
|-----------|---------|
| `wave_lang/kernel/wave/` | Core Wave DSL compiler (main compilation pipeline) |
| `wave_lang/kernel/wave/compile.py` | Main compilation entry point |
| `wave_lang/kernel/wave/analysis/` | Index sequence analysis, partition strided operators |
| `wave_lang/kernel/wave/expansion/` | Graph expansion passes |
| `wave_lang/kernel/wave/scheduling/` | Instruction scheduling and software pipelining |
| `wave_lang/kernel/wave/scheduling/aplp/` | Rust-based APLP scheduling solver (PyO3 bindings) |
| `wave_lang/kernel/wave/mlir_converter/` | FX graph to MLIR conversion (fx_emitter, water_emitter) |
| `wave_lang/kernel/wave/templates/` | Pre-built kernel templates (GEMM, attention, conv, MoE, etc.) |
| `wave_lang/kernel/wave/runtime/` | C++ GPU runtime with nanobind Python bindings |
| `wave_lang/kernel/wave/asm/` | Assembly generation driver |
| `wave_lang/kernel/wave/memory_analysis/` | Shared memory allocation minimization |
| `wave_lang/kernel/compiler/wave_codegen/` | MLIR emission handlers |
| `wave_lang/kernel/lang/` | Language primitives (types, grids, kernel buffers) |
| `wave_lang/kernel/ops/` | Operation definitions (math, memory, control flow, reduction) |
| `wave_lang/runtime/` | Device management, kernel launch, multi-device |
| `wave_lang/debugging/` | HTML-based schedule visualization |
| `water/` | "Water" MLIR dialect — C++ middle-end passes and dialect |
| `waveasm/` | WaveASM backend — MLIR to AMDGCN assembly translation |
| `lit_tests/` | LLVM LIT/FileCheck tests for codegen verification |
| `tests/unittests/` | Python unit tests (scheduling, constraints, graph utils, type inference) |
| `tests/kernel/` | Kernel-level tests (mostly require AMD GPU) |

### Tech Stack

- **Python** — DSL, compiler passes, tests
- **C++17** — Water MLIR dialect, WaveASM backend, GPU runtime (nanobind)
- **Rust** — APLP scheduling solver (PyO3)
- **MLIR/TableGen** — Dialect definitions for Water and WaveASM
- **PyTorch** (≥2.6) — FX graph tracing, tensor abstractions
- **IREE** (~3.11.0rc) — MLIR compiler and runtime for GPU codegen
- **SymPy** — Symbolic index expressions and constraint solving

## C++ Coding Conventions (Water/WaveASM)

These follow LLVM standards with additional rules:
- Prefer LLVM ADT/Support data structures over STL
- Only use `auto` when the RHS is a cast, constructor, static get, or iterator
- Pass `Attribute`, `Type`, `Value` by value (they are cheap to copy)
- Never use integer literals in `SmallVector` template parameters
- Mark functions `static` instead of using anonymous namespaces
- No braces around single-line statements
- End comments with full stops; start diagnostic/assertion messages lowercase, no period
