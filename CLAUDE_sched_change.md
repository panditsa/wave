# Register Pressure Reduction — Session Notes

## Goal

Reduce VGPR pressure in the WaveASM GEMM kernel to enable the tight interleaving schedule (`base_offsets=[0,1,1,3]`, `base_intervals=[2,4,3,6]`) and eventually support 256×256 tile sizes. The tight schedule packs memory ops closer to MFMAs for better latency hiding.

## Current State

| Schedule | VGPRs | VALU in loop | Status |
|----------|-------|-------------|--------|
| Old (`[0,3,2,0]`, `[4,4,2,4]`) | **253** | **0** | Passes |
| Tight (`[0,1,1,3]`, `[2,4,3,6]`) | **256** | **0** | Passes (at limit) |
| Original tight (`[0,1,1,0]`, `[2,2,2,4]`) | **261** | **0** | Fails (>256) |

The tight schedule with adjusted intervals passes at 256 VGPRs. The original tight intervals caused +8 VGPRs from excess ds_read live range overlap (A-data prefetch rate too aggressive).

## VGPR Pressure Breakdown (old schedule, 253 VGPRs at peak)

| Category | VGPRs | Values | Notes |
|----------|-------|--------|-------|
| `<block_arg>` (loop-carried) | **85** | 25 | Unavoidable: iter_args for accumulators, addresses, prefetch buffers |
| `v_add_u32` (address computation) | **53** | 53 | Prologue address chains extended into loop by Pass 2b |
| `buffer_load_dwordx4` (B-data) | **48** | 12 | Double-buffered B-matrix prefetch from global |
| `ds_read_b128` (A-data from LDS) | **48** | 12 | Double-buffered A-matrix from shared memory |
| `buffer_load_dword` (B-scale) | 3 | 3 | Scale values |
| `ds_read_b32` (A-scale) | 2 | 2 | Scale values |
| Other arith | 14 | 14 | v_and, v_cndmask, v_mul, v_mov, etc. |

AITER (256×256 kernel) uses 251 VGPRs with 256 AGPRs. Key difference: **AITER uses ~10 VGPRs for addresses vs our ~53**. AITER precomputes 8 voffset VGPRs (one per row pattern) reused across all loads; we compute 26 unique voffsets.

## What Was Done

### 1. Tight schedule interleaving (committed)

Adjusted `base_offsets` and `base_intervals` per partition to match AITER's load distribution:
- Partition 0: B-data at interval 2, A-data prefetch at interval 4 (matches AITER's rate)
- Partition 1: g2s_a + g2s_a_scale merged at interval 4 (prevents end-bunching)
- B-scale at interval 6, offset 3 (staggered from B-data, no dependency)
- Prologue g2s interleaved as 4+1 groups (4 A-data + 1 A-scale)

### 2. SALU support in BufferLoadStrengthReduction (committed)

Added `S_MUL_I32`, `S_ADD_U32`, `S_SUB_U32`, `S_MOV_B32` to the address op allowlist and `computeStaticStride()`. Previously all 50 buffer loads were rejected because `s_mul_i32` (computing `iv * stride`) wasn't in the allowlist. Now 50/50 candidates are processed, creating 6 SRD groups with shared soffset SGPR iter_args.

### 3. reorder_mmas_for_register_reuse utility (committed)

General-purpose MFMA reordering function in `wave_schedule_ops.py` for 2D micro-tile ordering. Not currently used in the schedule because MFMA reordering doesn't reduce register pressure when A-data is pre-loaded (see below).

## What Was Explored But Not Committed

### Rematerialization inside loop bodies — DO NOT DO THIS

Any form of rematerialization that adds VALU instructions inside the main MFMA loop is **unacceptable**:

- **Transitive rematerialization**: Cloned entire address computation chains (thread-ID decomposition, swizzle, row/col) near each ds_read. Inflated VALU from 0 → 1530 instructions. Even the "fixed" version (rootRange correctness fix) was terrible.
- **Direct remat of VGPR arithmetic**: Cloning v_add_u32, v_and_b32, etc. inside loops multiplies address computation. Only V_MOV_B32 (from immediate) and S_MOV_B32 should ever be cloned inside loops.
- The transitive remat rootRange fix IS correct (covering checks use root's range, not intermediate's), but transitive remat must NEVER run inside loop bodies.

### MFMA reordering — does NOT reduce pressure

Tested N-major 2×2 micro-tile reordering (hold B/N fixed, sweep A/M in pairs). Result: pressure INCREASED from 256 → 268 because A-data is pre-loaded before the scheduling barrier — ALL A-data is live when MFMAs start, regardless of consumption order.

AITER's MFMA ordering is about **latency hiding**, not register pressure. Their A-data is also fully pre-loaded in a burst of 16 ds_reads. Our schedule structure already matches AITER's 2-partition cross-prefetching pattern.

### v15 scratch VGPR removal — caused regression

Removing the v15 reservation and using result-as-scratch for literal materialization actually increased pressure (253 → >256). The allocator's compaction and packing order changed in ways that hurt.

### Bounds check / mask elimination — not the issue

The 474 `v_cndmask_b32` instructions in the kernel are NOT from per-load bounds checking. They're from **dynamic shape arithmetic** (workgroup remapping with `arith.select`, ceildiv/mod emulation). The SRD-based bounds checking via `_compute_valid_bytes` is already working correctly.

## Root Cause: Duplicate Address Computation Chains

Each of the 26 B-data reads independently emits the full address computation chain through `gen_sympy_index`. The expressions share a massive common subexpression — the Piecewise workgroup remapping:

```
Piecewise((Mod($WG0 + $WG1*ceiling(M/256) - ..., Max(1, ...)) + 32*floor(...),
           CONDITION),
          (FALLBACK, True))
```

This is identical across all 24 B-data reads. The only differences are:
- **K-step offset**: `+ 0` vs `+ 1024` (2 variants)
- **N-tile offset**: `+ 0, + 16, + 32, + 48, + 64, + 80` (6 variants)
- **$ARGK term**: present in loop reads, absent in prologue

Each read independently lowers the full expression via `gen_sympy_index`, creating 24+ copies of the same VALU chain. MLIR CSE downstream doesn't fully merge them due to the complexity.

## Next Steps: Sympy-Level CSE for Read Expressions

### What to implement

Apply `sympy.cse()` to ALL read_b index expressions as a batch before lowering to MLIR. This extracts the shared Piecewise subexpression as a single CSE replacement, emitted once, with per-read variations as small constant offsets.

### Why it should help

The shared Piecewise subexpression generates ~30 VALU ops (v_cndmask, v_and, v_lshr, v_mul, v_add for workgroup remapping). With 24 reads × ~30 ops = ~720 duplicate VALU ops in the prologue. CSE reduces this to ~30 shared + 24 × ~3 per-read = ~102 ops. This reduces:
1. **Prologue VALU**: 1406 → ~700 (fewer ops to execute during kernel init)
2. **Unique voffset VGPRs**: 26 → ~8-12 (reads sharing the same base voffset + constant instOffset)
3. **Peak VGPR pressure**: fewer simultaneously-live address VGPRs

### How to implement (approaches tried and issues)

**Approach A: Subexpression cache in gen_sympy_index** — Cache full expressions and replace cached sub-trees with dummy symbols during traversal. Issue: Piecewise expressions break when their conditions get replaced with dummy symbols, causing `Unsupported piecewise` errors.

**Approach B: Batch CSE in emitter pre-pass** — Collect all global read nodes in `_emit_graph`, apply `sympy.cse()` on all their start indices at once, store replacements + reduced expressions as node metadata, emit replacements once in `handle_read`. Issue: the CSE replacement values are emitted in the deep prologue, extending their live ranges into the loop body, actually INCREASING pressure.

**Approach C (recommended): CSE within BufferLoadStrengthReduction** — After `cloneChainBeforeLoop` creates precomputed voffsets, run MLIR-level CSE on the cloned chains. Since the clones are right before the loop (not deep in the prologue), their live ranges are short. Structurally identical chains (same ops, same operands) get merged to shared values. This is a C++ change in the WaveASM backend.

**Approach D (alternative): Restructure gen_sympy_index to emit shared subexpressions lazily** — Instead of caching at the full-expression level, modify `gen_sympy_index` to accept a shared expression context. Before emitting a read's offset, decompose the sympy expression into `shared_base + per_read_delta` at the Python level, emit `shared_base` once (hoisted), and emit `per_read_delta` as a simple constant addition. The key is to do the decomposition at the sympy level where the shared structure is visible, then pass the decomposed parts to the MLIR emitter. The shared base should be hoisted to a point that dominates all reads but is close to them (not deep in the prologue) to minimize live range extension.

### Key constraint

The CSE must NOT extend live ranges of shared values across the entire kernel. The shared subexpressions should be emitted at the latest dominating point — right before the first read that uses them, not at the top of the function. This requires careful insertion point management.

## Reproducing

```bash
docker exec -it 27d65447d0a7fa35086019f0ca9d9019a11b6758178b6f491c93a414ded9a573 zsh
cd /workspace/wave_ref && source .venv/bin/activate
cd waveasm/build && ninja  # rebuild waveasm after C++ changes

cd /workspace/wave_ref/examples/python
HIP_VISIBLE_DEVICES=2 python 7.1_schedule.py \
  --test test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm \
  --block 256,192,256 --shape 13056,43392,2048

# Check register counts:
grep "vgpr_count" build/intermediates/gemm.rocmasm

# Check loop body:
awk '/L_loop_0:/,/s_cbranch/' build/intermediates/gemm.rocmasm | wc -l
awk '/L_loop_0:/,/s_cbranch/' build/intermediates/gemm.rocmasm | grep -c "v_mfma"

# Check voffset reuse:
grep "buffer_load_dwordx4" build/intermediates/gemm.rocmasm | grep -v lds | \
  grep -oP "v\d+(?=,)" | sort | uniq -c | sort -rn
```

## Key Files

| File | Purpose |
|------|---------|
| `wave_lang/kernel/wave/schedules/gemm_mxfp4_double_buffer.py` | Schedule definition (interleaving, prologue, clusters) |
| `wave_lang/kernel/compiler/wave_codegen/read_write.py` | Read/write codegen (handle_read, _try_iv_split_offset, gen_sympy_index calls) |
| `wave_lang/kernel/compiler/wave_codegen/emitter.py` | gen_sympy_index (sympy → MLIR lowering) |
| `waveasm/lib/Transforms/BufferLoadStrengthReduction.cpp` | Hoists buffer_load voffsets, creates soffset iter_args |
| `waveasm/lib/Transforms/LinearScanPass.cpp` | Rematerialization + register allocation pass wrapper |
| `waveasm/lib/Transforms/Liveness.cpp` | Liveness analysis, pressure computation |
| `waveasm/lib/Transforms/LinearScanRegAlloc.cpp` | Linear scan allocator |
| `waveasm/lib/Transforms/VGPRCompaction.cpp` | Post-alloc register compaction |
| `wave_lang/kernel/ops/wave_schedule_ops.py` | reorder_mmas_for_register_reuse utility |
