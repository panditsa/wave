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

**Approach C: CSE within BufferLoadStrengthReduction (committed, partially effective)** — Added `localCSERange()` after `cloneChainBeforeLoop` to merge structurally identical cloned chains, plus `decomposeVoffset()` to strip both constants (→ instOffset) and SGPR addends (→ per-load `s_add_u32(group_soff, sgpr_delta)`). **Works correctly for synthetic tests** where chains are truly identical, but the real GEMM kernel has chains that are **genuinely structurally different** at the WaveASM level. Each read's preshuffle mapping formula bakes per-read constants (n_idx, k_idx) into intermediate `v_mul_lo_u32` / `v_sub_u32` ops deep in the tree, so the resulting VALU chains are not CSE-able — different operands at every level, not just at the final `V_ADD_U32`. Confirmed via debug output: 50 candidates → 50 unique voffsets after CSE, 24-member groups with 23 mismatches.

**Approach D (root fix, required): Factor expressions at sympy/Python level before MLIR emission** — The per-read constants must be separated from thread-dependent terms BEFORE `gen_sympy_index` emits MLIR. Use probing: substitute concrete values for thread symbols (THREAD_0/1/2), evaluate the linearized address for each read, and observe that all reads sharing the same `(k_idx)` produce `shared_thread_function + per_read_constant_offset`. Emit `shared_thread_function` once via `gen_sympy_index`, then each read's voffset is `shared_base + constant_delta` (simple `arith.addi`). This makes the WaveASM chains structurally identical (shared base), and the existing BufferLoadSR CSE + constant-stripping + SGPR-splitting then works.

**How probing works**: For a linearized address expression `f(T0, WG0, WG1, n_idx, k_idx)`:
1. Pick reference read (n_idx=0, k_idx=0), get its base_expr (with iv=0).
2. For each other read, compute `delta = other_base_expr - reference_base_expr`.
3. `delta` should be free of THREAD_0/1/2 (thread-independent constant offset). Verify by checking `delta.free_symbols ∩ {THREAD_0, THREAD_1, THREAD_2} == ∅`.
4. Emit `gen_sympy_index(reference_base_expr)` once as the shared VGPR base.
5. Emit each delta as `gen_sympy_index(delta)` — which will be SGPR or constant since it has no thread symbols.
6. Each read's voffset = `arith.addi(shared_base, delta_val)`.

This reduces N unique VALU chains to 1 shared chain + N-1 simple additions.

### Current state of Approach D (committed, 256 → 252 VGPRs)

**Python-level** (`read_write.py`):
- `_linearize_mapping_first()`: Applies divisibility subs (`K → 256*_K_div_256`) so `K/2 = floor(K/2) = 128*_K_div_256`, then linearizes with `linearize_dims` which cancels `floor(within_nblk/K_PACKED)*K_PACKED + Mod(within_nblk,K_PACKED) → within_nblk`. The linearized base becomes `(n_it//16)*16*K_PACKED + within_nblk` — no K_PACKED floor/Mod.
- `_try_reuse_hoisted_base()`: Caches the first read's linearized base (per loop × stride group). For subsequent reads, computes `delta = simplify(mem_simplify(lin_sym - ref_sym))`. If delta is free of THREAD symbols, emits `arith.addi(cached_base, gen_sympy_index(delta))` instead of the full chain.
- **27 out of 28 B-data reads reuse the cached base**. Deltas are clean: `2048*_K_div_256` (N-tile offset × K_PACKED), `1024` (K-half constant). One read is the reference (cached).

**C++ level** (`BufferLoadStrengthReduction.cpp`):
- `localCSERange()`: CSEs cloned pre-loop chains.
- `decomposeVoffset()`: Strips constants (→ instOffset) and single-level SGPR addends (→ per-load soffset).
- `_combineSGPRParts()`: Pre-combines multiple SGPR parts before the loop, skipping provably-zero values (s_mov_b32(0), s_mul_i32(0, x)).

**C++ level** (`Utils.h`):
- `getConstantValue()`: Now handles `S_MOV_B32(ConstantOp)` in addition to `V_MOV_B32(ConstantOp)`.

**Result**: 256 → 252 VGPRs. All B-data reads share a single VGPR base at the MLIR level. The remaining gap: after `cloneChainBeforeLoop`, each precomputed voffset is `V_ADD_U32(V_ADD_U32(shared_base, sgpr_delta_N), s_mul_i32(ivInit, stride))`. Single-level `decomposeVoffset` strips the outer SGPR (iv-init term, same for all reads), leaving `V_ADD_U32(shared_base, sgpr_delta_N)` as vgprBase — still different per read.

### Remaining blocker: multi-level SGPR splitting → SGPR allocation failure

Multi-level `decomposeVoffset` (stripping both iv-init and delta SGPRs to reach shared_base) is **mathematically correct** but causes "Failed to allocate SGPR" in the linear scan allocator. The pattern: 12 unique per-load `s_add_u32(group_soff, combined_delta)` in the loop body creates too many simultaneously-live SGPR values. Peak SGPR pressure rises to 87 (under the 102 limit), but the allocator fails — likely due to live range fragmentation from the interleaved `s_add_u32` ops across the loop body.

**Fix paths**:
1. **Schedule per-load `s_add_u32` right before their consuming buffer_load** to minimize SGPR live range overlap (currently done but loads are spread across the loop)
2. **Improve SGPR allocator** to handle the pattern of many short-lived SGPRs
3. **Reduce unique deltas** — some deltas are identical across reads (e.g., k_idx=0 reads at same n_idx share delta=0); cache dedup of `s_add_u32` results is implemented but insufficient

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
