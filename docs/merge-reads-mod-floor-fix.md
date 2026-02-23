# Merge Reads: Mod/Floor Contiguity Fix

## Problem

For MXFP4 GEMM kernels with certain shapes (e.g., 32768x57344x16384 where
`K_SCALE_SHUFFLED = 512`), B-scale loads were emitted as individual
`buffer_load_ubyte` instructions instead of single `buffer_load_dword`
instructions. This caused:

- 4x more memory instructions for B-scale loads (16 ubyte vs 4 dword)
- Extra `v_lshl_or_b32` byte-packing instructions in the loop body
- Wasted VGPR pressure carrying individual bytes through `scf.for` iter_args

The working shape (1024x1024x8192, `K_SCALE_SHUFFLED = 256`) produced
`buffer_load_dword` correctly.

## Root Cause

The merge reads pass (`partition_strided_operators.py`) combines adjacent
`vector<1xi8>` loads into wider `vector<4xi8>` loads. It checks contiguity
by computing the flat-offset difference between two reads and verifying it
equals the current `elements_per_thread`.

The B-scale buffer uses a preshuffle mapping (`b_scale_mapping` in
`tagged_mxfp4_gemm.py`) that decomposes a flat index into 2D physical
coordinates:

```
phys_N = b_scale_flat // K_SCALE_SHUFFLED
phys_K = b_scale_flat % K_SCALE_SHUFFLED
```

The merge pass reconstructs the flat offset as
`flat_offset = phys_N * KSS + phys_K`. Mathematically this equals
`b_scale_flat` (by the identity `x = floor(x/M)*M + Mod(x, M)`), but
**sympy does not recognize this identity**. The flat_offset remains
expressed as `Mod(bflat, M) + M*floor(bflat/M)` and sympy cannot simplify
the diff of two such expressions.

### Why KSS=256 worked

When `K_SCALE_SHUFFLED = 256`, the `b_scale_flat` formula contains
`(k_s // 8) * 256`. Since 256 is a multiple of `KSS = 256`, the `Mod(..., 256)`
drops this term, producing simpler expressions that `sym_simplify` can reduce.

### Why KSS=512 failed

When `K_SCALE_SHUFFLED = 512`, the term `(k_s // 8) * 256` is **not** a multiple
of `KSS = 512`, so `Mod(..., 512)` retains it. The resulting expressions contain
nested `Mod`/`floor` compositions that sympy cannot reduce to constants.

## Fix

Three changes, each addressing a different level of the simplification pipeline:

### 1. Mod expansion in `simplify()` (symbol_utils.py)

Added the identity `Mod(x, M) = x - M*floor(x/M)` as a rewrite rule. When the
standard bounds-based simplification reaches a fixed point but `Mod` terms remain
inside an `Add`, the rewrite expands them. This lets `floor` terms from the 2D
decomposition cancel algebraically with their `Mod` counterparts, recovering the
original flat index where the diff is trivially 1.

```python
# In simplify(), after standard passes reach fixed point:
if _has_mod_in_add(new_expr):
    expanded = _expand_mod(new_expr)  # Mod(x,M) -> x - M*floor(x/M)
    expanded = sympy.simplify(expanded)
```

This is a **general mathematical identity** — it applies to any expression
containing `Mod` inside sums, not just B-scale addresses.

### 2. Lane-isolation fallback (partition_strided_operators.py)

For cases where the Mod expansion alone is insufficient (edge cases in the
iterative merge rounds), a fallback strips non-lane symbols from the diff
before re-simplifying.

**Justification**: Two reads in a merge candidate pair come from the same
thread, same wave, same workgroup, same loop iteration. Therefore:

| Symbol | Meaning | In the diff |
|--------|---------|-------------|
| `$WG0`, `$WG1` | `blockIdx.x/y` | Identical in both reads, cancels |
| `$T1` | `threadIdx.y` (wave ID) | Identical in both reads, cancels |
| `$ARGK` | Loop induction variable | Identical in both reads, cancels |
| `$T0` | `threadIdx.x` (lane ID) | Different mapping functions, does NOT cancel |

Substituting the cancelling symbols to 0 isolates the `$T0`-only expression.
The bounds-based simplifier handles the remaining expression because
`Mod($T0, 16)` has known bounds `[0, 15]`, enabling `floor` resolution:

```
floor(Mod($T0, 16) / 16)              -> 0   (bounds [0, 15/16])
floor(Mod(Mod($T0, 16) + 16, 32) / 16) -> 1   (bounds [16/16, 31/16])
```

### 3. Stride-1 dimension fallback (partition_strided_operators.py)

When the flat-offset diff is proven (by method 1 or 2) but the per-dimension
decomposition cannot symbolically identify which dimension advances, the
innermost (stride-1) dimension is inferred as the merge dimension. This is safe
when `ept < min(non-unit strides)`, guaranteeing no row-boundary crossing.

## Assumptions

1. **GPU execution model**: All threads in a workgroup execute the same code on
   different data tiles. Workgroup IDs, wave IDs, and loop induction variables
   contribute identical offsets to both reads in a merge pair.

2. **Thread ID structure**: `$T0 = threadIdx.x` (lane within wave, 0-63),
   `$T1 = threadIdx.y` (wave ID). This follows from
   `linearized_thread_id = $T0 + $T1 * threads_per_block[0]` where
   `threads_per_block[0] = waves_per_block[0] * 64`.

3. **Mod expansion safety**: The identity `Mod(x, M) = x - M*floor(x/M)` is
   universally valid for positive integer `M`. The expansion may increase
   expression complexity; it is only applied when standard simplification has
   already reached a fixed point and `Mod` terms remain in `Add` nodes.

4. **Stride-1 fallback**: When `ept < min(non-unit strides)`, adjacent flat
   addresses must lie in the same row of the 2D memref layout. The innermost
   dimension (stride 1) is the only one that can advance by `ept`.

## Files Changed

| File | Change |
|------|--------|
| `wave_lang/kernel/wave/utils/symbol_utils.py` | Added `_expand_mod`, `_has_mod_in_add`, and Mod-expansion step in `simplify()` |
| `wave_lang/kernel/wave/analysis/partition_strided_operators.py` | Added `_hw_simplified_diff` fallback and stride-1 dimension fallback in `_merge_contiguous_reads_once` |

## Validation

```bash
# Default shape (KSS=256) -- must still pass:
python 7.1_schedule.py --test test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp --block 128,256,256

# Target shape (KSS=512) -- must produce buffer_load_dword, not buffer_load_ubyte:
python 7.1_schedule.py --test test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp \
    --shape 32768,57344,16384 --block 128,256,256

# Assembly checks:
rg 'buffer_load_ubyte' build/intermediates/preshuffle_b_cpp.s | wc -l   # 0
rg 'buffer_load_dword\b' build/intermediates/preshuffle_b_cpp.s | wc -l  # 12
```
