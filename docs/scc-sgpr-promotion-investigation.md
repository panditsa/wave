# SCC & SGPR Promotion Investigation

## Summary

This document captures the findings from investigating SCC (Scalar Condition Code) tracking and VGPR→SGPR promotion in the WaveASM backend. The goal is to reduce VGPR pressure by ~40-49 registers to match the aiter reference kernel.

## What was built

### SCC Infrastructure (committed, working)

1. **SCCDef / SCCUse traits** — every SALU op now correctly declares its SCC behavior via ODS traits
2. **SALUBinaryWithSCCOp / SALUUnaryWithSCCOp** — op classes with `NoMemoryEffect + AlwaysSpeculatableImplTrait + SCCDef` (equivalent to Pure for MLIR passes, but identifiable as SCC-clobbering)
3. **SCC verifier pass** (`--waveasm-scc-verifier`) — walks IR in emission order, catches SCC clobber between producer and consumer
4. **ScopedCSE guard** — SCCUse ops excluded from CSE (result depends on implicit SCC)

Key insight: `Pure` in MLIR ODS = `NoMemoryEffect + AlwaysSpeculatableImplTrait`. Composing these with `SCCDef` keeps MLIR pass behavior (DCE, LICM) identical while enabling trait-based SCC verification. Adding `NativeOpTrait` to non-Pure ops (SALUBinaryWithCarryOp, SALUCmpOp) is safe; adding it to Pure ops requires the explicit composition to avoid changing codegen.

### Handler SALU Promotions (committed, working)

| Handler | Change | Impact |
|---------|--------|--------|
| `handleArithOrI` | `emitOr` (S_OR_B32 when scalar) | No trigger in GEMM kernel |
| `handleArithXorI` | `emitXor` (S_XOR_B32 when scalar) | No trigger in GEMM kernel |
| `handleArithDivUI` | `emitLshr` for power-of-2 | No trigger (VGPR operands) |
| `handleArithRemUI` | `emitAnd` for power-of-2 | No trigger (VGPR operands) |
| `handleArithMaxSI/UI/MinSI/UI` | `emitMaxI32` etc. | 1 promotion: `v_max_i32 → s_max_i32` |

### V_CMP Constant Bus Fix (committed, working)

`handleArithCmpI` vector path: only `ensureVGPR` when BOTH operands are SGPR (V_CMP VOP3 accepts one SGPR via constant bus). Eliminates 2 `v_mov_b32 sN→vN` instructions.

`handleArithSelect` vector path: don't `ensureVGPR` cond before `V_CMP_NE_U32` when other operand is immediate.

### `emitScalarCmp` Helper (committed)

Shared `inline` function in `Handlers.h` that emits `S_CMP_*` for any `arith::CmpIPredicate`. Used by `handleArithCmpI` and available for `AMDGPUHandlers.cpp`.

## What was attempted but failed

### 1. CmpI Fusion in handleArithCmpI (all-scalar cmpi → SGPR boolean)

**Approach:** When both cmpi operands are scalar (not just ConditionOp use), emit `s_cmp + s_cselect(1, 0)` to produce an SGPR boolean instead of V_CMP + VGPR boolean.

**Result:** Memory fault. The SGPR boolean propagates through the type system (via arith.extui, arith.addi, arith.select) and changes register allocation for buffer descriptors downstream. The cascading SGPR changes corrupt an SRD.

**Root cause:** Unknown. The assembly logic is correct, SCC chains are valid. The issue is in how the changed register allocation (from keeping more values in SGPRs) shifts physical register assignments for precolored SRD registers.

### 2. CmpI Fusion in handleArithSelect (scalar cmpi → s_cmp + s_cselect)

**Approach:** In `handleArithSelect`, when the condition comes from `arith.cmpi` with scalar operands and select values are scalar, re-emit `s_cmp + s_cselect` directly.

**Result:** Memory fault (same root cause as #1). Also tried with `allUsersScalarSafe` check — when restricted enough to not change register allocation, no patterns fire.

### 3. SRD Scalar Select in emitSrdNumRecords

**Approach:** In `emitSrdNumRecords`, detect `arith.select(arith.cmpi(scalar, scalar), scalar, scalar)` feeding `fat_raw_buffer_cast`'s `validBytes`, and emit `s_cmp + s_cselect` directly into the SRD num_records SGPR (PSRegType).

**Result:** Memory fault. Tried multiple emission approaches:
- Direct PSRegType result with DCEProtectOp
- Virtual SGPR + s_mov_b32 copy to PSRegType
- Virtual SGPR + s_add_u32 copy (discovered s_add_u32 clobbers SCC — eliminated as cause)
- Fresh SGPR copy inside loop body (to avoid cross-loop liveness issues)

All produce correct-looking assembly: `s_cmp_lt_i32 s37, s10` → `s_cselect_b32 s66, s26, 0` (SRD word 2 = num_records). SCC chain is valid (no clobber between s_cmp and s_cselect). SCC verifier reports no hazards. The register allocator keeps the validBytes SGPR alive. Both individual SRDs fail when targeted separately.

**Fault pattern:** "Write access to a read-only page" on various GPU addresses. Consistent with an SRD corruption (wrong num_records allowing access beyond buffer allocation).

## The VGPR Pressure Gap (analysis)

### Loop Body VGPR Breakdown: Our Code vs aiter Reference

| Category | Our Code | Reference | Delta |
|----------|----------|-----------|-------|
| MFMA data operands | 180 | 160 | +20 |
| MFMA scale operands | 24 | 10 | +14 |
| buffer_load destinations (B prefetch) | 58 | 34 | +24 |
| buffer_load voffset addresses | 24 | 10 | +14 |
| buffer_load_lds voffsets | 18 | 10 | +8 |
| **VALU address/control in loop** | **21** | **0** | **+21** |
| Total | ~276 | ~227 | ~49 |

### How the reference achieves zero loop-body VALU

The aiter reference kernel does ALL in-loop address computation with SALU:

```asm
; aiter: 2 SALU ops, 0 VGPRs
s_cmp_lt_u32 0x200, s51          ; scalar comparison
s_cselect_b32 s61, s61, 0        ; scalar select into SRD soffset
```

Our code does the same thing with 7 VALU ops and 3+ wasted VGPRs:

```asm
; our code: 7 VALU ops, 3+ VGPRs
v_mov_b32 v151, s27             ; SGPR→VGPR (wasted)
v_cmp_lt_i32 vcc, v151, s37     ; uniform comparison in VALU
v_mov_b32 v151, 1               ; constant→VGPR (wasted)
v_cndmask_b32 v152, 0, v151     ; bool→VGPR (wasted)
v_cmp_ne_u32 vcc, v152, 0       ; re-restore VCC (wasted)
v_cndmask_b32 v151, 0, v29      ; select validBytes or 0
v_readfirstlane_b32 s66, v151   ; back to SGPR for SRD word 2
```

The root cause: `handleArithCmpI` materializes VCC→VGPR boolean, then `handleArithSelect` re-materializes VGPR→VCC for v_cndmask.

### The MLIR pattern

```mlir
%1377 = affine.apply (s0 + 2)[%arg8]                    ; next K step (uniform)
%1378 = arith.cmpi slt, %1377, %238 : index             ; uniform comparison
%1379 = arith.select %1378, %323, %c0_i64 : i64         ; validBytes or 0
%1380 = amdgpu.fat_raw_buffer_cast ... validBytes(%1379) ; SRD setup
```

Both `%1377` and `%238` are provably uniform (derived from kernel args + loop IV). The select values (`%323` = `(M*K)/2`, `%c0_i64` = 0) are also scalar constants.

## Next Steps

### To debug the s_cselect memory fault

1. Add `--mlir-print-ir-after=waveasm-linear-scan` support to `compile.py` (pipe stderr to file)
2. Generate IR dumps for both passing (v_readfirstlane) and failing (s_cselect) versions
3. Diff the IR after register allocation to find which physical register assignment changed
4. The diff will reveal the SRD corruption path — likely a precolored SGPR that got displaced

### Alternative approaches to reduce VGPR pressure

1. **Schedule-level fix:** In the Python schedule (`gemm_mxfp4_double_buffer.py`), avoid `arith.select` for the branchless guard and instead directly produce scalar validBytes values that the WaveASM backend can keep in SGPRs.

2. **Post-translation SGPR promotion pass:** A WaveASM-level pass that identifies uniform v_cmp + v_cndmask + v_readfirstlane chains and converts them to s_cmp + s_cselect. This would run after translation but before register allocation, avoiding the handler-level issues.

3. **Buffer soffset approach (matching aiter):** Instead of modifying num_records, use the buffer soffset field for the branchless guard. The aiter kernel uses `s_cselect_b32 s61, s61, 0` on the SOFFSET, not num_records. This may bypass the PSRegType interaction that causes the fault.

## Key Files

| File | Role |
|------|------|
| `waveasm/include/waveasm/Dialect/WaveASMInterfaces.h` | SCCDef/SCCUse C++ trait classes |
| `waveasm/include/waveasm/Dialect/WaveASMOps.td` | Op class definitions with SCC traits |
| `waveasm/lib/Transforms/SCCVerifier.cpp` | SCC verification pass |
| `waveasm/lib/Transforms/handlers/Handlers.h` | emitScalarCmp, emitOr/Xor/Min/Max helpers |
| `waveasm/lib/Transforms/handlers/ArithHandlers.cpp` | handleArithCmpI, handleArithSelect |
| `waveasm/lib/Transforms/handlers/AMDGPUHandlers.cpp` | emitSrdNumRecords, handleFatRawBufferCast |
| `waveasm/lib/Transforms/handlers/AffineHandlers.cpp` | emitCeilFromFloorQuotient, handleAffineApply |
| `wave_ref/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s` | aiter reference kernel |
