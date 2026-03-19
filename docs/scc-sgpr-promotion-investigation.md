# SCC & SGPR Promotion Investigation

## Summary

This document captures the findings from investigating SCC (Scalar Condition Code) tracking and VGPR→SGPR promotion in the WaveASM backend. The work spans SCC infrastructure, handler-level SALU promotions, and the unresolved SRD scalar select issue.

## What Was Built

### SCC Infrastructure (committed, working)

- **SCCDef / SCCUse traits** on all SALU ops — every op now declares whether it writes or reads SCC
- **SCC verifier pass** (`--waveasm-scc-verifier`) — walks IR in emission order, catches SCC clobbers between producer and consumer
- Key insight: `Pure` in MLIR ODS = `NoMemoryEffect + AlwaysSpeculatableImplTrait`. Composing these with `SCCDef` gives identical MLIR pass behavior while enabling SCC verification. Adding bare `NativeOpTrait` to existing ODS classes changes tablegen output and alters MLIR pass behavior — must use the explicit composition.
- `S_CSELECT_B32` carries `SCCUse` trait (not Pure, not CSE-eligible — result depends on implicit SCC)

### Handler SALU Promotions (committed, working)

| Handler | Change | Impact |
|---------|--------|--------|
| `handleArithOrI` | `emitOr` helper (S_OR_B32 when both scalar) | No trigger in GEMM kernel |
| `handleArithXorI` | `emitXor` helper (S_XOR_B32) | No trigger |
| `handleArithDivUI` | Uses `emitLshr` (has scalar path) | No trigger |
| `handleArithRemUI` | Uses `emitAnd` (has scalar path) | No trigger |
| `handleArithMaxSI/UI` | `emitMaxI32/U32` (S_MAX_I32 when scalar) | **1 v_max → s_max** |
| `handleArithMinSI/UI` | `emitMinI32/U32` (S_MIN_I32 when scalar) | No trigger |
| `handleArithCmpI` | SGPR accepted in V_CMP (constant bus) | **2 v_mov_b32 eliminated** |
| `handleArithSelect` | SGPR cond accepted in V_CMP_NE_U32 | No trigger |
| AffineHandlers OR | `emitOr` instead of `ensureBothVGPR + V_OR_B32` | No trigger |

### `emitScalarCmp` Helper (committed)

Shared inline function in `Handlers.h` that emits `S_CMP_*` for any `arith::CmpIPredicate`. Used by `handleArithCmpI` and available for `AMDGPUHandlers.cpp`.

## The VGPR Pressure Problem

### Numbers (256x192 GEMM, our kernel vs aiter reference)

| Category | Ours | Reference | Delta |
|----------|------|-----------|-------|
| VALU address/control in loop | 21 VGPRs | 0 | **+21** |
| MFMA scale operands | 24 | 10 | +14 |
| buffer_load voffset addresses | 24 | 10 | +14 |
| Peak arch VGPRs | ~276 | ~227 | **+49** |

The biggest single win is eliminating VALU from the loop body (+21 VGPRs).

### Root Cause: cmpi→select→fat_raw_buffer_cast Chain

In the MLIR, the loop body has:

```mlir
%next_K = affine.apply (s0 + 2)[%loop_iv]       // scalar
%K_bound = affine.apply (s0 ceildiv 256)[%K]     // scalar  
%cond = arith.cmpi slt, %next_K, %K_bound        // uniform comparison
%validBytes = arith.select %cond, %bufSize, 0     // branchless guard
%srd = amdgpu.fat_raw_buffer_cast ... validBytes(%validBytes)
```

All values are provably uniform (scalar). The aiter reference emits this as 2 SALU ops:

```asm
s_cmp_lt_u32 0x200, s51          ; scalar comparison
s_cselect_b32 s61, s61, 0        ; scalar select → SRD soffset
```

Our code emits 7 VALU ops + v_readfirstlane because:
1. `handleArithCmpI` emits `V_CMP + materializeVCCToBoolVGPR` (VGPR boolean)
2. `handleArithSelect` emits `V_CMP_NE_U32 + V_CNDMASK_B32` (VGPR result)
3. `emitSrdNumRecords` does `v_readfirstlane_b32` to extract back to SGPR

## What Was Tried (and Failed)

### Approach 1: CmpI Fusion in handleArithCmpI

**Idea:** When both cmpi operands are scalar, emit `s_cmp + s_cselect(1, 0)` to produce an SGPR boolean instead of VGPR.

**Result:** Memory fault. The SGPR boolean propagates through `arith.extui` and `arith.addi` chains, changing the type of downstream values from VGPR to SGPR. This cascading type change alters register allocation for buffer descriptors, corrupting SRD addresses.

**Why:** The SGPR result enters the mapper and changes every downstream consumer's type decision. Values that were VGPR (with v_readfirstlane extraction) become SGPR, shifting the entire allocation picture.

### Approach 2: CmpI Fusion in handleArithSelect

**Idea:** Add a fusion path that detects `arith.select(arith.cmpi(scalar, scalar), scalar, scalar)` and emits `s_cmp + s_cselect` directly.

**Result:** Memory fault when the select result feeds non-SRD consumers (cascading SGPR changes). With user-safety guards (only fire for specific downstream patterns), the fusion never triggers for the loop-body pattern.

### Approach 3: Targeted SRD Scalar Select in emitSrdNumRecords

**Idea:** Only in `emitSrdNumRecords`, detect the `arith.select(arith.cmpi)` pattern feeding `fat_raw_buffer_cast`'s `validBytes` and emit `s_cmp + s_cselect` directly into the precolored SRD word 2 (`PSRegType`).

**Result:** Memory fault ("Write access to a read-only page"). Tried four emission variants:
- Direct `S_CSELECT_B32` into `PSRegType(srdBase+2)` with `DCEProtectOp`
- `S_CSELECT_B32` into virtual SGPR → `S_MOV_B32` copy to PSReg
- `S_CSELECT_B32` into virtual SGPR → `S_ADD_U32` copy to PSReg (clobbers SCC — eliminated as cause)
- Fresh `S_MOV_B32` copy of trueOp inside loop body before `S_CSELECT_B32`

All produce correct-looking assembly. The SCC verifier reports no hazards. Register allocation appears correct (validBytes SGPR kept alive from prologue through loop body, not clobbered by intermediates).

### Why All Approaches Fail

The assembly emitted by all three approaches looks correct when inspected manually:
- SCC chain: `s_cmp → s_cselect` with nothing between them
- SRD formation: word 0 (base lo), word 1 (base hi + flags), word 2 (num_records from s_cselect), word 3 (stride/swizzle)
- Comparison semantics: `s_cmp_lt_i32` matches the original `v_cmp_lt_i32` predicate
- Value selection: `s_cselect src0=validBytes, src1=0` matches the original `v_cndmask 0, validBytes, vcc`

The fault address patterns suggest a corrupted buffer descriptor, but the assembly-level SRD setup appears identical to the working version (with different SGPR numbers from changed register allocation).

## Debugging Next Steps

The bug is invisible at the assembly level. The investigation needs:

1. **WaveASM IR dumps after register allocation** — compare the IR between passing (v_readfirstlane) and failing (s_cselect) versions to find which SSA value got a different physical register assignment
2. **The waveasm-translate tool supports `--mlir-print-ir-after-all`** (or the tool-specific `--print-ir` flag) — pipe stderr to a file
3. **Diff the IR** — look for PSRegType assignments that differ, especially for SRD word 0/1 (base address) registers

The key hypothesis: adding new SALU ops (s_cmp, s_cselect) in the loop body changes the SGPR pressure, which shifts the linear scan allocator's register assignments. One of those shifted assignments happens to clobber a precolored SRD register that's being set up elsewhere.

## V_CNDMASK_B32 Constant Bus Issue

Attempted to allow one SGPR in `V_CNDMASK_B32` (VOP3 constant bus allows one SGPR + VCC). This caused test failures — `v_cndmask_b32` VOP3 encoding with VCC counts as one constant bus slot, and adding an SGPR as src0/src1 requires a SECOND slot, exceeding the limit on some instructions.

## Files Modified

| File | Changes |
|------|---------|
| `WaveASMInterfaces.h/.td` | SCCDef, SCCUse trait definitions |
| `WaveASMOps.td` | SCC-aware op classes, trait assignments |
| `SCCVerifier.cpp` | SCC verification pass |
| `ScopedCSE.cpp` | SCCUse exclusion from CSE |
| `Handlers.h` | emitOr, emitXor, emitMinI32/U32, emitMaxI32/U32, emitScalarCmp |
| `ArithHandlers.cpp` | SALU paths for OR/XOR/Min/Max/DivUI/RemUI, V_CMP constant bus |
| `AffineHandlers.cpp` | emitOr for non-overlapping Add |
| `AMDGPUHandlers.cpp` | SRD scalar select TODO documentation |
| `Passes.td` | SCC verifier pass registration |
| `CMakeLists.txt` | SCCVerifier.cpp |
| `compile.py` | `--waveasm-scc-verifier` in pipeline |
| `waveasm_e2e.py` | Same |

## Commits

1. `d54817dc` — SCC verifier pass and trait infrastructure
2. `bc9726f8` — Migrate all SALU ops to SCC-aware trait classes
3. `d3e69752` — SALU paths for OR, XOR, DivUI, RemUI handlers
4. `3bcf0b85` — Eliminate unnecessary SGPR-to-VGPR coercions before V_CMP
5. `c22ce840` — emitScalarCmp helper and emitMin/emitMax SALU promotion
6. `8c65e6e0` — Move emitScalarCmp to Handlers.h
7. `963444f3` — SRD scalar select TODO with failure analysis
