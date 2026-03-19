# SCC & SGPR Promotion Investigation

## Summary

This document captures the findings from investigating SCC (Scalar Condition Code) tracking and VGPR→SGPR promotion in the WaveASM backend. The work spans SCC infrastructure, handler-level promotions, and a deep dive into the `s_cmp + s_cselect` memory fault.

## What Was Built

### SCC Infrastructure (committed, working)

1. **SCCDef / SCCUse traits** — every SALU op now correctly declares whether it writes or reads SCC
2. **SALUBinaryWithSCCOp / SALUUnaryWithSCCOp** — op classes with `NoMemoryEffect + AlwaysSpeculatableImplTrait + SCCDef` (equivalent to `Pure` for MLIR passes, but identifiable as SCC-clobbering)
3. **SCC verifier pass** (`--waveasm-scc-verifier`) — walks IR in emission order, detects SCC clobbers between producers and consumers
4. **`emitScalarCmp`** — shared helper in `Handlers.h` that emits `S_CMP_*` for any `arith::CmpIPredicate`

### Handler SALU Promotions (committed, working)

| Handler | Change | Impact |
|---------|--------|--------|
| `handleArithOrI` | Added `emitOr` (S_OR_B32 when scalar) | No change for GEMM (operands are VGPRs) |
| `handleArithXorI` | Added `emitXor` (S_XOR_B32 when scalar) | Same |
| `handleArithDivUI` | Uses `emitLshr` (has scalar path) | Same |
| `handleArithRemUI` | Uses `emitAnd` (has scalar path) | Same |
| `handleArithMaxSI/UI/MinSI/UI` | Added `emitMaxI32/MinI32/MaxU32/MinU32` | `v_max_i32 → s_max_i32` in prologue |
| `handleArithCmpI` | V_CMP accepts SGPR via constant bus | Eliminates 2 `v_mov_b32` in prologue |
| `handleArithSelect` | V_CMP_NE cond accepts SGPR | No change for GEMM |
| AffineHandlers OR | Uses `emitOr` instead of `ensureBothVGPR` | No change for GEMM |

### Key Finding: Adding NativeOpTrait Changes Codegen

Adding `NativeOpTrait<"SCCDef">` to existing op classes (like `SALUBinaryWithCarryOp`) changes the ODS-generated C++ class hierarchy. This causes MLIR's `--canonicalize` and `--loop-invariant-code-motion` passes to produce different IR, even though the trait is a pure marker with no semantic meaning.

**Solution:** Use `NoMemoryEffect + AlwaysSpeculatableImplTrait + SCCDef` explicitly (equivalent to `Pure + SCCDef`) instead of just adding `SCCDef` to classes that previously had `Pure`.

## The VGPR Pressure Problem

### Gap: Our kernel vs aiter reference

| Category | Our Code | Reference | Delta |
|----------|----------|-----------|-------|
| VALU address/control in loop | 21 VGPRs | 0 | +21 |
| Scale VGPRs | 24 | 10 | +14 |
| MFMA data layout | 180 | 160 | +20 |
| **Total** | **~276** | **~227** | **~49** |

### Root Cause: `v_cmp + v_cndmask + v_readfirstlane` for SRD num_records

The loop body has this pattern (7 VALU ops, 3+ wasted VGPRs):
```asm
v_mov_b32 v151, s27             ; next_K → VGPR (wasted)
v_cmp_lt_i32 vcc, v151, s37     ; uniform comparison in VALU
v_mov_b32 v151, 1               ; constant → VGPR (wasted)
v_cndmask_b32 v152, 0, v151     ; bool → VGPR (wasted)
v_cmp_ne_u32 vcc, v152, 0       ; re-restore VCC (wasted)
v_cndmask_b32 v151, 0, v29      ; select validBytes or 0
v_readfirstlane_b32 s66, v151   ; back to SGPR for SRD word 2
```

The aiter reference does this in 2 SALU ops:
```asm
s_cmp_lt_u32 0x200, s51          ; scalar comparison
s_cselect_b32 s61, s61, 0        ; scalar select into SRD num_records
```

### Why It Happens

1. `handleArithCmpI` emits `V_CMP + materializeVCCToBoolVGPR` (VGPR boolean) for ALL non-ConditionOp uses
2. `handleArithSelect` takes VGPR boolean, re-restores VCC via `V_CMP_NE_U32`, then `V_CNDMASK_B32`
3. `emitSrdNumRecords` receives a VGPR result, extracts via `v_readfirstlane_b32`

The comparison and both operands are provably uniform (kernel args + loop IV), but the type system forces VGPR at the cmpi handler.

## The s_cmp + s_cselect Memory Fault

### What Was Tried

Three approaches to emit `s_cmp + s_cselect` instead of the VGPR chain:

1. **CmpI fusion in `handleArithCmpI`** — produce SGPR boolean for all scalar cmpi
2. **CmpI fusion in `handleArithSelect`** — re-emit s_cmp + s_cselect when cmpi has scalar operands
3. **Targeted SRD fusion in `emitSrdNumRecords`** — detect `arith.select(arith.cmpi)` and emit s_cmp + s_cselect directly into PSRegType

All three produce correct-looking assembly and pass the SCC verifier, but cause GPU memory faults: "Write access to a read-only page."

### Emission Approaches Within (3)

| Approach | Assembly Output | Result |
|----------|-----------------|--------|
| Direct `S_CSELECT_B32` into PSRegType + DCEProtect | `s_cselect_b32 s66, s26, 0` | Memory fault |
| S_CSELECT into virtual SGPR + `S_MOV_B32` copy to PSReg | `s_cselect + s_mov` | Memory fault |
| S_CSELECT into virtual SGPR + `S_ADD_U32` copy to PSReg | `s_cselect + s_add (clobbers SCC)` | Memory fault |
| Fresh `S_MOV_B32` copy of trueOp inside loop + S_CSELECT | Same as above | Memory fault |

### What Was Verified

- SCC chain: `s_cmp → s_cselect` are adjacent, nothing clobbers SCC between them
- SCC verifier: passes with no errors
- Register liveness: the validBytes SGPR (`s26`) has only 2 definitions (prologue SRD setup + pre-loop copy), no overwrites between definition and loop-body use
- Comparison semantics: `s_cmp_lt_i32` matches `v_cmp_lt_i32` (both signed)
- Select semantics: `s_cselect_b32 dst, trueOp, 0` → SCC=1: trueOp, SCC=0: 0 (matches `v_cndmask`)
- Individual SRDs: both first (A-tile) and second (B-scale) fail independently

### What's Unknown

The assembly looks correct at every level:
- SRD words 0-3 are set correctly
- num_records value is the right validBytes SGPR
- The comparison and selection logic matches the baseline

The fault happens despite correct-looking assembly. Possible causes:
1. **Register allocation cascade** — the new s_cmp/s_cselect/s_mov ops change SGPR allocation, displacing another value that was part of a different SRD
2. **Precolored register conflict** — the PSRegType result interacts badly with the register allocator's precoloring constraints
3. **IR-level issue** — something in the WaveASM IR (not visible in assembly) is malformed

### Next Debugging Step

Compare the WaveASM IR after register allocation between the passing (v_readfirstlane) and failing (s_cselect) versions. The diff will show which physical register assignments changed and which SRD got corrupted.

To generate the IR dump, add `--mlir-print-ir-after=waveasm-linear-scan` (or the equivalent flag that `waveasm-translate` supports) to the compilation command and pipe stderr to a file.

## MLIR-Level Analysis

The MLIR source (before WaveASM translation) shows the critical loop-body pattern:

```mlir
%1377 = affine.apply [s0] -> (s0 + 2) [%arg8]        // next K step
%1378 = arith.cmpi slt, %1377, %238 : index           // next_K < K_bound
%1379 = arith.select %1378, %323, %c0_i64 : i64       // validBytes or 0
%1380 = amdgpu.fat_raw_buffer_cast %cast validBytes(%1379) ...
```

Both `%1377` and `%238` are uniform (kernel args + loop IV). The comparison and select are provably scalar. The MLIR itself is correct — the issue is purely in the WaveASM translation handlers' VGPR/SGPR decisions.

## Files Modified

| File | Changes |
|------|---------|
| `WaveASMInterfaces.h/.td` | SCCDef, SCCUse trait definitions |
| `WaveASMOps.td` | Op class reclassification with correct SCC traits |
| `Handlers.h` | `emitOr`, `emitXor`, `emitMin*`, `emitMax*`, `emitScalarCmp` |
| `ArithHandlers.cpp` | Handler promotions, V_CMP constant bus fix |
| `AffineHandlers.cpp` | `emitOr` for non-overlapping Add |
| `AMDGPUHandlers.cpp` | SRD scalar select TODO documentation |
| `SCCVerifier.cpp` | SCC hazard verification pass |
| `ScopedCSE.cpp` | SCCUse exclusion from CSE |
| `Passes.td` | SCC verifier pass registration |
| `compile.py` / `waveasm_e2e.py` | Pipeline integration |
