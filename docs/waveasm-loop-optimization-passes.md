# WaveASM Loop Optimization Passes

## Goal

Eliminate all unnecessary VALU instructions from the main GEMM loop body in the C++ WaveASM backend, matching AITER's reference kernel where the loop contains only MFMAs, memory loads, and minimal SALU control flow.

## Results

| Metric | Original | Current | Target |
|--------|----------|---------|--------|
| Loop VALU for B-scale packing | 14 (6 LSHL_OR + 8 BFE) | 6 LSHL_OR | 0 |
| Loop VALU for B-data buffer_load addresses | ~69 | 0 | 0 |
| Loop VALU for A-data buffer_load_lds addresses | 10 | 10 → 0 (pending) | 0 |
| Loop SALU for soffset bumping | 0 | 3 | 3 |
| Loop body size (asm lines) | 332 | 277 | ~260 |

### Remaining VALU ops in the loop body

**B-scale byte repacking (6 ops)**: B-scale values are loaded as individual bytes via `buffer_load_ubyte` (8 per half-iteration) because the preshuffle memory layout does not group 4 contiguous scale bytes per dword. Each iteration repacks 8 bytes into 2 dwords via `v_lshl_or_b32` chains for use as MFMA scale operands. Fix requires changing the preshuffle B-scale layout to produce contiguous 4-byte groups, enabling `buffer_load_dword` instead of 8× `buffer_load_ubyte`.

**A-data LDS prefetch voffset (10 ops, fix pending)**: The `buffer_load_dwordx4 ... lds` ops for A-data prefetch recompute their voffsets each iteration from the IV. The existing `BufferLoadLDSSoffsetPattern` in the Peephole pass should extract the scalar shift into `soffset` and let LICM hoist the remaining voffset, but `LshlAddPattern` fires first on the inner `V_ADD_U32`, fusing it into `V_LSHL_ADD_U32` which hides the scalar shift. A Case 3 has been added to `BufferLoadLDSSoffsetPattern` to look through `V_LSHL_ADD_U32` and extract the buried scalar shift.

## Passes

### 1. ScalePackElimination

**File**: `wave_asm/lib/Transforms/ScalePackElimination.cpp` (new)

**Problem**: B-scale dwords (loaded via `buffer_load_dword`) were decomposed into 4 individual bytes via `v_bfe_u32`, carried as 8 separate `scf.for` iter_args, then repacked via a 3-instruction `v_lshl_or_b32` chain at the top of each iteration. This is an identity round-trip: the repacked dword equals the original loaded dword.

```
Loop body (before):
  ; 6 LSHL_OR ops to repack bytes into dword
  v_lshl_or_b32 v3, v64, 8, v63      ; pack byte1<<8 | byte0
  v_lshl_or_b32 v10, v65, 16, v3     ; pack byte2<<16 | prev
  v_lshl_or_b32 v3, v66, 24, v10     ; pack byte3<<24 | prev = full dword
  ...
  ; 8 BFE ops to extract bytes from next loaded dword
  v_bfe_u32 v63, v172, 0, 8
  v_bfe_u32 v64, v172, 8, 8
  ...
```

**Fix**: The pass detects `v_lshl_or_b32` chains (shifts 8, 16, 24) whose 4 byte inputs are loop block arguments. It verifies that the init args and yield args are `v_bfe_u32` extractions (offsets 0, 8, 16, 24, width 8) from the same source dword. Then it rebuilds the loop: replaces the 4 byte iter_args with 1 dword iter_arg, eliminating the LSHL_OR and BFE ops.

**Approach**:
1. Scan each `LoopOp` for `v_lshl_or_b32` pack chains whose byte inputs are block arguments
2. Verify init args are BFE extractions from the same dword
3. Verify yield (ConditionOp) args are BFE extractions from the same new dword
4. Rebuild the loop with the dword as iter_arg instead of 4 bytes, removing the 3 LSHL_OR ops and 4 BFE ops from the body

---

### 2. BFEPackIdentityPattern

**File**: `wave_asm/lib/Transforms/Peephole.cpp` (modified — added pattern)

**Problem**: After ScalePackElimination replaces byte loop results with dword results, epilogue code still had BFE-then-LSHL_OR identity chains (extract 4 bytes from a dword then repack into the same dword).

**Fix**: Greedy rewrite pattern added to the existing Peephole pass. Detects a 3-instruction `v_lshl_or_b32` chain (shifts 8, 16, 24) where all 4 byte inputs come from `v_bfe_u32` extracting bytes 0, 8, 16, 24 from the same source. Since the result equals the source, replaces the chain output with the original source value.

**Pipeline note**: A second peephole pass run was added after ScalePackElimination to catch these epilogue cases.

---

### 3. BufferLoadStrengthReduction

**File**: `wave_asm/lib/Transforms/BufferLoadStrengthReduction.cpp` (new)

**Problem**: ~69 VALU instructions per iteration computing `buffer_load_dwordx4` and `buffer_load_dword` voffsets from scratch. These mix the loop induction variable (`s10`) with thread-local values (thread_id, workgroup_id) through complex affine expressions involving shifts, floor divisions, and bit masking.

AITER has zero VALU for this — it precomputes voffsets before the loop and uses SALU `soffset` bumping.

**Fix (soffset approach)**:
1. Precompute all voffsets at `iv=initial_value` before the loop (they become loop-invariant constants)
2. Group buffer_loads by SRD (buffer descriptor). Compute one stride per group: `stride = voffset(iv+step) - voffset(iv)`, converted from VGPR to SGPR via `v_readfirstlane_b32`
3. Carry one SGPR soffset per SRD group as an iter_arg (initialized to 0)
4. Each iteration: `soffset += stride` via `s_add_u32` (2 SALU ops total for B-data + B-scale)
5. Set each buffer_load's soffset operand to the group's soffset iter_arg

```
Loop body (after):
  ; buffer_loads use precomputed voffset + SGPR soffset
  buffer_load_dwordx4 v[124:127], v106, s[24:27], s15 offen
  buffer_load_dwordx4 v[132:135], v20,  s[24:27], s15 offen
  ...
  buffer_load_dword   v26,        v108, s[28:31], s16 offen
  ...
  ; Only 2 SALU ops for soffset bumping
  s_add_u32 s15, s15, s14    ; B-data soffset += stride
  s_add_u32 s16, s16, s11    ; B-scale soffset += stride
```

**Key details**:
- The `soffset` update (`s_add_u32`) must be placed before the `s_cmp`/`s_cbranch` condition check because `s_add_u32` clobbers SCC. The pass inserts it before the condition producer.
- The stride is uniform across all threads (the K-dimension stride is the same per lane), so `v_readfirstlane_b32` safely converts it to an SGPR.
- The voffset dependency chain is cloned before the loop using `IRMapping` that replaces all block arguments with their corresponding init args.

**Required infrastructure change**: Added `soffset` operand to `VMEMLoadOp` in the dialect definition. Previously the assembly emitter hardcoded `soffset=0`. Now it emits the actual soffset register.

---

### 4. BufferLoadLDSSoffsetPattern (Case 3)

**File**: `wave_asm/lib/Transforms/Peephole.cpp` (modified — added case to existing pattern)

**Problem**: 10 VALU ops per iteration computing A-data `buffer_load_dwordx4 ... lds` voffsets. The voffset chain is:

```
%shift_iv = v_lshlrev_b32 7, s10           ; scalar shift of IV
%fused    = v_lshl_add_u32 v4, 13, %shift_iv  ; LshlAddPattern fused this
%voff     = v_add_u32 base, %fused
buffer_load_dwordx4_lds %voff, srd, 0
```

The existing `BufferLoadLDSSoffsetPattern` (Cases 1 & 2) can extract `v_lshlrev_b32(const, sgpr)` from a `V_ADD_U32` chain into `soffset`. But `LshlAddPattern` fires first on the inner add, fusing the VGPR shift (`v4 << 13`) with the scalar shift into `V_LSHL_ADD_U32`. The pattern then can't see through the fused op to find the scalar shift.

The `LshlAddPattern` guard (lines 135-153) only protects when the **shifted base is SGPR**. Here the fused shift has a VGPR base (`v4`), so the guard doesn't trigger — the VGPR shift gets fused, burying the scalar shift as `src2`.

**Fix (Case 3)**: When an operand of the outer `V_ADD_U32` is `V_LSHL_ADD_U32(vgpr, const, V_LSHLREV_B32(const, sgpr))`, decompose it:
1. Extract `s_lshl_b32(sgpr, const)` → soffset
2. Recreate `v_lshlrev_b32(const, vgpr)` → the VGPR shift from the fused op
3. New voffset = `v_add_u32(row, v_lshlrev_b32(...))`  — loop-invariant, hoisted by LICM

After this pattern + LICM, the A-data voffsets are precomputed before the loop and the soffset is `s_lshl_b32(s10, 7)` (1 SALU per iteration). AITER achieves the same with SRD base bumping (3 SALU), so our approach is slightly cheaper.

---

## Files Modified

### New files

| File | Description |
|------|-------------|
| `wave_asm/lib/Transforms/ScalePackElimination.cpp` | Scale pack elimination pass |
| `wave_asm/lib/Transforms/BufferLoadStrengthReduction.cpp` | Buffer load strength reduction pass |

### Modified files

| File | Change |
|------|--------|
| `wave_asm/include/waveasm/Transforms/Passes.h` | Added `createWAVEASMScalePackEliminationPass()` and `createWAVEASMBufferLoadStrengthReductionPass()` declarations |
| `wave_asm/include/waveasm/Dialect/WaveASMOps.td` | Added `soffset` operand to `VMEMLoadOp` class (was 2 operands: saddr, voffset; now 3: saddr, voffset, soffset) |
| `wave_asm/lib/Transforms/Peephole.cpp` | Added `BFEPackIdentityPattern`; added Case 3 to `BufferLoadLDSSoffsetPattern` for `V_LSHL_ADD_U32` decomposition |
| `wave_asm/lib/Transforms/AssemblyEmitter.cpp` | Updated `emitBufferLoad` to emit soffset register instead of hardcoded `0` |
| `wave_asm/lib/Transforms/TranslateFromMLIR.cpp` | Updated all `BUFFER_LOAD_*::create` calls to pass `soffset=0` |
| `wave_asm/lib/Transforms/handlers/AMDGPUHandlers.cpp` | Updated `BUFFER_LOAD_*::create` calls to pass `soffset=0` |
| `wave_asm/lib/Transforms/handlers/MemRefHandlers.cpp` | Updated `BUFFER_LOAD_DWORD::create` call to pass `soffset=0` |
| `wave_asm/lib/Transforms/CMakeLists.txt` | Added `ScalePackElimination.cpp` and `BufferLoadStrengthReduction.cpp` |
| `wave_asm/tools/waveasm-translate/waveasm-translate.cpp` | Added passes to pipeline: ScalePackElimination, second Peephole run, BufferLoadStrengthReduction |

### Pipeline order (in `waveasm-translate.cpp`)

```
ScopedCSE
Peephole (fuse lshl+or, LICM, etc.)
ScalePackElimination (remove BFE/LSHL_OR round-trips for B-scale iter_args)
Peephole again (fold epilogue BFE→LSHL_OR identity chains)
BufferLoadStrengthReduction (precompute voffsets, use soffset bumping)
LoopAddressPromotion (LDS address rotation)
MemoryOffsetOpt + Canonicalizer + CSE
LinearScan (register allocation)
InsertWaitcnt
HazardMitigation
```

## AITER Reference

AITER's loop body has zero VALU for buffer_load addressing — both B-data/B-scale and A-data/A-scale.

**B-data & B-scale**: Precomputes voffsets before the loop, bumps the SRD base address with SALU ops each iteration:

```asm
; AITER: SRD pointer bumping (3 SALU per SRD group)
s_add_u32  s16, s62, s16     ; SRD base_lo += stride
s_addc_u32 s17, 0, s17       ; carry to base_hi
s_sub_u32  s18, s18, s62     ; shrink SRD size
```

Our approach uses `soffset` bumping instead of SRD modification (equivalent effect, simpler to implement in SSA IR since it doesn't require modifying the 4-SGPR buffer descriptor tuple):

```asm
; Wave: soffset bumping (1 SALU per SRD group)
s_add_u32 s15, s15, s14      ; soffset += stride
```

**A-data (buffer_load_lds)**: AITER precomputes voffsets v212-v219 once before the loop. They encode only the per-thread offset within a K-tile and never change. K-iteration advancement is via SRD base bumping (`s_add_u32 s12, s61, s12` + carry). LDS write addresses use fixed SALU: `s_add_u32 m0, <literal>, s59`.

Our approach extracts the IV-dependent scalar shift from the voffset into the `soffset` field via `BufferLoadLDSSoffsetPattern`, then LICM hoists the remaining loop-invariant voffset before the loop:

```asm
; Wave: soffset from IV (1 SALU per iteration, computed inside loop)
s_lshl_b32 sN, s10, 7        ; soffset = iv << 7
; voffsets precomputed before loop (hoisted by LICM)
buffer_load_dwordx4_lds v_precomputed, srd, sN
```

Both achieve zero VALU in the loop for address computation.
