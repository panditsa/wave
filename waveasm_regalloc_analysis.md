# WaveASM Register Allocation & Codegen Analysis

**Kernel:** 4-wave asymmetric MXFP4 GEMM (256x192x256 tile, 8192x3072x184576 shape)
**Target:** GFX950 (MI300)
**Source IR:** `examples/python/run.log` (46,204 lines, 11 pass dumps)

## Test Command

```bash
# Inside the docker container
```bash
# Enter the docker container
docker exec -it 27d65447d0a7fa35086019f0ca9d9019a11b6758178b6f491c93a414ded9a573 zsh
cd /workspace/wave && source .venv/bin/activate

# Build waveasm on the CURRENT branch first
cd /workspace/wave/waveasm/build && ninja

cd /workspace/wave/examples/python
HIP_VISIBLE_DEVICES=2 python 7.1_schedule.py \
  --test test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm \
  --block 256,192,256 --shape 8192,3072,184576
```

---

## Current Register Budget

| Resource | Used | Max Available | Occupancy |
|----------|------|---------------|-----------|
| VGPRs | 254 | 256 | **1 wave/SIMD** (minimum) |
| SGPRs | 78 | 108 | OK |
| AccVGPRs | 192 | 256 | OK (48 MFMA tiles x 4 regs) |

The 254 VGPR usage is the critical bottleneck — it pins occupancy to 1 wave per SIMD, meaning zero latency hiding from wave switching. Every VGPR saved is progress toward 2-wave occupancy (which requires ≤128 VGPRs — a steep cliff, so realistically the goal is reducing instruction count and freeing headroom).

No VGPR spilling occurs (no scratch loads/stores), which is good.

---

## Pass Pipeline (as captured in run.log)

```
WAVEASMScopedCSE          → WAVEASMPeephole
  → LoopInvariantCodeMotion → WAVEASMBufferLoadStrengthReduction
  → WAVEASMMemoryOffsetOpt  → CanonicalizerPass
  → WAVEASMScopedCSE (2nd)  → WAVEASMLinearScan (regalloc)
  → WAVEASMVGPRCompaction   → WAVEASMInsertWaitcnt
  → WAVEASMHazardMitigation (final)
```

---

## Identified Inefficiencies

### 1. SCC Modeling is Wrong — Blocks CSE, Wastes SGPRs, Has Verifier Gaps

The same `s_cmp_ne_u32` comparison is emitted repeatedly with identical operands:
- `s_cmp_ne_u32 %dst, 0` — appears 12+ times (often 2-3x consecutively)
- `s_cmp_ne_u32 %108, 0` — appears 18+ times

**Root cause:** `s_cmp` cannot be CSE'd because SCC is modeled incorrectly in the IR. The current model (`SALUCmpOp`) gives `s_cmp` a fake SGPR result to allow SSA dataflow, but:

1. **The assembly emitter never writes that SGPR.** `s_cmp` is emitted as `s_cmp_lg_u32 src0, src1` with no destination register. The SGPR allocated by regalloc is wasted — 1 SGPR per `s_cmp` (up to 30 wasted SGPRs in a kernel with many comparisons).

2. **The "result" is the hardware SCC flag**, a 1-bit implicit register. If CSE merges two `s_cmp` ops, the second consumer reads stale SCC because any intervening `SCCDef` op (s_and, s_or, s_add, etc.) clobbers SCC between the first definition and the second use.

3. **The SCC verifier has gaps.** It catches clobbers for `ConditionOp` and `IfOp` consumers but does NOT verify the path between an `s_cmp` and an `s_cselect_b32` consumer — it only checks that *some* preceding SCC writer exists, not that it's the correct one.

**Correct fix (medium effort, high impact):** Introduce a dedicated `SCCType` (not `SRegType`) for `s_cmp` results. Then:
- The register allocator would NOT assign a physical SGPR (saving 1 SGPR per `s_cmp`).
- CSE could be made safe: when the CSE pass merges two `s_cmp` ops and extends the SCC live range, the register allocator (or a post-CSE fixup pass) would **spill SCC to a real SGPR** by inserting `s_cselect_b32 sN, 1, 0` right after the first `s_cmp`, then later reloading with `s_cmp_ne_u32 sN, 0` before each consumer — but only when SCC-clobbering ops intervene.
- The SCC verifier could be strengthened to check ALL SCC consumers (including `s_cselect_b32` and `s_addc_u32`).
- The `ConditionOp` emitter would branch on SCC directly (as it already does), while the `s_cselect_b32` emitter would rely on the properly-tracked SCC.

**Interim workaround (low effort):** Without `SCCType`, the comparison duplication can be reduced by sinking identical `s_cmp` ops closer to their consumers in a dedicated pass, or by having the emitter re-emit `s_cmp` when it detects the SCC was clobbered.

**Where to fix:** `waveasm/include/waveasm/Dialect/WaveASMOps.td` (SALUCmpOp result type), `waveasm/include/waveasm/Dialect/WaveASMTypes.td` (new SCCType), `waveasm/lib/Transforms/LinearScanRegAlloc.cpp` (skip SCCType allocation), `waveasm/lib/Transforms/AssemblyEmitter.cpp` (SCC spill/reload), `waveasm/lib/Transforms/SCCVerifier.cpp` (check all consumers).

### 2. Integer Division Done in VGPRs for Uniform Values (~6 divisions, ~120 VALU wasted)

The Barrett reduction integer division sequence uses ~20 VALU instructions:
```
v_cvt_f32_u32 → v_rcp_f32 → v_mul_f32(magic) → v_cvt_u32_f32
→ Newton-Raphson refinement → v_mul_hi_u32 (quotient) → 2x correction steps
```

This runs in VGPRs even when both dividend and divisor are wave-uniform SGPR values. The result is then extracted back via `v_readfirstlane_b32`. The SGPR→VGPR→SGPR round-trip is pure waste for uniform values.

6 division sequences appear in the prologue. At least 2 share the same divisor (`%53`) but recompute the reciprocal from scratch.

**Where to fix:** `waveasm/lib/Transforms/handlers/AffineHandlers.cpp` (lines 257-319, `emitUnsignedFloordiv`). Add a scalar ALU path using `s_mul_hi_u32` for uniform operands, and cache reciprocals by divisor SSA value.

### 3. Multiply-by-Zero and Add-Zero Not Fully Folded (~60 VALU wasted)

Line 138: `v_mul_lo_u32 0, -256` computes literal zero in a VGPR. This zero then feeds ~59 `v_add_u32 0, %x` instructions throughout the IR — each a no-op addition.

The peephole pass has `MulZeroPattern` and `AddZeroPattern` but they don't fire here, likely because both operands are immediates (pattern expects vreg + imm), or the zero operand is in the wrong position.

**Where to fix:** `waveasm/lib/Transforms/Peephole.cpp` — extend `MulZeroPattern` to handle `imm * imm` constant folding, and `AddZeroPattern` to be operand-order agnostic.

### 4. Redundant VREG-to-VREG Copies (~30 VALU + VGPR pressure)

~30 `v_mov_b32 vreg → vreg` identity copies appear, mostly right before `buffer_load_dwordx4_lds` instructions. The source VREG (from `v_cndmask_b32`) is copied into a fresh VREG for the load address.

These exist because `buffer_load_dwordx4_lds` may clobber its address VGPR, requiring a protective copy. But if the address is dead after the load, the copy is unnecessary.

**Where to fix:** `waveasm/lib/Transforms/Peephole.cpp` (`RedundantMovePattern`) — add a last-use check.

### 5. Epilogue is Catastrophically Scalar (biggest perf opportunity)

The epilogue occupies ~2,270 IR ops (55% of the final pass). It issues **192 individual `buffer_store_short`** (2-byte) stores, each with:
- Its own `s_and_saveexec_b64` / `s_mov_b64_exec` exec mask pair (384 SALU ops total)
- Individual `v_accvgpr_read_b32` + `v_cvt_bf16_f32` per element
- A single bf16 written per store instruction

This is 4x-32x underutilization of store bandwidth. The kernel should:
- Pack bf16 pairs using `v_cvt_pk_bf16_f32` (already supported in the codebase)
- Use `buffer_store_dword` (4 bytes = 2 bf16) or `buffer_store_dwordx4` (16 bytes = 8 bf16)
- Share exec masks across groups of stores with the same bounds check

**Where to fix:** `waveasm/lib/Transforms/TranslateFromMLIR.cpp` (lines 1276-1349 `convertF32ToBF16ForStore`, lines 1660-1705 store dispatch, lines 1710-1739 masked store).

### 6. Dead SRD Writes (~4-8 SALU wasted)

SRD fields `s22` and `s23` are initialized in the prologue (`0x7FFFFFFE` and `0x20000`) then overwritten before use when swizzle patterns are applied (`0x27000`). The initial writes are dead stores.

**Where to fix:** `waveasm/lib/Transforms/TranslateFromMLIR.cpp` (lines 154-442, `emitSRDPrologue`) — defer SRD[2:3] initialization for SRDs that will be adjusted.

### 7. Dropped Scheduling Barriers

10 `unhandled: rocdl.sched.barrier` comments in the loop body indicate scheduling hints from the higher-level IR are silently ignored. These are meant to prevent reordering across MFMA/load boundaries.

**Where to fix:** Add a handler in `TranslateFromMLIR.cpp` to emit `s_sched_barrier` or a WaveASM fence op.

### 8. Wait Count Observations

The loop uses alternating `vmcnt(10)` / `vmcnt(20)` with `lgkmcnt(0)` at each barrier — this is reasonable for double-buffered software pipelining. The `lgkmcnt(0)` is conservative but required for correctness with the LDS double-buffer scheme. No consecutive waitcnts detected. Overall, waitcnt placement looks sound.

---

## What's Working Well

- **No VGPR spilling** — the register allocator avoids scratch despite 254 VGPRs
- **Software-pipelined loop** — proper double-buffering with MFMA/load interleaving
- **Direct global-to-LDS loads** — `buffer_load_dwordx4_lds` bypasses VGPRs
- **Accumulator management** — 48 MFMA tiles (192 AccVGPRs) with proper chaining
- **Wait count structure** — well-tuned for the load/compute overlap pattern

---

## Priority Order for Fixes

| Pri | Issue | Est. Savings | Effort | Status |
|-----|-------|-------------|--------|--------|
| **1** | **Fix SCC modeling (SCCType)** | **~20 SALU + ~30 SGPRs + enables CSE** | **Medium** | **NEW** |
| 2 | Vectorize epilogue stores | 300+ SALU, 150+ stores | Medium | |
| 3 | Scalar ALU for uniform divisions | ~90 VALU, ~30 VGPRs | Medium | |
| ~~4~~ | ~~Deduplicate division reciprocals~~ | ~~\~20 VALU, ~10 VGPRs~~ | ~~Low~~ | **DONE** (reciprocal caching) |
| ~~5~~ | ~~Fold mul-by-zero / add-zero~~ | ~~\~60 VALU~~ | ~~Low~~ | **DONE** (const folding + getConstantValue) |
| 6 | Eliminate VREG-to-VREG copies | ~30 VALU, ~30 VGPRs | N/A | Copies from regalloc/emitter, not fixable at peephole level |
| 7 | Dead SRD store elimination | ~4-8 SALU | Low | Attempted; pre-scan approach had correctness issues |
| ~~8~~ | ~~Handle sched.barrier hints~~ | ~~Better scheduling~~ | ~~Low~~ | **DONE** (emits comment; pseudo-instruction, not real HW op) |
