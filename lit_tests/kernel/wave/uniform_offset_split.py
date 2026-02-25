# RUN: python %s | FileCheck %s

"""
Lit test for the uniform (induction-variable) offset splitting in GatherToLDS.

When use_buffer_ops=True, the codegen separates the induction-variable
contribution from the per-lane thread offset so the backend can fold the
uniform part into the buffer_load soffset field (SGPR) instead of a VALU add.

Without the split, the loop body computes:
  offset = thread_offset + iv_offset   (single arith.addi, all in VGPR)

With the split:
  iv_off  = arith.muli(iv, stride)     (uniform across lanes)
  offset  = arith.addi(thread_off, iv_off)

The arith.addi(VGPR, SGPR) pattern lets the AMDGPU backend's
SIFoldOperands fold the SGPR into the buffer_load soffset field.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_uniform_offset_split():
    """
    GEMM with GatherToLDS (A matrix goes global -> LDS).

    With use_buffer_ops=True and the three-way index split, the
    induction-variable offset inside the scf.for loop body should appear
    as a separate arith.muli + arith.addi chain feeding into
    amdgpu.gather_to_lds, distinct from the thread-dependent offset.
    """
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        use_global_to_shared=True,
        use_buffer_ops=True,
        target="gfx950",
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL: test_uniform_offset_split
    # CHECK:       func.func @gemm

    # The per-lane thread offset (%thread_off) is computed OUTSIDE the
    # scf.for loop -- it is loop-invariant.
    # CHECK:       %[[THREAD_OFF:.*]] = arith.addi %{{.*}}, %{{.*}} overflow<nsw>
    # CHECK:       scf.for %[[IV:[a-z0-9]+]] =

    # Inside the loop, the induction-variable contribution is computed
    # via affine.apply (IV * stride) and combined with a workgroup base
    # via arith.addi.  The per-lane thread offset is then added as a
    # SEPARATE arith.addi -- this separation lets the backend place the
    # lane-uniform part in buffer_load soffset (SGPR).
    # CHECK:         %[[IV_OFF:.*]] = affine.apply {{.*}}()[%[[IV]]]
    # CHECK:         %[[WG_IV:.*]] = arith.addi %{{.*}}, %[[IV_OFF]] overflow<nsw>
    # CHECK:         %[[COMBINED:.*]] = arith.addi %[[THREAD_OFF]], %[[WG_IV]] overflow<nsw>
    # CHECK:         amdgpu.gather_to_lds %{{.*}}[%[[COMBINED]]]


if __name__ == "__main__":
    pass
