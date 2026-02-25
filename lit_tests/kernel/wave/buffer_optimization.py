# RUN: python %s | FileCheck %s

"""
Lit test for the buffer count optimization pass (post_scheduling_buffer_opt).

Uses the asymmetric MXFP4 GEMM from the templates/schedules infrastructure
(same kernel as test_dbuf_4wave_mxfp_asymmetric_gemm in 7.1_schedule.py).
The 3-stage asymmetric pipeline has:
  - A (data + scale): global -> LDS -> VGPRs (triple-buffered without opt)
  - B (data + scale): global -> VGPRs directly (no LDS)
The optimization detects that LDS reads are at later pipeline stages than
writes and reduces A's LDS buffer count from triple to double.
"""

import logging

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE


@run_test
def test_asymmetric_mxfp4_buffer_optimization():
    """
    Asymmetric MXFP4 GEMM with the real 3-stage pipeline schedule.

    A (data + scale) goes through LDS via GatherToLDS at stage 0,
    with shared reads at stages 1-2.
    B (data + scale) goes directly from global memory — no LDS.

    Without optimization: triple-buffered LDS for A data and A scale
      → 3 views for A data (256x128xi8) + 3 views for A scale (256x8xi8) = 6 views
    With consume-before-write optimization: double-buffered
      → 2 views for A data (256x128xi8) + 2 views for A scale (256x8xi8) = 4 views
    """
    shape = (1024, 1024, 8192)
    block = (256, 256, 256)

    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    schedule = get_mxfp4_asymmetric_schedule()

    options.compile_to_mlir = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True

    result = wave_compile(options, gemm, schedule)

    print(result.asm)

    # After minimize_shared_allocs, all LDS buffers are merged into a single
    # memref.alloc and accessed via memref.view at different offsets.
    #
    # With the consume-before-write optimization (triple → double):
    #   2 views for A data (256x128xi8 each, 32KB)
    #   2 views for A scale (256x8xi8 each, 2KB)
    #   Total: 4 views, ~68KB LDS
    #
    # Without the optimization (triple-buffer):
    #   3 views for A data + 3 views for A scale = 6 views, ~102KB LDS

    # CHECK-LABEL: func.func @gemm

    # Single LDS allocation for all buffers.
    # CHECK: memref.alloc() : memref<{{.*}}xi8, #gpu.address_space<workgroup>>

    # Exactly 4 memref.view ops (double-buffered):
    #   2 for A scale (256x8xi8) then 2 for A data (256x128xi8).
    # With triple-buffering there would be 6 views (3 + 3).
    # CHECK-COUNT-2: memref.view {{.*}} to memref<256x8xi8, #gpu.address_space<workgroup>>
    # CHECK-COUNT-2: memref.view {{.*}} to memref<256x128xi8, #gpu.address_space<workgroup>>

    # No 5th view (would exist with triple-buffering).
    # CHECK-NOT: memref.view {{.*}} to memref<256x{{.*}}xi8, #gpu.address_space<workgroup>>

    # Kernel loop.
    # CHECK: scf.for


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
