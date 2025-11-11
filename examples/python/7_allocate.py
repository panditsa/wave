"""
Memory Allocation Examples

Demonstrates explicit memory allocation patterns, particularly for shared memory management
in GEMM operations.
"""

import torch
import pdb
import traceback

import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.dtype import f16, f32
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)

from utils import parse_args, list_tests, run_test


# Define symbolic dimensions for our matrices
M = sym.M  # Rows of A and C
N = sym.N  # Rows of B and columns of C
K = sym.K  # Columns of A and B

# Define workgroup tile sizes
BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K

# Define the address space for our memory buffers
ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C


def explicit_shared_gemm_test(is_debug=False):
    """GEMM with explicit shared memory management."""

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 4),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        ),
    ]

    @tkw.wave(constraints)
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],
        b: Memory[N, K, ADDRESS_SPACE_B, f16],
        c: Memory[M, N, ADDRESS_SPACE_C, f32],
    ):
        # Allocate shared memory explicitly
        # shape: logical shape, distributed_shape: per-workgroup tile size
        a_shared = tkw.allocate((M, K), (BLOCK_M, BLOCK_K), f16, SHARED_ADDRESS_SPACE)
        b_shared = tkw.allocate((N, K), (BLOCK_N, BLOCK_K), f16, SHARED_ADDRESS_SPACE)

        c_acc = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_acc])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            a_global = tkw.read(a)
            b_global = tkw.read(b)

            tkw.write(a_global, a_shared)
            tkw.write(b_global, b_shared)

            tkw.shared_memory_barrier()

            a_reg = tkw.read(a_shared)
            b_reg = tkw.read(b_shared)

            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Test setup
    m, n, k = 128, 256, 128

    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    hyperparams = {
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
        M: m,
        N: n,
        K: k,
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        use_scheduling_barriers=False,
        print_ir_after="all" if is_debug else [],
        minimize_shared_allocs=False,
    )
    options = set_default_run_config(options)

    compiled_gemm = wave_compile(options, gemm)

    if is_debug:
        print(compiled_gemm.asm)
        with open("manual_pingpong.mlir", "w") as f:
            f.write(compiled_gemm.asm)

    compiled_gemm(a, b, c)

    expected = torch.matmul(a, b.t())
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"Max difference: {(c - expected).abs().max()}"

    print("Explicit shared GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
    else:
        # run the test with error handling for debugging
        for i in range(args.repeat):
            try:
                success = run_test(args.test, globals(), args.debug, 1)
                if success:
                    print(f"Test {i} passed")
                else:
                    exit(1)
            except SystemExit as e:
                print(f"SystemExit: code={e.code!r}")
                traceback.print_exc()
                pdb.post_mortem(e.__traceback__)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Test {i} failed")
                traceback.print_exc()
                exit(1)
