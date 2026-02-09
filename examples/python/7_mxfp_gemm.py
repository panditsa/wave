"""
Basic MXFP4 Scaled GEMM for MI350 (GFX950) - No Manual Schedule

A minimal scaled MMA GEMM kernel using MXFP4 (f4e2m1fn) inputs with
f8e8m0fnu scales. This uses the compiler's default scheduling (no manual
schedule), making it a good starting point for correctness testing before
adding advanced scheduling optimizations.

Usage:
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --debug
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --shape 512,512,4096
    python 7_mxfp_gemm.py --list_tests
"""

import torch

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.constraints import ScaledMMAType

from utils import parse_args, list_tests, run_test


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate MXFP4 inputs for scaled GEMM."""
    M, N, K = shape
    torch.manual_seed(5)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)
    # w is transposed here (matches original test)
    w = w.T
    # Scales are created transposed then transposed back
    x_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device)
    w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device)
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert packed MXFP4 (e2m1) values to f32."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 scale values to f32."""
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(
    x: torch.Tensor, w: torch.Tensor, x_scales: torch.Tensor, w_scales: torch.Tensor
) -> torch.Tensor:
    """Reference implementation for scaled MXFP4 GEMM."""
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


def test_basic_mxfp_gemm(is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)):
    """
    Basic MXFP4 scaled GEMM on GFX950 - no manual schedule.

    Uses the compiler's default scheduling. Good for correctness testing
    and as a baseline before adding manual scheduling optimizations.

    Configuration:
    - ScaledMMAType.F32_16x16x128_F8F6F4 (CDNA4/GFX950)
    - MXFP4 (f4e2m1fn) data with f8e8m0fnu scales
    - 256x256 blocks with 256 K tile
    """
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    M, N, K = shape
    BLOCK_M, BLOCK_N, BLOCK_K = block

    print(f"M: {M}, N: {N}, K: {K}, BLOCK_M: {BLOCK_M}, BLOCK_N: {BLOCK_N}, BLOCK_K: {BLOCK_K}")

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Constraints - GFX950 uses wave64
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Hyperparams - no scheduling delays needed since we're not manually scheduling
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block[0],
        BLOCK_N: block[1],
        BLOCK_K: block[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    if is_debug:
        with open("gemm_mxfp4_basic.mlir", "w") as f:
            f.write(compiled_gemm.asm)
        print("MLIR written to gemm_mxfp4_basic.mlir")

    # Generate inputs on CPU and compute reference BEFORE touching GPU
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(
        shape, device=torch.device("cpu")
    )
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    # Move inputs to GPU
    x = x.cuda()
    w = w.cuda()
    x_scales = x_scales.cuda()
    w_scales = w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    # Transpose w for the kernel (kernel expects K x N layout)
    w_t = w.T.contiguous()

    # Run the kernel
    compiled_gemm(x, x_scales, w_t, w_scales, out)

    # Verify correctness
    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)

    print("Basic MXFP GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat, args.shape, args.block)
    exit(0 if success else 1)
