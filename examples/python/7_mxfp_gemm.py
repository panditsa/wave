"""
MXFP4 Scaled GEMM for GFX950 (MI350) - No Manual Schedule

Usage:
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --debug
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --shape 512,512,4096
    python 7_mxfp_gemm.py --list_tests
"""

import torch

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.schedules import get_tagged_mxfp4_gemm

from utils import parse_args, list_tests, run_test

SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(shape, device=torch.device("cpu")):
    """Generate random MXFP4 inputs and scales."""
    M, N, K = shape
    torch.manual_seed(5)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)
    w = w.T
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x):
    """Convert packed MXFP4 (e2m1) values to f32."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    return torch.tensor(lut, dtype=torch.float32, device=x.device)[x.long()]


def e8m0_to_f32(x):
    """Convert e8m0 scale values to f32."""
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(x, w, x_scales, w_scales):
    """Reference scaled MXFP4 GEMM in PyTorch."""
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T).T
    x_scales_f32 = e8m0_to_f32(
        x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    w_scales_f32 = e8m0_to_f32(
        w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    return torch.mm(x_f32 * x_scales_f32, w_f32 * w_scales_f32.T)


def test_basic_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """MXFP4 scaled GEMM with compiler-default scheduling (no manual schedule)."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)

    # Override to use default scheduling instead of manual
    options.schedule = None
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    if is_debug:
        with open("gemm_mxfp4_basic.mlir", "w") as f:
            f.write(compiled_gemm.asm)
        print("MLIR written to gemm_mxfp4_basic.mlir")

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    compiled_gemm(x, x_scales, w.T.contiguous(), w_scales, out)
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

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
