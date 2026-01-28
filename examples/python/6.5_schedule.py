"""
MXFP GEMM Scheduling: Scaled WMMA with MXFP4 Data Type

This example demonstrates GFX1250 scaled WMMA GEMM using MXFP4 (f4e2m1fn) inputs
with f8e8m0fnu scales, with simple prefetch pipelining.
"""

import torch

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.constraints import ScaledMMAType

from utils import parse_args, list_tests, run_test


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate MXFP4 inputs for scaled GEMM.
    
    Matches the original wave_gemm_mxfp_test.py implementation exactly.
    """
    M, N, K = shape
    torch.manual_seed(5)
    # Create inputs on specified device (CPU by default, like original test)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)
    # IMPORTANT: w is transposed here (matches original test)
    w = w.T
    # Scales are created transposed then transposed back (matches original test)
    x_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device)
    w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device)
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert packed MXFP4 (e2m1) values to f32.
    
    Matches the original wave_gemm_mxfp_test.py implementation.
    """
    # 2 because we pack fp4 in uint8
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
    """Convert e8m0 scale values to f32.
    
    Matches the original wave_gemm_mxfp_test.py implementation.
    """
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(
    x: torch.Tensor, w: torch.Tensor, x_scales: torch.Tensor, w_scales: torch.Tensor
) -> torch.Tensor:
    """Reference implementation for scaled MXFP4 GEMM.
    
    Matches the original wave_gemm_mxfp_test.py implementation.
    """
    # First convert the x and w inputs to f32
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    # Next convert the e8m0 scales to f32
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


def test_gfx1250_mxfp_gemm(is_debug=False):
    """
    GFX1250 scaled WMMA GEMM with MXFP4 data type and simple prefetch pipelining.
    
    This test uses ScaledMMAType.GFX1250_F32_16x16x128_F8F6F4 which is
    the GFX1250-specific scaled WMMA instruction for MXFP4 inputs.
    
    Key configuration:
    - 128x256 blocks with 256 K tile (64 WMMAs per wave)
    - Wave32 (GFX1250 uses 32 threads per wave)
    - MXFP4 (f4e2m1fn) data with f8e8m0fnu scales
    - Simple prefetch pipelining to overlap memory and compute
    
    WMMA Math:
    - wave_M = 64 → 64/16 = 4 WMMAs in M
    - wave_N = 128 → 128/16 = 8 WMMAs in N
    - BLOCK_K = 256 → 256/128 = 2 K sub-tiles
    - Total: 4 × 8 × 2 = 64 WMMAs per wave
    """
    shape: tuple[int, int, int] = (1024, 1024, 8192) 
    mfma_variant = ScaledMMAType.GFX1250_F32_16x16x128_F8F6F4

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Constraints - GFX1250 uses wave32
    # 64 WMMAs per wave: (64/16) × (128/16) × (256/128) = 4 × 8 × 2 = 64
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]  # 64 per wave → 4 M-tiles
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]  # 128 per wave → 8 N-tiles

    # GFX1250 uses wave32
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32,
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

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")
            b_reg = tkw.read(b, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma")
            return acc

        tkw.write(repeat, c)

    # Define the simple prefetch schedule
    @wave_schedule.wave_schedule()
    def mxfp_prefetch_schedule():
        """
        Simple 2-stage prefetch pipeline for MXFP4 scaled GEMM.
        
        Stage 0: Global-to-shared prefetch for A, B, and their scales
        Stage 1: Shared memory loads + scaled MMA compute
        
        This overlaps memory operations for iteration N+1 with compute for iteration N.
        """
        # Get the k loop to pipeline
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes for matrix A data
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.TensorLoadToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)
        reshape_a = tkw.filter_nodes(all_read_a, node_type=tkw.Reshape)

        # Get all nodes for matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.TensorLoadToLDS)
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Get all nodes for matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.TensorLoadToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)
        reshape_b = tkw.filter_nodes(all_read_b, node_type=tkw.Reshape)

        # Get all nodes for matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.TensorLoadToLDS)
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Get all bitcast operations (needed in Stage 1 with compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Check for fused TensorLoadToLDS nodes
        # For MXFP4, fusion pairs ADJACENT loads, so we may get:
        #   Pattern 1: "read_a,read_a_scale" and "read_b,read_b_scale" (data+scale fused)
        #   Pattern 2: "read_a,read_b" and "read_a_scale,read_b_scale" (data+data, scale+scale fused)
        #   Pattern 3: No fusion (all 4 separate TensorLoadToLDS nodes)
        
        # Try pattern 1 first (adjacent fusion: data+scale)
        global_to_shared_fused_a = tkw.get_node_by_tag("read_a,read_a_scale")
        global_to_shared_fused_b = tkw.get_node_by_tag("read_b,read_b_scale")
        
        # Try pattern 2 (data+data, scale+scale fusion)
        global_to_shared_data_fused = tkw.get_node_by_tag("read_a,read_b")
        global_to_shared_scale_fused = tkw.get_node_by_tag("read_a_scale,read_b_scale")
        # Combine all global-to-shared operations based on what fusion pattern occurred
        all_global_to_shared = []
        
        if len(global_to_shared_fused_a) > 0 or len(global_to_shared_fused_b) > 0:
            # Pattern 1: Adjacent fusion (data+scale)
            all_global_to_shared.extend(global_to_shared_fused_a)
            all_global_to_shared.extend(global_to_shared_fused_b)
        elif len(global_to_shared_data_fused) > 0 or len(global_to_shared_scale_fused) > 0:
            # Pattern 2: Data+data, scale+scale fusion
            all_global_to_shared.extend(global_to_shared_data_fused)
            all_global_to_shared.extend(global_to_shared_scale_fused)
        else:
            # Pattern 3: No fusion - use individual unfused nodes
            all_global_to_shared.extend(global_to_shared_a)
            all_global_to_shared.extend(global_to_shared_a_scale)
            all_global_to_shared.extend(global_to_shared_b)
            all_global_to_shared.extend(global_to_shared_b_scale)

        # Combine all shared memory loads
        all_shared_loads = []
        all_shared_loads.extend(shared_load_a)
        all_shared_loads.extend(shared_load_b)
        all_shared_loads.extend(shared_load_a_scale)
        all_shared_loads.extend(shared_load_b_scale)

        # Combine all reshape operations (for packed data)
        all_reshapes = []
        all_reshapes.extend(reshape_a)
        all_reshapes.extend(reshape_b)

        # Combine all bitcast operations
        all_bitcasts = []
        all_bitcasts.extend(bitcast_a)
        all_bitcasts.extend(bitcast_a_scale)
        all_bitcasts.extend(bitcast_b)
        all_bitcasts.extend(bitcast_b_scale)
        
        # Get the scaled MMA operation
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # Create 2-stage pipeline
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch for all inputs
            pl.set_stage(
                [
                    (all_global_to_shared,),
                    (),
                    (),
                    (),
                ],
            )

            # Stage 1: Shared memory loads + reshapes + bitcasts + compute
            pl.set_stage(
                [
                    (),
                    (),
                    (),
                    (),
                ],
            )
            
            # Stage 2: Shared memory loads + reshapes + bitcasts + compute
            pl.set_stage(
                [
                    (all_shared_loads,),
                    (all_reshapes,),
                    (all_bitcasts,),
                    (scaled_mma,),
                ],
            )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 128,
        BLOCK_N: 256,  # Increased for 64 WMMAs per wave (128 N per wave × 2 waves)
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 8,
        SHUFFLE_UNITS: 8,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        #schedule=SchedulingType.NONE,
        print_ir_after="all" if is_debug else [],
        target="gfx1250",
        dump_binaries="./",
        dump_intermediates="./",
        use_global_to_shared=True,
        #use_water_backend=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, mxfp_prefetch_schedule)
    #gemm = wave_compile(options, gemm)

    with open("gemm_mxfp4.mlir", "w") as f:
        f.write(gemm.asm)

    # Generate inputs on CPU and compute reference BEFORE touching GPU
    # (matches original test pattern exactly)
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

    # Transpose w for the kernel
    w_t = w.T.contiguous()

    # Run the kernel
    gemm(x, x_scales, w_t, w_scales, out)

    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)

    print("GFX1250 MXFP GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
