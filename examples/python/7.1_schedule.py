"""
MXFP GEMM Scheduling: Scaled MMA with MXFP4 Data Type for MI350 (GFX950)

This example demonstrates GFX950 (MI350/CDNA4) scaled MMA GEMM using MXFP4 (f4e2m1fn)
inputs with f8e8m0fnu scales, with advanced buffering strategies and wave staggering.

Two buffering strategies are demonstrated:
1. Triple buffering (3-stage pipeline) - test_gfx950_mxfp_gemm
   - Prefetches 2 iterations ahead for maximum memory/compute overlap
   - Uses more shared memory but provides better latency hiding

2. Double buffering (2-stage pipeline) - test_gfx950_mxfp_gemm_double_buffering
   - Prefetches 1 iteration ahead (ping-pong buffering)
   - Uses less shared memory with simpler synchronization

Key scheduling techniques:
1. Multi-stage pipeline for memory/compute overlap
2. GatherToLDS for async global-to-shared transfers (no fusion on GFX950)
3. Wave priority manipulation (SetWavePrio) to prioritize compute waves over memory waves
4. Staggering waves for better overlap of computation and memory access
5. Custom barrier placements for fine-grained control
6. Partitioning by K dimension to interleave compute with memory ops
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
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )
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


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """
    GFX950 (MI350) scaled MMA GEMM with MXFP4 data type and double buffering (2-stage pipeline).

    This test uses ScaledMMAType.F32_16x16x128_F8F6F4 which is
    the CDNA4 (GFX950) scaled MMA instruction for MXFP4 inputs.

    Key configuration:
    - 4 waves
    - 256x256 blocks with 256 K tile (64 MMAs per wave)
    - MXFP4 (f4e2m1fn) data with f8e8m0fnu scales
    - Double buffering (2-stage pipeline) with wave staggering
    - GatherToLDS for async global-to-shared transfers
    """
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    M, N, K = shape
    BLOCK_M, BLOCK_N, BLOCK_K = block

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    m_iter = tkl.sym.m_iter
    n_iter = tkl.sym.n_iter
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Constraints - GFX950 uses wave64
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]  # 128 per wave → 2 M-tiles
    constraints += [
        tkw.WaveConstraint(N, BLOCK_N / 2)
    ]  # 128 per wave → 2 N-tiles (total 8 waves)
    constraints += [tkw.IteratorBindings({m_iter: M, n_iter: N})]
    # GFX950 uses wave64
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
            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    # Define the 2-stage double-buffering schedule adapted for GFX950
    @wave_schedule.wave_schedule()
    def mxfp_double_buf_schedule():
        """
        2-stage double-buffering pipeline for MXFP4 scaled GEMM on GFX950.

        Uses GatherToLDS for async global-to-shared transfers (no fusion on GFX950).

        - Prologue: Prefetch 1 iteration worth of data (handled by pipeline)
        - Main loop:
          * Partitioned by K dimension to interleave compute with memory ops
          * Async GatherToLDS for next iteration overlapped with current compute
        - Wave staggering via tkw.stagger()
        """
        # Get the k loop to pipeline
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes for matrix A data - uses GatherToLDS on GFX950 (no fusion)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes for matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Get all nodes for matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Get all nodes for matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Get all bitcast operations (needed with compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Get the scaled MMA operation
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # Create 2-stage pipeline (double buffering)
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch (GatherToLDS, no fusion)
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_a_scale,
                        global_to_shared_b,
                        global_to_shared_b_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                        shared_load_a_scale,
                        shared_load_b_scale,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )
        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        # Combine all global-to-shared operations
        loop_global_to_shared = (
            tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL)
        )

        # Combine all loop shared loads
        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_scale = tkw.filter_nodes(
            shared_load_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b_scale = tkw.filter_nodes(
            shared_load_b_scale, subgraph=pipeline_loop.KERNEL
        )

        # Combine all loop bitcasts
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition MMAs, shared loads, and bitcasts by K dimension for interleaving
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_0, loop_shared_load_a_scale_1 = tkw.partition_by_dim(
            loop_shared_load_a_scale, dim=K, num_partitions=2
        )
        loop_shared_load_b_scale_0, loop_shared_load_b_scale_1 = tkw.partition_by_dim(
            loop_shared_load_b_scale, dim=K, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        independent_global_count = len(loop_global_to_shared)

        # Double buffering cluster structure (follows 6.2's async_gemm_schedule pattern)
        clusters = [
            # Cluster 1: First half of shared loads + bitcasts + async global-to-shared
            tkw.cluster(
                [
                    loop_shared_load_a_0,
                    loop_shared_load_a_scale_0,
                    loop_shared_load_b_0,
                    loop_shared_load_b_scale_0,
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b_0,
                    loop_bitcast_b_scale_0,
                    tkw.SchedulingBarrier([]),
                    # Issue async global-to-shared for next iteration
                    loop_global_to_shared,
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: First half of MMAs with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second half of shared loads + bitcasts
            tkw.cluster(
                [
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_shared_load_b_1,
                    loop_shared_load_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 4: Second half of MMAs with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block[0],
        BLOCK_N: block[1],
        BLOCK_K: block[2],
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
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
        print_mlir_file="gemm_mxfp4_dbuf_4wave.mlir",
        print_mlir=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, mxfp_double_buf_schedule)

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

    # Transpose w for the kernel
    w_t = w.T.contiguous()

    # Run the kernel
    gemm(x, x_scales, w_t, w_scales, out)

    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)

    print("GFX950 MXFP GEMM double buffering 4-wave test passed!")


def test_dbuf_8wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """
    GFX950 (MI350) scaled MMA GEMM with MXFP4 data type and double buffering (2-stage pipeline).

    This test uses ScaledMMAType.F32_16x16x128_F8F6F4 which is
    the CDNA4 (GFX950) scaled MMA instruction for MXFP4 inputs.

    Key configuration:
    - 8 waves
    - 256x256 blocks with 256 K tile (64 MMAs per wave)
    - MXFP4 (f4e2m1fn) data with f8e8m0fnu scales
    - Double buffering (2-stage pipeline) with wave staggering
    - GatherToLDS for async global-to-shared transfers
    """
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    M, N, K = shape
    BLOCK_M, BLOCK_N, BLOCK_K = block

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    m_iter = tkl.sym.m_iter
    n_iter = tkl.sym.n_iter
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Constraints - GFX950 uses wave64
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]  # 64 per wave → 4 M-tiles
    constraints += [
        tkw.WaveConstraint(N, BLOCK_N / 2)
    ]  # 128 per wave → 2 N-tiles (total 8 waves)
    constraints += [tkw.IteratorBindings({m_iter: M, n_iter: N})]
    # GFX950 uses wave64
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
            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    # Define the 2-stage double-buffering schedule adapted for GFX950
    @wave_schedule.wave_schedule()
    def mxfp_double_buf_schedule():
        """
        2-stage double-buffering pipeline for MXFP4 scaled GEMM on GFX950.

        Uses GatherToLDS for async global-to-shared transfers (no fusion on GFX950).

        - Prologue: Prefetch 1 iteration worth of data (handled by pipeline)
        - Main loop:
          * Partitioned by K dimension to interleave compute with memory ops
          * Async GatherToLDS for next iteration overlapped with current compute
        - Wave staggering via tkw.stagger()
        """
        # Get the k loop to pipeline
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes for matrix A data - uses GatherToLDS on GFX950 (no fusion)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes for matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Get all nodes for matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Get all nodes for matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Get all bitcast operations (needed with compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Get the scaled MMA operation
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # Create 2-stage pipeline (double buffering)
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch (GatherToLDS, no fusion)
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_a_scale,
                        global_to_shared_b,
                        global_to_shared_b_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                        shared_load_a_scale,
                        shared_load_b_scale,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )
        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        # Combine all global-to-shared operations
        loop_global_to_shared = (
            tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL)
        )

        # Combine all loop shared loads
        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_scale = tkw.filter_nodes(
            shared_load_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b_scale = tkw.filter_nodes(
            shared_load_b_scale, subgraph=pipeline_loop.KERNEL
        )

        # Combine all loop bitcasts
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition MMAs, shared loads, and bitcasts by K dimension for interleaving
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_0, loop_shared_load_a_scale_1 = tkw.partition_by_dim(
            loop_shared_load_a_scale, dim=K, num_partitions=2
        )
        loop_shared_load_b_scale_0, loop_shared_load_b_scale_1 = tkw.partition_by_dim(
            loop_shared_load_b_scale, dim=K, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        independent_global_count = len(loop_global_to_shared)

        # Double buffering cluster structure (follows 6.2's async_gemm_schedule pattern)
        clusters = [
            # Cluster 1: First half of shared loads + bitcasts + async global-to-shared
            tkw.cluster(
                [
                    loop_shared_load_a_0,
                    loop_shared_load_a_scale_0,
                    loop_shared_load_b_0,
                    loop_shared_load_b_scale_0,
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b_0,
                    loop_bitcast_b_scale_0,
                    tkw.SchedulingBarrier([]),
                    # Issue async global-to-shared for next iteration
                    loop_global_to_shared,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: First half of MMAs with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second half of shared loads + bitcasts
            tkw.cluster(
                [
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_shared_load_b_1,
                    loop_shared_load_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 4: Second half of MMAs with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling
        tkw.stagger(pipeline_loop.KERNEL)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block[0],
        BLOCK_N: block[1],
        BLOCK_K: block[2],
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
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, mxfp_double_buf_schedule)

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

    # Transpose w for the kernel
    w_t = w.T.contiguous()

    # Run the kernel
    gemm(x, x_scales, w_t, w_scales, out)

    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)

    print("GFX950 MXFP GEMM double buffering 8-wave test passed!")


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
