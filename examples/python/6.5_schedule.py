"""
MXFP GEMM Scheduling: Scaled WMMA with MXFP4 Data Type

This example demonstrates GFX1250 scaled WMMA GEMM using MXFP4 (f4e2m1fn) inputs
with f8e8m0fnu scales, with advanced triple-buffering and wave staggering.

Key scheduling techniques:
1. Triple buffering (3-stage pipeline) for optimal memory/compute overlap
2. Wave priority manipulation (SetWavePrio) to prioritize compute waves over memory waves
3. Staggering waves for better overlap of computation and memory access
4. Custom barrier placements for fine-grained control
"""

import math
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
    GFX1250 scaled WMMA GEMM with MXFP4 data type and advanced triple-buffering.
    
    This test uses ScaledMMAType.GFX1250_F32_16x16x128_F8F6F4 which is
    the GFX1250-specific scaled WMMA instruction for MXFP4 inputs.
    
    Key configuration:
    - 256x256 blocks with 256 K tile (64 WMMAs per wave)
    - Wave32 (GFX1250 uses 32 threads per wave)
    - MXFP4 (f4e2m1fn) data with f8e8m0fnu scales
    - Triple buffering with wave staggering
    
    Advanced scheduling features (matching 6.3):
    - Prologue: Prefetch 2 iterations worth of data with barrier synchronization
    - Main loop: Custom cluster ordering with SetWavePrio for compute prioritization
    - Epilogue: Staggered loads and MMAs in 2 chunks for optimal overlap
    - Wave staggering: Conditional barriers for hi/lo wave groups
    
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
    m_iter = tkl.sym.m_iter
    n_iter = tkl.sym.n_iter
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Constraints - GFX1250 uses wave32
    # 64 WMMAs per wave: (64/16) × (128/16) × (256/128) = 4 × 8 × 2 = 64
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    # Best config: 8 waves (4x2 wave grid) - XDL efficiency 0.252
    # 16 waves (4x4) tried in iteration 11 - hurt performance (-12%)
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]  # 64 per wave → 4 M-tiles
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]  # 128 per wave → 2 N-tiles (total 8 waves)
    constraints += [tkw.IteratorBindings({m_iter: M, n_iter: N})]
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
        c_reg = tkl.Register[N, M, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[N, M, tkl.f32]) -> tkl.Register[N, M, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")
            b_reg = tkw.read(b, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")
            acc = tkw.scaled_mma(b_reg, b_scale_reg, a_reg, a_scale_reg, acc, tag="scaled_mma")
            return acc

        tkw.write(repeat, c, source=(n_iter, m_iter), target=(m_iter, n_iter))

    # Define the advanced triple-buffering schedule with wave staggering
    @wave_schedule.wave_schedule()
    def mxfp_tbuf_schedule():
        """
        Advanced 3-stage triple-buffering pipeline for MXFP4 scaled GEMM with wave staggering.
        
        - Prologue: Prefetch 2 iterations worth of data
        - Main loop:
          * Stage 0: Load from shared memory (i iteration)
          * Stage 1: Async load i+2 iteration + Compute current data (i) (overlapped)
        - Epilogue: Continue staggering
        
        This implements the same scheduling pattern as 6.3 but adapted for MXFP4
        with its 4 input tensors (A, A_scale, B, B_scale).
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

        # Get all bitcast operations (needed with compute)
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
        global_to_shared_fused = []
        
        if len(global_to_shared_fused_a) > 0 or len(global_to_shared_fused_b) > 0:
            # Pattern 1: Adjacent fusion (data+scale)
            global_to_shared_fused.extend(global_to_shared_fused_a)
            global_to_shared_fused.extend(global_to_shared_fused_b)
        elif len(global_to_shared_data_fused) > 0 or len(global_to_shared_scale_fused) > 0:
            # Pattern 2: Data+data, scale+scale fusion
            global_to_shared_fused.extend(global_to_shared_data_fused)
            global_to_shared_fused.extend(global_to_shared_scale_fused)
        else:
            # Pattern 3: No fusion - use individual unfused nodes
            global_to_shared_fused.extend(global_to_shared_a)
            global_to_shared_fused.extend(global_to_shared_a_scale)
            global_to_shared_fused.extend(global_to_shared_b)
            global_to_shared_fused.extend(global_to_shared_b_scale)

        # Get the scaled MMA operation
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # Create 3-stage pipeline (triple buffering)
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch
            pl.set_stage(
                [
                    (global_to_shared_fused,),
                    (),
                    (),
                    (),
                ],
            )
            # Stage 1: Empty (for triple buffering depth)
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
                    (shared_load_a, shared_load_b, shared_load_a_scale, shared_load_b_scale),
                    (reshape_a, reshape_b),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # PROLOGUE: Setup before the main loop
        # =====================================================================

        # Filter nodes for PROLOGUE stage (before the loop)
        # Need to re-fetch and filter for the prologue subgraph
        prologue_global_to_shared = []
        
        # Check fusion patterns again for prologue
        prologue_global_to_shared = tkw.filter_nodes(
            global_to_shared_fused, subgraph=pipeline_loop.PROLOGUE
        )
        # Prologue cluster: Best configuration from Iteration 11
        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_global_to_shared,
                    tkw.TensorCounterWait(1),  # Fully wait for prefetch
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            )
        ]

        # Create wave stagger condition using hardware constraint
        hw = tkw.get_hardware_constraint()
        mid_wave = math.prod(hw.waves_per_block) // 2
        wave_hi = hw.wave_id >= mid_wave

        # Insert conditional barrier before kernel for hi waves (staggering)
        tkw.insert_cond_barrier_before(wave_hi, pipeline_loop.KERNEL)
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SetWavePrio(1))

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================
        
        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_fused, subgraph=pipeline_loop.KERNEL
        )

        # Combine all loop shared loads
        loop_shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_a_scale = tkw.filter_nodes(shared_load_a_scale, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_b_scale = tkw.filter_nodes(shared_load_b_scale, subgraph=pipeline_loop.KERNEL)

        # Combine all loop reshapes
        loop_reshape_a = tkw.filter_nodes(reshape_a, subgraph=pipeline_loop.KERNEL)
        loop_reshape_b = tkw.filter_nodes(reshape_b, subgraph=pipeline_loop.KERNEL)

        # Combine all loop bitcasts
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(bitcast_a_scale, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(bitcast_b_scale, subgraph=pipeline_loop.KERNEL)
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)


        # similar to epilogue

        loop_shared_load_a_chunks = [loop_shared_load_a[i::2] for i in range(2)]
        loop_shared_load_a_scale_chunks = [loop_shared_load_a_scale[i::2] for i in range(2)]
        loop_reshape_a_chunks = [loop_reshape_a[i::2] for i in range(2)]
        loop_bitcast_a_chunks = [loop_bitcast_a[i::2] for i in range(2)]
        loop_bitcast_a_scale_chunks = [loop_bitcast_a_scale[i::2] for i in range(2)]

        loop_shared_load_b_chunks = [loop_shared_load_b[i::2] for i in range(2)]
        loop_shared_load_b_scale_chunks = [loop_shared_load_b_scale[i::2] for i in range(2)]
        loop_reshape_b_chunks = [loop_reshape_b[i::2] for i in range(2)]
        loop_bitcast_b_chunks = [loop_bitcast_b[i::2] for i in range(2)]
        loop_bitcast_b_scale_chunks = [loop_bitcast_b_scale[i::2] for i in range(2)]
        loop_scaled_mma_chunks = [loop_scaled_mma[i::2] for i in range(2)]


        # OPTIMIZATION: Delay scale loads until right before MMAs to reduce VGPR pressure
        # during pipeline drain (transition from main loop to epilogue)
        clusters = [
            tkw.cluster(
                [
                    # Load matrix data first (A and B chunks)
                    loop_shared_load_a_chunks[0],
                    loop_shared_load_b_chunks[0],
                    loop_shared_load_a_chunks[1],
                    loop_shared_load_b_chunks[1],
                    # First set of reshapes/bitcasts for matrix data
                    loop_reshape_a_chunks[0],
                    loop_reshape_b_chunks[0],
                    loop_bitcast_a_chunks[0],
                    loop_bitcast_b_chunks[0],
                    # NOW load scales right before MMA (reduces live range)
                    loop_shared_load_a_scale_chunks[0],
                    loop_shared_load_b_scale_chunks[0],
                    loop_bitcast_a_scale_chunks[0],
                    loop_bitcast_b_scale_chunks[0],
                    # Barrier for global-to-shared sync
                    tkw.SetWavePrio(0),
                    tkw.TensorCounterWait(1),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=True),
                    tkw.SharedMemoryBarrierWait(-1),
                    loop_global_to_shared,
                    # MMA M-tiles 0-1, K sub-tile 0 (16 MMAs)
                    loop_scaled_mma_chunks[0],


                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    tkw.SetWavePrio(1),
                    # Second chunk: reshapes/bitcasts for matrix data first
                    loop_reshape_a_chunks[1],
                    loop_reshape_b_chunks[1],
                    loop_bitcast_a_chunks[1],
                    loop_bitcast_b_chunks[1],
                    # Then scales right before MMA
                    loop_shared_load_a_scale_chunks[1],
                    loop_shared_load_b_scale_chunks[1],
                    loop_bitcast_a_scale_chunks[1],
                    loop_bitcast_b_scale_chunks[1],
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=True),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # MMA M-tiles 0-1, K sub-tile 1 (16 MMAs)
                    loop_scaled_mma_chunks[1],
                    
                    # End of loop barrier pattern
                    tkw.SetWavePrio(1),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            ),
        ]

        # =====================================================================
        # EPILOGUE: Staggered loads and MMAs in 2 chunks
        # =====================================================================

        epilogue_shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.EPILOGUE)
        epilogue_shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_shared_load_a_scale = tkw.filter_nodes(shared_load_a_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_shared_load_b_scale = tkw.filter_nodes(shared_load_b_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_reshape_a = tkw.filter_nodes(reshape_a, subgraph=pipeline_loop.EPILOGUE)
        epilogue_reshape_b = tkw.filter_nodes(reshape_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_a_scale = tkw.filter_nodes(bitcast_a_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_b_scale = tkw.filter_nodes(bitcast_b_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.EPILOGUE)

        epilogue_shared_load_a_chunks = [epilogue_shared_load_a[i::2] for i in range(2)]
        epilogue_shared_load_b_chunks = [epilogue_shared_load_b[i::2] for i in range(2)]
        epilogue_shared_load_a_scale_chunks = [epilogue_shared_load_a_scale[i::2] for i in range(2)]
        epilogue_shared_load_b_scale_chunks = [epilogue_shared_load_b_scale[i::2] for i in range(2)]
        epilogue_reshape_a_chunks = [epilogue_reshape_a[i::2] for i in range(2)]
        epilogue_reshape_b_chunks = [epilogue_reshape_b[i::2] for i in range(2)]
        epilogue_bitcast_a_chunks = [epilogue_bitcast_a[i::2] for i in range(2)]
        epilogue_bitcast_a_scale_chunks = [epilogue_bitcast_a_scale[i::2] for i in range(2)]
        epilogue_bitcast_b_chunks = [epilogue_bitcast_b[i::2] for i in range(2)]
        epilogue_bitcast_b_scale_chunks = [epilogue_bitcast_b_scale[i::2] for i in range(2)]
        epilogue_scaled_mma_chunks = [epilogue_scaled_mma[i::2] for i in range(2)]

        # Epilogue pattern matching manual MLIR:
        # 1. First set of loads (A, B, and scales)
        # 2. First set of reshapes + bitcasts
        # 3. SetWavePrio(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 4. First set of scaled MMAs
        # 5. SetWavePrio(1) + wait.tensorcnt(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 6. Second set of loads (A, B, and scales)
        # 7. Second set of reshapes + bitcasts
        # 8. SetWavePrio(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 9. Second set of scaled MMAs
        # 10. SetWavePrio(1)
        # 11. Conditional barrier (placed by insert_cond_barrier_after)
        # 12. barrier.signal(-1) + barrier.wait(-1)
        # Epilogue: Keep SchedulingBarrier - removing it hurts performance (Iteration 6 showed 0.128)
        # OPTIMIZATION: Delay scale loads until right before MMAs to reduce VGPR spills
        # Scale values were being spilled because they were loaded early but used late,
        # while massive matrix data loads in between stole their registers.
        epilogue_clusters = [
            tkw.cluster(
                [
                    # First set of matrix loads (A, B) - NO scales yet to reduce register pressure
                    tkw.TensorCounterWait(0),
                    epilogue_shared_load_a_chunks[0],
                    epilogue_shared_load_b_chunks[0],
                    # Reshapes/bitcasts for matrix data
                    epilogue_reshape_a_chunks[0],
                    epilogue_reshape_b_chunks[0],
                    epilogue_bitcast_a_chunks[0],
                    epilogue_bitcast_b_chunks[0],
                    # NOW load scales - right before MMAs that use them
                    epilogue_shared_load_a_scale_chunks[0],
                    epilogue_shared_load_b_scale_chunks[0],
                    epilogue_bitcast_a_scale_chunks[0],
                    epilogue_bitcast_b_scale_chunks[0],
                    # Barrier before first MMAs (SchedulingBarrier needed in epilogue)
                    tkw.SetWavePrio(0),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    epilogue_scaled_mma_chunks[0],
                    # Second set - barrier before loads
                    tkw.SetWavePrio(1),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # Matrix loads first
                    epilogue_shared_load_a_chunks[1],
                    epilogue_shared_load_b_chunks[1],
                    epilogue_reshape_a_chunks[1],
                    epilogue_reshape_b_chunks[1],
                    epilogue_bitcast_a_chunks[1],
                    epilogue_bitcast_b_chunks[1],
                    # Then scales right before use
                    epilogue_shared_load_a_scale_chunks[1],
                    epilogue_shared_load_b_scale_chunks[1],
                    epilogue_bitcast_a_scale_chunks[1],
                    epilogue_bitcast_b_scale_chunks[1],
                    # Barrier before second MMAs
                    tkw.SetWavePrio(0),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    epilogue_scaled_mma_chunks[1],
                    # Final barrier
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            )
        ]

        # Combine all clusters and apply reordering
        all_clusters = prologue_clusters + clusters + epilogue_clusters
        tkw.reorder_graph(pipeline_loop.KERNEL, all_clusters)

        # Get reference to last epilogue scaled MMA for conditional barrier placement
        last_epilogue_scaled_mma = epilogue_scaled_mma_chunks[1][-1]
        # Apply stagger with custom placement for post-loop conditional barrier
        wave_lo = hw.wave_id < mid_wave
        tkw.insert_cond_barrier_after(wave_lo, last_epilogue_scaled_mma)
        tkw.insert_after(last_epilogue_scaled_mma, tkw.SetWavePrio(1))

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 128,
        BLOCK_N: 256,
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
    gemm = wave_compile(options, gemm, mxfp_tbuf_schedule)
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
