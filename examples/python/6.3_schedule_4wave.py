"""
GEMM Scheduling Part 3: 4-Wave Configuration Matching hipBLASLt Style

This example demonstrates a simpler, more compact scheduling approach similar to
hipBLASLt's optimized kernels:
1. Using 4 waves instead of 8 (reduces wave management overhead)
2. Using larger K-dimension MFMA instructions (K=32 vs K=16)
3. Simpler pipelining without wave staggering
4. More aggressive instruction interleaving within a single pipeline stage
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
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params

from utils import parse_args, list_tests, run_test


def test_gemm_4wave_scheduling(is_debug=False):
    """
    Compact GEMM scheduling with 4 waves, matching hipBLASLt style.
    """
    shape: tuple[int, int, int] = (4096, 4096, 4096)
    # Use the larger K-dimension MFMA variant (K=32 vs K=16)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Configure for 4 waves: 2x2 grid
    # BLOCK_M=128, BLOCK_N=128 with WAVE_M=64, WAVE_N=64 gives 2x2=4 waves
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]  # 2 waves in M dimension
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]  # 2 waves in N dimension

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,  # Using K=32 MFMA
        )
    ]

    # Define the kernel
    @tkw.wave(constraints)
    def gemm_4wave(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the schedule - simpler than the 8-wave version
    @wave_schedule.wave_schedule()
    def compact_schedule():
        """
        Compact 4-wave scheduling without staggering.
        
        This schedule:
        1. Uses simple 2-stage pipelining
        2. Partitions MMA by K dimension for better instruction interleaving
        3. Uses wave priorities strategically but minimally
        4. NO staggering - all waves execute the same schedule
        """

        # Get nodes - DIFFERENT pattern with global_to_shared!
        k_loop = tkw.get_node_by_tag("k_loop")
    
        # With global_to_shared, we get GatherToLDS nodes instead of separate load+write
        # Pattern from 6.2_schedule.py lines 343-351:
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)
    
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)
    
        # NO shared_write_a or shared_write_b! They're fused into global_to_shared!
    
        mma = tkw.get_node_by_tag("mma")

        # Create pipeline - DIFFERENT with global_to_shared!
        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            # Stage 0: Global-to-shared (fused operation, no separate write!)
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),  # Empty! No separate shared_write
                ],
            )
            # Stage 1: Load from shared and compute
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # NOW: Apply aggressive interleaving in the KERNEL stage
        # Filter nodes to KERNEL subgraph
        global_to_shared_a = tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)
    
        # CRITICAL CHANGE 1: Partition MMA into 2 groups (like 6.2_schedule.py)
        # The scheduling graph has limited unique IDs, so we use 2 partitions
        # This creates: First half (32 MFMAs) and Second half (32 MFMAs)
        # Still enables good interleaving like hipBLASLt
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
    
        # CRITICAL CHANGE 2: Partition shared loads to match
        # Each half needs its corresponding data
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )
    
        # Calculate how many global_to_shared ops for memory counter waits
        independent_global_count = tkw.get_node_count(global_to_shared_a) + tkw.get_node_count(global_to_shared_b)

        # CRITICAL CHANGE 3: Interleave loads and MFMAs (matching hipBLASLt)
        # HipBLASLt: Load all → Execute MFMAs with interleaved prefetch
        # We use 2 K-partitions matching BLOCK_K=64, MFMA_K=32
        clusters = []
    
        # Cluster 1: Load first half + issue gather_to_lds for next iteration
        # Matches hipBLASLt pattern (713-733): load, then issue prefetch
        clusters.append(
            tkw.cluster([
                # Load data for first K-slice (32 MFMAs)
                shared_load_a_0,
                shared_load_b_0,
                tkw.SchedulingBarrier([]),
            
                # Issue gather_to_lds for NEXT loop iteration
                # Matches hipBLASLt's buffer_load at 699-710
                global_to_shared_a,
                global_to_shared_b,
                tkw.SchedulingBarrier([]),
                tkw.WorkgroupBarrier(),
                tkw.SchedulingBarrier([]),
            ])
        )
    
        # Cluster 2: First half of MFMAs
        # Execute while gather_to_lds runs in background (hipBLASLt 744-832)
        clusters.append(
            tkw.cluster([
                tkw.SetWavePrio(1),
                mma_0,  # First 32 MFMAs (first K-slice)
                tkw.SetWavePrio(0),
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(load=independent_global_count),
                tkw.WorkgroupBarrier(),
                tkw.SchedulingBarrier([]),
            ])
        )
    
        # Cluster 3: Load second half
        # Matches hipBLASLt's mid-compute data loading
        clusters.append(
            tkw.cluster([
                shared_load_a_1,
                shared_load_b_1,
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(load=0),
                tkw.WorkgroupBarrier(),
                tkw.SchedulingBarrier([]),
            ])
        )
        
        # Cluster 4: Second half of MFMAs
        # Matches hipBLASLt pattern (835-984)
        clusters.append(
            tkw.cluster([
                tkw.SetWavePrio(1),
                mma_1,  # Second 32 MFMAs (second K-slice)
                tkw.SetWavePrio(0),
                tkw.SchedulingBarrier([]),
            ])
        )

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

    # Compile options

    hyperparams = {
        M: shape[0],
        N: shape[1],
        K: shape[2],
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,  # Required for custom schedules
        use_global_to_shared=True,
        use_scheduling_barriers=True,  # CRITICAL: Enable fine-grained scheduling
        print_ir_after="all" if is_debug else [],
    )

    options = set_default_run_config(options)

    # Compile with compact schedule
    gemm_4wave = wave_compile(options, gemm_4wave, compact_schedule)

    # Test
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    gemm_4wave(a, b, c)

    if is_debug:
        print("=" * 80)
        print("4-WAVE COMPACT SCHEDULE")
        print("=" * 80)
        print(f"Block size: 128x128")
        print(f"Wave grid: 2x2 = 4 waves")
        print(f"MFMA type: F32_16x16x32_F16 (K=32)")
        print(f"K tile: 128 (4 MFMA instructions per K-tile)")
        print("=" * 80)
        print(gemm_4wave.asm)
        # write to a file
        with open("6.3_schedule_4wave.asm", "w") as f:
            f.write(gemm_4wave.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("4-wave compact scheduling test passed!")


def test_gemm_4wave_minimal(is_debug=False):
    """
    Even more minimal 4-wave GEMM - closest to hipBLASLt style.
    """
    shape: tuple[int, int, int] = (4096, 4096, 4096)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def gemm_minimal(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def minimal_schedule():
        """Minimal scheduling - just basic pipelining."""
        k_loop = tkw.get_node_by_tag("k_loop")
        load_a = tkw.get_node_by_tag_and_type("read_a", tkw.Read)
        global_load_a, shared_load_a = tkw.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = tkw.get_node_by_tag_and_type("read_a", tkw.Write)
        load_b = tkw.get_node_by_tag_and_type("read_b", tkw.Read)
        global_load_b, shared_load_b = tkw.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = tkw.get_node_by_tag_and_type("read_b", tkw.Write)
        mma = tkw.get_node_by_tag("mma")

        # Just basic pipeline, no fancy reordering
        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 128,
            BLOCK_K: 128,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
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
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
        dump_binaries="binaries/6.3_schedule_4wave_minimal",
    )

    options = set_default_run_config(options)
    gemm_minimal = wave_compile(options, gemm_minimal, minimal_schedule)

    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    gemm_minimal(a, b, c)

    if is_debug:
        print("=" * 80)
        print("4-WAVE MINIMAL SCHEDULE")
        print("=" * 80)
        print(gemm_minimal.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("4-wave minimal scheduling test passed!")

def test_gemm_4wave_simple(is_debug=False):
    """
    Even more minimal 4-wave GEMM - closest to hipBLASLt style.
    """
    shape: tuple[int, int, int] = (4096, 4096, 4096)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def gemm_simple(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128, 
            BLOCK_N: 256, 
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )

    options = set_default_run_config(options)
    gemm_simple = wave_compile(options, gemm_simple)

    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    gemm_simple(a, b, c)

    if is_debug:
        print("=" * 80)
        print("4-WAVE MINIMAL SCHEDULE")
        print("=" * 80)
        print(gemm_simple.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("4-wave minimal scheduling test passed!")


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

