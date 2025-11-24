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
        # Get nodes
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

        # Create pipeline
        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            # Stage 0: Load from global memory and write to shared
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            # Stage 1: Load from shared and compute
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # # Now apply advanced scheduling to the KERNEL stage
        # global_load_a = tkw.filter_nodes(global_load_a, subgraph=pipeline_loop.KERNEL)
        # shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        # shared_write_a = tkw.filter_nodes(shared_write_a, subgraph=pipeline_loop.KERNEL)
        # global_load_b = tkw.filter_nodes(global_load_b, subgraph=pipeline_loop.KERNEL)
        # shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        # shared_write_b = tkw.filter_nodes(shared_write_b, subgraph=pipeline_loop.KERNEL)
        # mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)
        
        # # Partition MMA operations by K dimension into 2 groups
        # # This allows us to interleave the first half of MMA with prefetch for next iteration
        # mma_0, mma_1, mma_2, mma_3 = tkw.partition_by_dim(mma, dim=K, num_partitions=4)

        # # Similarly partition the shared memory loads
        # shared_load_a_0, shared_load_a_1, shared_load_a_2, shared_load_a_3 = tkw.partition_by_dim(
        #     shared_load_a, dim=K, num_partitions=4
        # )
        # shared_load_b_0, shared_load_b_1, shared_load_b_2, shared_load_b_3 = tkw.partition_by_dim(
        #     shared_load_b, dim=K, num_partitions=4
        # )

        # # Create instruction clusters that define the execution order
        # # Each cluster groups instructions that should execute together
        # clusters = [
        #     # Cluster 1: First half of loads + prefetch for next iteration
        #     tkw.cluster(
        #         [
        #             shared_load_a_0,  # Load first half of A from shared memory
        #             shared_load_b_0,  # Load first half of B from shared memory
        #             tkw.SchedulingBarrier([]),  # Barrier for scheduling control
        #             global_load_a,  # Prefetch A for next iteration (overlapped!)
        #             tkw.SchedulingBarrier([]),
        #             shared_load_a_1,  # Load second half of A
        #             shared_load_b_1,  # Load second half of B
        #             tkw.SchedulingBarrier([]),
        #             global_load_b,  # Prefetch B for next iteration (overlapped!)
        #             tkw.WorkgroupBarrier(),  # Ensure all waves complete loads
        #             tkw.SchedulingBarrier([]),
        #         ],
        #     ),
        #     # Cluster 2: First half of MMA operations with high priority
        #     tkw.cluster(
        #         [
        #             tkw.SetWavePrio(1),  # Increase priority for compute
        #             mma_0,  # Execute first half of MMA operations
        #             tkw.SetWavePrio(0),  # Reset priority
        #             shared_load_a_2,
        #             shared_load_b_2,
        #             mma_1,
        #             shared_load_a_3,
        #             shared_load_b_3,
        #             tkw.SharedMemoryBarrier(),  # Sync shared memory
        #             tkw.SchedulingBarrier([]),
        #         ],
        #     ),
        #     # Cluster 3: Write prefetched data to shared memory
        #     tkw.cluster(
        #         [
        #             shared_write_a,  # Write prefetched A to shared memory
        #             shared_write_b,  # Write prefetched B to shared memory
        #             mma_2,
        #             tkw.WorkgroupBarrier(),  # Ensure writes complete
        #             tkw.SchedulingBarrier([]),
        #         ],
        #     ),
        #     # Cluster 4: Second half of MMA operations
        #     tkw.cluster(
        #         [
        #             tkw.SetWavePrio(1),  # Increase priority for compute
        #             mma_3,  # Execute second half of MMA operations
        #             tkw.SetWavePrio(0),  # Reset priority
        #             tkw.SchedulingBarrier([]),
        #         ],
        #     ),
        # ]

        # # Insert barriers before the for loop and at the end of the for loop
        # tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        # tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # # Apply the cluster-based reordering to the KERNEL stage
        # tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

    # Compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 256,  # Square blocks for 4 waves
            BLOCK_K: 64,  # Larger K tile since K=32 per MFMA
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
        print_ir_after="all" if is_debug else [],
    )

    options = set_default_run_config(options)

    # Compile with compact schedule
    gemm_4wave = wave_compile(options, gemm_4wave)

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
    constraints += [tkw.WaveConstraint(M, BLOCK_M/2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N/2)]
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
            BLOCK_N: 128,
            BLOCK_K: 128,
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

