import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_zeros,
)
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType, MMAOperand, GenericDot
from wave_lang.kernel.wave.templates.gemm import (
    get_gemm_kernel,
    get_gemm_kernel_transpose_a_b,
)
from wave_lang.kernel.wave.templates.test_kernels import (
    get_gemm_prefetch_kernel_and_schedule,
)
from wave_lang.kernel.wave.schedules.gemm_two_pp_cluster import (
    get_two_pp_cluster_schedule,
)
from wave_lang.kernel.lang import DataType

def main():
    shape = (128, 256, 1024)
    mfma_variant = MMAType.GFX1250_F32_16x16x32_F16
    threads_per_wave = 32

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=threads_per_wave, mma_type=mfma_variant)
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, tag="read_a")
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, tag="read_b")
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    # Define the advanced schedule
    @wave_schedule.wave_schedule()
    def advanced_schedule():
        """
        Advanced scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates a sophisticated instruction ordering that:
        1. Interleaves compute (MMA) with memory operations
        2. Uses wave priorities to ensure compute waves get resources when needed
        3. Implements ping-pong buffering via stagger() for double buffering
        4. Carefully places barriers to ensure correctness while maximizing parallelism
        """
    
        # Get nodes to be manipulated in the schedule
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and GatherToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.TensorLoadToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and GatherToLDS nodes
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.TensorLoadToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        mma = tkw.get_node_by_tag("mma")

        breakpoint()
        # Create a pipeline with 2 stages
        pipeline_loop = tkw.pipeline(k_loop)

        # First, create the basic 2-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        # Filter nodes to only include those in the KERNEL stage
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Partition node lists by K dimension for fine-grained scheduling
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        # Calculate the number of independent global_to_shared operations for MemoryCounterWait
        independent_global_count = len(global_to_shared_a) + len(global_to_shared_b)

        # Create cluster ordering with async operations
        clusters = [
            tkw.cluster(
                [
                    shared_load_a_0,
                    shared_load_b_0,
                    tkw.SchedulingBarrier([]),
                    global_to_shared_a,
                    global_to_shared_b,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWait(load=independent_global_count),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWait(load=0),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_1,
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

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)
    
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=False,
        use_global_to_shared=True,
        print_ir_after="all",
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, advanced_schedule)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    torch.testing.assert_close(c, iree_ref, check_device=False)
    print("ok")


if __name__ == "__main__":
    main()
