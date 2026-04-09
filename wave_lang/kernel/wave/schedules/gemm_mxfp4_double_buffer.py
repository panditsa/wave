# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MXFP4 Scaled GEMM Double Buffer Schedule for CDNA4 (GFX950)

Reusable 2-stage pipeline schedule for MXFP4 scaled GEMM on GFX950.
Handles 4 input tensors (A data, A scale, B data, B scale) with bitcasts.

Stage 0: GatherToLDS async prefetch | Stage 1: shared loads + bitcasts + MMA
K-dimension partitioned into 2 halves for memory/compute interleaving.

Required kernel tags: k_loop, read_a, read_a_scale, read_b, read_b_scale,
bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale, scaled_mma.
Requires use_global_to_shared=True and threads_per_wave=64.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.wave.scheduling.resources import Operation


def get_mxfp4_dbuf_schedule(use_stagger: bool = True):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
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
        loop_global_to_shared = (
            tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL)
        )

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

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition by K dimension for interleaving compute with memory ops.
        # NOTE: Bitcasts MUST also be partitioned by K to match their producer
        # shared loads, otherwise reorder_graph fails with
        # "Cannot find producer(s)" because bitcasts in an earlier cluster
        # would depend on shared loads in a later cluster.
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

        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            loop_shared_load_a_0,
            loop_shared_load_a_scale_0,
            loop_shared_load_b_0,
            loop_shared_load_b_scale_0,
            loop_bitcast_a_0,
            loop_bitcast_a_scale_0,
            loop_bitcast_b_0,
            loop_bitcast_b_scale_0,
            tkw.SchedulingBarrier([]),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
        ]
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
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
            # Cluster 2: Second K-partition shared loads/bitcasts
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
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert shared memory barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_pingpong_schedule(use_stagger: bool = True, shape: tuple = None):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
        shape: Tuple of (M, N, K) dimensions. If provided and bigger than
            (1024, 1024, 1024), an extra WorkgroupBarrier will be added
            after the first SchedulingBarrier in cluster 0.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_b,
                        all_read_a_scale,
                        all_read_b_scale,
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
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        ) + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_a_scale = tkw.filter_nodes(
            all_read_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_b_scale = tkw.filter_nodes(
            all_read_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_all_read_a_scale_0, loop_all_read_a_scale_1 = tkw.partition_by_dim(
            loop_all_read_a_scale, dim=K, num_partitions=2
        )
        loop_all_read_b_scale_0, loop_all_read_b_scale_1 = tkw.partition_by_dim(
            loop_all_read_b_scale, dim=K, num_partitions=2
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

        # If the bus gets congested and cluster memory dependency are affected, we must add a second barrier to fix the timing and prevent incorrect output results.
        # In case a second a second workgroup barrier is needed, another schedule is created to hide the latency of that second barrier, by scheduling safe ds_read ops before the second barrier (see get_mxfp4_dbuf_mixed_pingpong_schedule).
        use_extra_barrier = True
        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=0),
            tkw.WorkgroupBarrier(),
        ]
        if use_extra_barrier:
            cluster_0_ops.append(tkw.WorkgroupBarrier())
        cluster_0_ops.extend(
            [
                loop_global_to_shared,
                tkw.SchedulingBarrier([]),
                loop_shared_load_a_0,
                loop_shared_load_b_0,
                loop_bitcast_a_0,
                loop_bitcast_a_scale_0,
                loop_bitcast_b_0,
                loop_bitcast_b_scale_0,
                loop_all_read_a_scale_0,  # prefetch A & B scales for next iteration
                loop_all_read_b_scale_0,
                tkw.SchedulingBarrier([]),
            ]
        )
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_shared_load_a_1,
                    loop_shared_load_b_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    loop_all_read_a_scale_1,
                    loop_all_read_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        # tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_pingpong_schedule_Bshuffled(
    use_stagger: bool = True, shape: tuple = None
):
    """Return a double-buffered MXFP4 schedule for wave_compile().
    Same as get_mxfp4_dbuf_pingpong_schedule(), but B data is shuffled and read
    from global memory directly to VGPRs.

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
        shape: Tuple of (M, N, K) dimensions. If provided and bigger than
            (1024, 1024, 1024), an extra WorkgroupBarrier will be added
            after the first SchedulingBarrier in cluster 0.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS + Global to VGPR prefetch
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        all_read_b,
                        all_read_a_scale,
                        all_read_b_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (shared_load_a,),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_b = tkw.filter_nodes(all_read_b, subgraph=pipeline_loop.KERNEL)
        loop_all_read_a_scale = tkw.filter_nodes(
            all_read_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_b_scale = tkw.filter_nodes(
            all_read_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )

        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )

        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )

        # If the bus gets congested and cluster memory dependency are affected, we must add a second barrier to fix the timing and prevent incorrect output results.
        # In case a second a second workgroup barrier is needed, another schedule is created to hide the latency of that second barrier, by scheduling safe ds_read ops before the second barrier (see get_mxfp4_dbuf_mixed_pingpong_schedule).
        use_extra_barrier = True
        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=0),
            tkw.WorkgroupBarrier(),
        ]
        if use_extra_barrier:
            cluster_0_ops.append(tkw.WorkgroupBarrier())
        cluster_0_ops.extend(
            [
                loop_global_to_shared,
                tkw.SchedulingBarrier([]),
                loop_shared_load_a_0,
                loop_bitcast_a_0,
                loop_bitcast_a_scale,
                loop_bitcast_b_0,
                loop_bitcast_b_scale,
                loop_all_read_b,
                loop_all_read_a_scale,  # prefetch A & B scales for next iteration
                loop_all_read_b_scale,
                tkw.SchedulingBarrier([]),
            ]
        )
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_shared_load_a_1,
                    loop_bitcast_a_1,
                    loop_bitcast_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
    use_stagger: bool = True, shape: tuple = None
):
    """Return a double-buffered MXFP4 schedule for wave_compile().
    Same as get_mxfp4_dbuf_pingpong_schedule_Bshuffled(), but B data is read
    from global memory to LDS.

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
        shape: Tuple of (M, N, K) dimensions. If provided and bigger than
            (1024, 1024, 1024), an extra WorkgroupBarrier will be added
            after the first SchedulingBarrier in cluster 0.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Matrix B data - GatherToLDS (global->shared) + Read (shared load)
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS + Global to VGPR prefetch
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_b,
                        all_read_a_scale,
                        all_read_b_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        ) + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_a_scale = tkw.filter_nodes(
            all_read_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_all_read_b_scale = tkw.filter_nodes(
            all_read_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )

        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )

        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )

        # If the bus gets congested and cluster memory dependency are affected, we must add a second barrier to fix the timing and prevent incorrect output results.
        # In case a second a second workgroup barrier is needed, another schedule is created to hide the latency of that second barrier, by scheduling safe ds_read ops before the second barrier (see get_mxfp4_dbuf_mixed_pingpong_schedule).
        use_extra_barrier = True
        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=0),
            tkw.WorkgroupBarrier(),
        ]
        if use_extra_barrier:
            cluster_0_ops.append(tkw.WorkgroupBarrier())
        cluster_0_ops.extend(
            [
                loop_global_to_shared,
                tkw.SchedulingBarrier([]),
                loop_shared_load_a_0,
                loop_shared_load_b_0,
                loop_bitcast_a_0,
                loop_bitcast_a_scale,
                loop_bitcast_b_0,
                loop_bitcast_b_scale,
                loop_all_read_a_scale,  # prefetch A & B scales for next iteration
                loop_all_read_b_scale,
                tkw.SchedulingBarrier([]),
            ]
        )
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_shared_load_a_1,
                    loop_shared_load_b_1,
                    loop_bitcast_a_1,
                    loop_bitcast_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger: bool = True):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Hides the latency of the second WorkgroupBarrier by issuing a "safe"
    subset of LDS vector.loads (rows owned by this wave) before the barrier,
    then interleaving the dependent loads with compute after it.

    Safe/dependent split (per K-partition):
      - A / A_scale : M:0,1 safe  |  M:2,3 dependent
      - B / B_scale : N:0,1,4,5 safe  |  N:2,3,6,7 dependent
      - MFMAs       : M:0,1 x N:0,1,4,5 safe  |  rest dependent

    Args:
        use_stagger: Enable wave staggering via tkw.stagger().
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K
    M = tkl.sym.M
    N = tkl.sym.N

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
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
        loop_global_to_shared = (
            tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL)
        )

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

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition by K dimension first
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

        # =====================================================================
        # Split A / A_scale by M dimension: safe = M:0,1 | dependent = M:2,3
        # These correspond to rows owned by the "early" wave group (wave_idx<4).
        # After memory_counter_wait(N_gather), this wave's LDS writes are done,
        # so M:0,1 rows are readable before the global workgroup barrier.
        # =====================================================================
        # K=0 partition
        loop_sla_0_safe, loop_sla_0_dep = tkw.partition_by_dim(
            loop_shared_load_a_0, dim=M, num_partitions=2
        )
        loop_slascale_0_safe, loop_slascale_0_dep = tkw.partition_by_dim(
            loop_shared_load_a_scale_0, dim=M, num_partitions=2
        )
        loop_bca_0_safe, loop_bca_0_dep = tkw.partition_by_dim(
            loop_bitcast_a_0, dim=M, num_partitions=2
        )
        loop_bcascale_0_safe, loop_bcascale_0_dep = tkw.partition_by_dim(
            loop_bitcast_a_scale_0, dim=M, num_partitions=2
        )
        # K=1 partition
        loop_sla_1_safe, loop_sla_1_dep = tkw.partition_by_dim(
            loop_shared_load_a_1, dim=M, num_partitions=2
        )
        loop_slascale_1_safe, loop_slascale_1_dep = tkw.partition_by_dim(
            loop_shared_load_a_scale_1, dim=M, num_partitions=2
        )
        loop_bca_1_safe, loop_bca_1_dep = tkw.partition_by_dim(
            loop_bitcast_a_1, dim=M, num_partitions=2
        )
        loop_bcascale_1_safe, loop_bcascale_1_dep = tkw.partition_by_dim(
            loop_bitcast_a_scale_1, dim=M, num_partitions=2
        )

        # =====================================================================
        # Split B / B_scale by N dimension (4 partitions):
        #   safe       = N:0,1 (p0) + N:4,5 (p2)
        #   dependent  = N:2,3 (p1) + N:6,7 (p3)
        # =====================================================================
        # K=0 partition
        slb_0_p0, slb_0_p1, slb_0_p2, slb_0_p3 = tkw.partition_by_dim(
            loop_shared_load_b_0, dim=N, num_partitions=4
        )
        slbscale_0_p0, slbscale_0_p1, slbscale_0_p2, slbscale_0_p3 = (
            tkw.partition_by_dim(loop_shared_load_b_scale_0, dim=N, num_partitions=4)
        )
        bcb_0_p0, bcb_0_p1, bcb_0_p2, bcb_0_p3 = tkw.partition_by_dim(
            loop_bitcast_b_0, dim=N, num_partitions=4
        )
        bcbscale_0_p0, bcbscale_0_p1, bcbscale_0_p2, bcbscale_0_p3 = (
            tkw.partition_by_dim(loop_bitcast_b_scale_0, dim=N, num_partitions=4)
        )
        loop_slb_0_safe = slb_0_p0 + slb_0_p2  # N:0,1,4,5
        loop_slb_0_dep = slb_0_p1 + slb_0_p3  # N:2,3,6,7
        loop_slbscale_0_safe = slbscale_0_p0 + slbscale_0_p2
        loop_slbscale_0_dep = slbscale_0_p1 + slbscale_0_p3
        loop_bcb_0_safe = bcb_0_p0 + bcb_0_p2
        loop_bcb_0_dep = bcb_0_p1 + bcb_0_p3
        loop_bcbscale_0_safe = bcbscale_0_p0 + bcbscale_0_p2
        loop_bcbscale_0_dep = bcbscale_0_p1 + bcbscale_0_p3

        # K=1 partition
        slb_1_p0, slb_1_p1, slb_1_p2, slb_1_p3 = tkw.partition_by_dim(
            loop_shared_load_b_1, dim=N, num_partitions=4
        )
        slbscale_1_p0, slbscale_1_p1, slbscale_1_p2, slbscale_1_p3 = (
            tkw.partition_by_dim(loop_shared_load_b_scale_1, dim=N, num_partitions=4)
        )
        bcb_1_p0, bcb_1_p1, bcb_1_p2, bcb_1_p3 = tkw.partition_by_dim(
            loop_bitcast_b_1, dim=N, num_partitions=4
        )
        bcbscale_1_p0, bcbscale_1_p1, bcbscale_1_p2, bcbscale_1_p3 = (
            tkw.partition_by_dim(loop_bitcast_b_scale_1, dim=N, num_partitions=4)
        )
        loop_slb_1_safe = slb_1_p0 + slb_1_p2
        loop_slb_1_dep = slb_1_p1 + slb_1_p3
        loop_slbscale_1_safe = slbscale_1_p0 + slbscale_1_p2
        loop_slbscale_1_dep = slbscale_1_p1 + slbscale_1_p3
        loop_bcb_1_safe = bcb_1_p0 + bcb_1_p2
        loop_bcb_1_dep = bcb_1_p1 + bcb_1_p3
        loop_bcbscale_1_safe = bcbscale_1_p0 + bcbscale_1_p2
        loop_bcbscale_1_dep = bcbscale_1_p1 + bcbscale_1_p3

        # =====================================================================
        # Split MFMAs:
        #   safe       = M:0,1 x N:0,1,4,5   (8 MFMAs per K-partition)
        #   dep_B      = M:0,1 x N:2,3,6,7   (8 MFMAs) -- safe A, dep B
        #   dep_A      = M:2,3 x N:0..7       (16 MFMAs) -- dep A, all B
        # =====================================================================
        # K=0
        mma_0_M01, mma_0_M23 = tkw.partition_by_dim(
            loop_scaled_mma_0, dim=M, num_partitions=2
        )
        mma_0_M01_N01, mma_0_M01_N23, mma_0_M01_N45, mma_0_M01_N67 = (
            tkw.partition_by_dim(mma_0_M01, dim=N, num_partitions=4)
        )
        loop_mma_0_safe = mma_0_M01_N01 + mma_0_M01_N45  # M:0,1 x N:0,1,4,5
        loop_mma_0_dep_B = mma_0_M01_N23 + mma_0_M01_N67  # M:0,1 x N:2,3,6,7
        loop_mma_0_dep_A = mma_0_M23  # M:2,3 x all N

        # K=1
        mma_1_M01, mma_1_M23 = tkw.partition_by_dim(
            loop_scaled_mma_1, dim=M, num_partitions=2
        )
        mma_1_M01_N01, mma_1_M01_N23, mma_1_M01_N45, mma_1_M01_N67 = (
            tkw.partition_by_dim(mma_1_M01, dim=N, num_partitions=4)
        )
        loop_mma_1_safe = mma_1_M01_N01 + mma_1_M01_N45
        loop_mma_1_dep_B = mma_1_M01_N23 + mma_1_M01_N67
        loop_mma_1_dep_A = mma_1_M23

        # Number of async gather_to_lds ops issued per loop iteration.
        # Used as the memory_counter_wait threshold placed after gather_to_lds.
        independent_global_count = len(loop_global_to_shared)

        # Build clusters
        # Cluster 0
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=independent_global_count),
            loop_sla_0_safe,
            loop_slascale_0_safe,
            loop_slb_0_safe,
            loop_slbscale_0_safe,
            loop_bca_0_safe,
            loop_bcascale_0_safe,
            loop_bcb_0_safe,
            loop_bcbscale_0_safe,
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            tkw.SchedulingBarrier([]),
        ]

        clusters = [
            tkw.cluster(cluster_0_ops),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_0_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_0_dep,
                    loop_slascale_0_dep,
                    loop_slb_0_dep,
                    loop_slbscale_0_dep,
                    loop_bca_0_dep,
                    loop_bcascale_0_dep,
                    loop_bcb_0_dep,
                    loop_bcbscale_0_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_0_dep_B,
                    loop_mma_0_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: K=1 safe loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_sla_1_safe,
                    loop_slascale_1_safe,
                    loop_slb_1_safe,
                    loop_slbscale_1_safe,
                    loop_bca_1_safe,
                    loop_bcascale_1_safe,
                    loop_bcb_1_safe,
                    loop_bcbscale_1_safe,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: K=1 safe MFMAs, dep loads/bitcasts, dep MFMAs
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_1_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_1_dep,
                    loop_slascale_1_dep,
                    loop_slb_1_dep,
                    loop_slbscale_1_dep,
                    loop_bca_1_dep,
                    loop_bcascale_1_dep,
                    loop_bcb_1_dep,
                    loop_bcbscale_1_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_1_dep_B,
                    loop_mma_1_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule(use_stagger: bool = True):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Hides the latency of the second WorkgroupBarrier by issuing a "safe"
    subset of LDS vector.loads (rows owned by this wave) before the barrier,
    then interleaving the dependent loads with compute after it.

    Safe/dependent split (per K-partition):
      - A  : M:0,1 safe  |  M:2,3 dependent
      - B  : N:0,1,4,5 safe  |  N:2,3,6,7 dependent
      - MFMAs       : M:0,1 x N:0,1,4,5 safe  |  rest dependent

    A_scale & B_scale are preshuffled and prefetched to VGPRs.

    Args:
        use_stagger: Enable wave staggering via tkw.stagger().
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K
    M = tkl.sym.M
    N = tkl.sym.N

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale - from Global to VGPR
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_b,
                    ),
                    (all_read_b_scale, all_read_a_scale),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        ) + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        # next-iteration b_scale prefetch (to be issued alongside g2s[k+1])
        loop_b_scale_prefetch = tkw.filter_nodes(
            all_read_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_a_scale_prefetch = tkw.filter_nodes(
            all_read_a_scale, subgraph=pipeline_loop.KERNEL
        )

        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition by K dimension first
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )

        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )

        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )

        # =====================================================================
        # Split A  by M dimension: safe = M:0,1 | dependent = M:2,3
        # These correspond to rows owned by the "early" wave group (wave_idx<4).
        # After memory_counter_wait(N_gather), this wave's LDS writes are done,
        # so M:0,1 rows are readable before the global workgroup barrier.
        # =====================================================================
        # K=0 partition
        loop_sla_0_safe, loop_sla_0_dep = tkw.partition_by_dim(
            loop_shared_load_a_0, dim=M, num_partitions=2
        )

        loop_bca_0_safe, loop_bca_0_dep = tkw.partition_by_dim(
            loop_bitcast_a_0, dim=M, num_partitions=2
        )

        # K=1 partition
        loop_sla_1_safe, loop_sla_1_dep = tkw.partition_by_dim(
            loop_shared_load_a_1, dim=M, num_partitions=2
        )

        loop_bca_1_safe, loop_bca_1_dep = tkw.partition_by_dim(
            loop_bitcast_a_1, dim=M, num_partitions=2
        )

        # =====================================================================
        # Split B by N dimension (4 partitions):
        #   safe       = N:0,1 (p0) + N:4,5 (p2)
        #   dependent  = N:2,3 (p1) + N:6,7 (p3)
        # =====================================================================
        # K=0 partition
        slb_0_p0, slb_0_p1, slb_0_p2, slb_0_p3 = tkw.partition_by_dim(
            loop_shared_load_b_0, dim=N, num_partitions=4
        )

        bcb_0_p0, bcb_0_p1, bcb_0_p2, bcb_0_p3 = tkw.partition_by_dim(
            loop_bitcast_b_0, dim=N, num_partitions=4
        )

        loop_slb_0_safe = slb_0_p0 + slb_0_p2  # N:0,1,4,5
        loop_slb_0_dep = slb_0_p1 + slb_0_p3  # N:2,3,6,7

        loop_bcb_0_safe = bcb_0_p0 + bcb_0_p2
        loop_bcb_0_dep = bcb_0_p1 + bcb_0_p3

        # K=1 partition
        slb_1_p0, slb_1_p1, slb_1_p2, slb_1_p3 = tkw.partition_by_dim(
            loop_shared_load_b_1, dim=N, num_partitions=4
        )

        bcb_1_p0, bcb_1_p1, bcb_1_p2, bcb_1_p3 = tkw.partition_by_dim(
            loop_bitcast_b_1, dim=N, num_partitions=4
        )

        loop_slb_1_safe = slb_1_p0 + slb_1_p2
        loop_slb_1_dep = slb_1_p1 + slb_1_p3

        loop_bcb_1_safe = bcb_1_p0 + bcb_1_p2
        loop_bcb_1_dep = bcb_1_p1 + bcb_1_p3

        # =====================================================================
        # Split MFMAs:
        #   safe       = M:0,1 x N:0,1,4,5   (8 MFMAs per K-partition)
        #   dep_B      = M:0,1 x N:2,3,6,7   (8 MFMAs) -- safe A, dep B
        #   dep_A      = M:2,3 x N:0..7       (16 MFMAs) -- dep A, all B
        # =====================================================================
        # K=0
        mma_0_M01, mma_0_M23 = tkw.partition_by_dim(
            loop_scaled_mma_0, dim=M, num_partitions=2
        )
        mma_0_M01_N01, mma_0_M01_N23, mma_0_M01_N45, mma_0_M01_N67 = (
            tkw.partition_by_dim(mma_0_M01, dim=N, num_partitions=4)
        )
        loop_mma_0_safe = mma_0_M01_N01 + mma_0_M01_N45  # M:0,1 x N:0,1,4,5
        loop_mma_0_dep_B = mma_0_M01_N23 + mma_0_M01_N67  # M:0,1 x N:2,3,6,7
        loop_mma_0_dep_A = mma_0_M23  # M:2,3 x all N

        # K=1
        mma_1_M01, mma_1_M23 = tkw.partition_by_dim(
            loop_scaled_mma_1, dim=M, num_partitions=2
        )
        mma_1_M01_N01, mma_1_M01_N23, mma_1_M01_N45, mma_1_M01_N67 = (
            tkw.partition_by_dim(mma_1_M01, dim=N, num_partitions=4)
        )
        loop_mma_1_safe = mma_1_M01_N01 + mma_1_M01_N45
        loop_mma_1_dep_B = mma_1_M01_N23 + mma_1_M01_N67
        loop_mma_1_dep_A = mma_1_M23

        # Number of async gather_to_lds ops issued per loop iteration.
        # Used as the memory_counter_wait threshold placed after gather_to_lds.
        independent_count = len(loop_global_to_shared)

        # Build clusters
        # Cluster 0
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=independent_count),
            loop_sla_0_safe,
            loop_slb_0_safe,
            loop_bca_0_safe,
            loop_bcb_0_safe,
            loop_bitcast_a_scale,
            loop_bitcast_b_scale,
            loop_a_scale_prefetch,
            loop_b_scale_prefetch,
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            tkw.SchedulingBarrier([]),
        ]

        clusters = [
            tkw.cluster(cluster_0_ops),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_0_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_0_dep,
                    loop_slb_0_dep,
                    loop_bca_0_dep,
                    loop_bcb_0_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_0_dep_B,
                    loop_mma_0_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_sla_1_safe,
                    loop_slb_1_safe,
                    loop_bca_1_safe,
                    loop_bcb_1_safe,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_1_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_1_dep,
                    loop_slb_1_dep,
                    loop_bca_1_dep,
                    loop_bcb_1_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_1_dep_B,
                    loop_mma_1_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_asymmetric_schedule(
    eliminate_epilogue: bool = False, is_bscale_shuffled: bool = False
):
    """Return an asymmetric-prefetch MXFP4 schedule for wave_compile().

    Asymmetric data paths:
      - A (data + scale): global -> LDS -> VGPRs, prefetch depth 2
        (triple-buffered in LDS).
      - B (data + scale): global -> VGPRs directly (no LDS).

    3-stage pipeline:
      Stage 0: Async global-to-LDS prefetch for A and A_scale.
      Stage 1: Global-to-VGPR loads for B and B_scale;
               LDS-to-VGPR loads for first M-partition of A.
      Stage 2: LDS-to-VGPR loads for second M-partition of A;
               bitcasts; scaled MMA accumulation.

    The main loop interleaves MMA with memory operations:
      First MMA half: interleaved with B loads and second-partition A reads.
      Second MMA half: interleaved with B_scale loads and next-iteration
                       first-partition A reads (plus G2S for the iteration
                       after next).

    When eliminate_epilogue=True the loop runs for the full K trip count
    and relies on OOB buffer loads returning zero (GFX9+ hardware guarantee)
    so that extra iterations contribute nothing to the accumulators.  This
    removes all epilogue code, reducing total code size.
    """
    M = tkl.sym.M
    K = tkl.sym.K
    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        g2s_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        s2v_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        g2s_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.GatherToLDS)
        s2v_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # partition s2v_a by M
        g2s_a_0, g2s_a_1 = tkw.partition_by_dim(g2s_a, dim=M, num_partitions=2)
        g2s_a_scale_0, g2s_a_scale_1 = tkw.partition_by_dim(g2s_a_scale, dim=M, num_partitions=2)
    
        s2v_a_0, s2v_a_1 = tkw.partition_by_dim(s2v_a, dim=M, num_partitions=2)
        s2v_a_scale_0, s2v_a_scale_1 = tkw.partition_by_dim(
            s2v_a_scale, dim=M, num_partitions=2
        )

        # Matrix B data and B scale - Global to Vector
        g2v_b = tkw.get_node_by_tag("read_b")
        g2v_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop, eliminate_epilogue=eliminate_epilogue)

        # This forces the pipeline to use double buffering
        pipeline_loop.multi_buffer_count = 2
        pipeline_loop.unroll_factor = 2

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        g2s_a_0,
                        g2s_a_scale_0,
                        g2s_a_1,
                        g2s_a_scale_1,
                    ),
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        g2v_b,
                        g2v_b_scale,
                    ),
                    (
                        s2v_a_0,
                        s2v_a_scale_0,
                    ),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        s2v_a_1,
                        s2v_a_scale_1,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # Constants derived from schedule structure
        num_m_partitions = (
            2  # we are dividing the M dimension into 2 partitions per loop iteration
        )
        num_pf_iters = (
            2  # prefetch depth of A and A_scale is 2 iterations (triple buffer)
        )

        if is_bscale_shuffled:
            b_scale_shuffling_factor = 4
        else:
            b_scale_shuffling_factor = 1

        # =====================================================================
        # Prologue: G2S_A + G2S_A_scale + G2V_B + G2V_B_scale + vmcnt(25) + s2v_a_0 + s2v_a_scale_0
        # =====================================================================
        prologue_g2s_a_0 = tkw.filter_nodes(g2s_a_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_a_1 = tkw.filter_nodes(g2s_a_1, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_a_scale_0 = tkw.filter_nodes(
            g2s_a_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_a_scale_1 = tkw.filter_nodes(
            g2s_a_scale_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_b_scale = tkw.filter_nodes(
            g2v_b_scale, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_s2v_a_scale_0 = tkw.filter_nodes(
            s2v_a_scale_0, subgraph=pipeline_loop.PROLOGUE
        )

        # A is prefetched twice in the prologue, we want to wait for just the first prefetch
        A_g2s_total = len(prologue_g2s_a_0) + len(prologue_g2s_a_scale_0) + len(prologue_g2s_a_1) + len(prologue_g2s_a_scale_1)
        A_g2s_per_iter = A_g2s_total // num_pf_iters
        B_g2v_prologue = len(prologue_g2v_b) + (
            len(prologue_g2v_b_scale) // b_scale_shuffling_factor
        )

        # Sort prologue G2S by multi_buffer target so the buffer that
        # s2v_a_0 reads from (multi_buffer_0) is issued first.  This
        # allows vmcnt(N) to skip waiting for the other buffer's ops.
        from wave_lang.kernel.ops.wave_ops import get_custom

        def _get_dst_name(n):
            """Get the destination buffer name from a G2S node."""
            c = get_custom(n)
            dst = c.dst
            if hasattr(dst, "name"):
                return dst.name
            if hasattr(dst, "__name__"):
                return dst.__name__
            return str(dst)

        def _sort_by_dst_buffer(nodes, target_first="multi_buffer_0"):
            """Sort G2S nodes: target buffer first, then the rest."""
            first = [n for n in nodes if target_first in _get_dst_name(n)]
            rest = [n for n in nodes if target_first not in _get_dst_name(n)]
            return first + rest

        sorted_g2s_a_0 = _sort_by_dst_buffer(prologue_g2s_a_0)
        sorted_g2s_a_1 = _sort_by_dst_buffer(prologue_g2s_a_1)
        sorted_g2s_a_scale_0 = _sort_by_dst_buffer(prologue_g2s_a_scale_0)
        sorted_g2s_a_scale_1 = _sort_by_dst_buffer(prologue_g2s_a_scale_1)

        # First group: all multi_buffer_0 ops (4 data + 1 scale per
        # M-partition = 10 ops for buf0), then buf1 ops.
        # s2v_a_0 reads from multi_buffer_0, so these must complete first.
        prologue_g2s_interleaved_0 = (
            sorted_g2s_a_0[:4] + [sorted_g2s_a_scale_0[0]]
            + sorted_g2s_a_1[:4] + [sorted_g2s_a_scale_1[0]]
        )
        prologue_g2s_interleaved_1 = (
            sorted_g2s_a_0[4:8] + [sorted_g2s_a_scale_0[1]]
            + sorted_g2s_a_1[4:8] + [sorted_g2s_a_scale_1[1]]
        )

        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_g2s_interleaved_0,
                    prologue_g2v_b,
                    prologue_g2v_b_scale,
                    prologue_g2s_interleaved_1,
                    tkw.MemoryCounterWaitBarrier(load=27),
                    tkw.SchedulingBarrier([]),
                    prologue_s2v_a_0,
                    prologue_s2v_a_scale_0,
                ],
            )
        ]

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_g2s_a_0 = tkw.filter_nodes(g2s_a_0, subgraph=pipeline_loop.KERNEL)
        loop_g2s_a_1 = tkw.filter_nodes(g2s_a_1, subgraph=pipeline_loop.KERNEL)
        loop_g2s_a_scale_0 = tkw.filter_nodes(g2s_a_scale_0, subgraph=pipeline_loop.KERNEL)
        loop_g2s_a_scale_1 = tkw.filter_nodes(g2s_a_scale_1, subgraph=pipeline_loop.KERNEL)

        loop_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.KERNEL)
        loop_g2v_b_scale = tkw.filter_nodes(g2v_b_scale, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_a_scale_0 = tkw.filter_nodes(
            s2v_a_scale_0, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_1 = tkw.filter_nodes(s2v_a_1, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_a_scale_1 = tkw.filter_nodes(
            s2v_a_scale_1, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)
        # Partition MFMAs and bitcasts by M for interleaving compute with
        # memory ops.  With odd M-tile counts (e.g. 7) the partitions will
        # be unequal (4+3); interleave_operations handles this via offset
        # clamping and tail flush.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=M, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=M, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=M, num_partitions=2
        )

        # Interleave MFMAs with memory ops (matching aiter f4gemm pattern).
        # Clamp start_offsets so they fit within each partition when the M
        # tile count is odd (e.g. 7 tiles split into 4+3).
        def _clamp_offsets(n, offsets):
            return [min(o, max(0, n - 1)) for o in offsets]

        # Partition 0: B-data (g2v_b) at interval 2, A-data prefetch at 4,
        # A-scale at 3, B-scale at 6 staggered from B-data.
        interleaved_mma_0 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_0,
            interleaved_ops=[
                loop_g2v_b,
                loop_shared_load_a_1,
                loop_shared_load_a_scale_1,
                loop_g2v_b_scale,
            ],
            intervals=[2, 4, 3, 6],
            start_offsets=_clamp_offsets(len(loop_scaled_mma_0), [0, 1, 1, 3]),
            start_after_groups=[[], [], [1], []],
        )

        # Partition 1: G2S and ds_reads at interval 3.
        # G2S (10 ops) at interval 3 → covers MFMAs 0,3,...,27 = 10 slots.
        # s2v_a_0+scale (10 ops) at interval 3 offset 1 → MFMAs 1,4,...,28.
        # Total: 20 ops at interval 3 interleaved → covers ~30 MFMAs,
        # leaving ~18 MFMA drain tail (down from ~24 at interval 4).
        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                loop_g2s_a_0 + [loop_g2s_a_scale_0[0]] + loop_g2s_a_1 + [loop_g2s_a_scale_1[0]],
                loop_shared_load_a_0 + loop_shared_load_a_scale_0,
            ],
            intervals=[3, 3],
            start_offsets=_clamp_offsets(len(loop_scaled_mma_1), [0, 1]),
            start_after_groups=[[], []],
        )

        loop_B_g2v_bs = len(loop_g2v_b) + (
            len(loop_g2v_b_scale) // b_scale_shuffling_factor
        )
        loop_A_s2v_bs = len(loop_g2s_a_0) + len(loop_g2s_a_scale_0) + len(loop_g2s_a_1) + len(loop_g2s_a_scale_1)
        clusters = [
            tkw.cluster(
                [
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b,
                    loop_bitcast_b_scale,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_0,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=loop_B_g2v_bs + 5, ds=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=loop_A_s2v_bs, ds=0),
                    tkw.SchedulingBarrier([]),
                ]
            ),
        ]

        if eliminate_epilogue:
            tkw.reorder_graph(pipeline_loop.PROLOGUE, prologue_clusters)
            tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        else:
            epilogue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.EPILOGUE)
            epilogue_g2v_b_scale = tkw.filter_nodes(
                g2v_b_scale, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_a_0 = tkw.filter_nodes(
                s2v_a_0, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_a_scale_0 = tkw.filter_nodes(
                s2v_a_scale_0, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_a_1 = tkw.filter_nodes(
                s2v_a_1, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_a_scale_1 = tkw.filter_nodes(
                s2v_a_scale_1, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_a = tkw.filter_nodes(
                bitcast_a, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_a_scale = tkw.filter_nodes(
                bitcast_a_scale, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_b = tkw.filter_nodes(
                bitcast_b, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_b_scale = tkw.filter_nodes(
                bitcast_b_scale, subgraph=pipeline_loop.EPILOGUE
            )

            epilogue_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.EPILOGUE)

            def split_by_iteration(nodes, key="name"):
                # TODO: Replace name-based splitting with a
                # pipeline_drain_iteration attribute (analogous to
                # unroll_iteration). expanded_dims can't be used here because
                # loop_reconstruction copies them verbatim for both drain
                # iterations.
                itr0 = []
                itr1 = []
                for node in nodes:
                    value = getattr(node, key)
                    if "1_2" in value:
                        itr0.append(node)
                    elif "2_2" in value:
                        itr1.append(node)
                    else:
                        raise ValueError(f"Unknown {key} for node: {value}")
                return itr0, itr1

            epilogue_mma_itr0, epilogue_mma_itr1 = split_by_iteration(epilogue_mma)
            epilogue_s2v_a_1_itr0, epilogue_s2v_a_1_itr1 = split_by_iteration(
                epilogue_s2v_a_1
            )
            (
                epilogue_s2v_a_scale_1_itr0,
                epilogue_s2v_a_scale_1_itr1,
            ) = split_by_iteration(epilogue_s2v_a_scale_1)
            epilogue_bitcast_a_itr0, epilogue_bitcast_a_itr1 = split_by_iteration(
                epilogue_bitcast_a
            )
            epilogue_bitcast_a_scale_itr0, epilogue_bitcast_a_scale_itr1 = (
                split_by_iteration(epilogue_bitcast_a_scale)
            )
            epilogue_bitcast_b_itr0, epilogue_bitcast_b_itr1 = split_by_iteration(
                epilogue_bitcast_b
            )
            epilogue_bitcast_b_scale_itr0, epilogue_bitcast_b_scale_itr1 = (
                split_by_iteration(epilogue_bitcast_b_scale)
            )

            epilogue_mma_itr0_0, epilogue_mma_itr0_1 = tkw.partition_by_dim(
                epilogue_mma_itr0, dim=M, num_partitions=2
            )
            epilogue_bitcast_a_itr0_0, epilogue_bitcast_a_itr0_1 = tkw.partition_by_dim(
                epilogue_bitcast_a_itr0, dim=M, num_partitions=2
            )
            epilogue_bitcast_a_scale_itr0_0, epilogue_bitcast_a_scale_itr0_1 = (
                tkw.partition_by_dim(
                    epilogue_bitcast_a_scale_itr0, dim=M, num_partitions=2
                )
            )

            epilogue_mma_itr1_0, epilogue_mma_itr1_1 = tkw.partition_by_dim(
                epilogue_mma_itr1, dim=M, num_partitions=2
            )
            epilogue_bitcast_a_itr1_0, epilogue_bitcast_a_itr1_1 = tkw.partition_by_dim(
                epilogue_bitcast_a_itr1, dim=M, num_partitions=2
            )
            epilogue_bitcast_a_scale_itr1_0, epilogue_bitcast_a_scale_itr1_1 = (
                tkw.partition_by_dim(
                    epilogue_bitcast_a_scale_itr1, dim=M, num_partitions=2
                )
            )

            epilogue_clusters_itr0 = [
                tkw.cluster(
                    [
                        epilogue_bitcast_a_itr0_0,
                        epilogue_bitcast_a_scale_itr0_0,
                        epilogue_bitcast_b_itr0,
                        epilogue_bitcast_b_scale_itr0,
                        tkw.SchedulingBarrier([]),
                        epilogue_mma_itr0_0,
                        epilogue_g2v_b,
                        epilogue_s2v_a_1_itr0,
                        epilogue_g2v_b_scale,
                        epilogue_s2v_a_scale_1_itr0,
                        epilogue_bitcast_a_itr0_1,
                        epilogue_bitcast_a_scale_itr0_1,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_mma_itr0_1,
                        tkw.SchedulingBarrier([]),
                        epilogue_s2v_a_0,
                        epilogue_s2v_a_scale_0,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_bitcast_a_itr1_0,
                        epilogue_bitcast_a_scale_itr1_0,
                        epilogue_bitcast_b_itr1,
                        epilogue_bitcast_b_scale_itr1,
                        tkw.SchedulingBarrier([]),
                        epilogue_mma_itr1_0,
                        epilogue_s2v_a_1_itr1,
                        epilogue_s2v_a_scale_1_itr1,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_bitcast_a_itr1_1,
                        epilogue_bitcast_a_scale_itr1_1,
                        epilogue_mma_itr1_1,
                    ],
                ),
            ]

            tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

        tkw.insert_at_start(
            pipeline_loop.KERNEL,
            tkw.MemoryCounterWaitBarrier(load=A_g2s_per_iter, ds=0),
        )

    return mxfp4_dbuf_schedule


def get_mxfp4_asymmetric_schedule_mirrored(
    eliminate_epilogue: bool = False, is_ascale_shuffled: bool = False
):
    """Return a mirrored asymmetric-prefetch MXFP4 schedule for wave_compile().

    Mirrored asymmetric data paths (swapped from standard asymmetric):
      - B (data + scale): global -> LDS -> VGPRs, prefetch depth 2
        (triple-buffered in LDS).
      - A (data + scale): global -> VGPRs directly (no LDS, preshuffled).

    3-stage pipeline:
      Stage 0: Async global-to-LDS prefetch for B and B_scale.
      Stage 1: Global-to-VGPR loads for A and A_scale;
               LDS-to-VGPR loads for first N-partition of B.
      Stage 2: LDS-to-VGPR loads for second N-partition of B;
               bitcasts; scaled MMA accumulation.

    The main loop interleaves MMA with memory operations:
      First MMA half: interleaved with A loads and second-partition B reads.
      Second MMA half: interleaved with A_scale loads and next-iteration
                       first-partition B reads (plus G2S for the iteration
                       after next).

    When eliminate_epilogue=True the loop runs for the full K trip count
    and relies on OOB buffer loads returning zero (GFX9+ hardware guarantee)
    so that extra iterations contribute nothing to the accumulators.  This
    removes all epilogue code, reducing total code size.
    """
    N = tkl.sym.N
    K = tkl.sym.K
    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix B data - GatherToLDS (global->shared) + Read (shared load)
        all_read_b = tkw.get_node_by_tag("read_b")
        g2s_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        g2s_b.extend(tkw.filter_nodes(all_read_b, node_type=tkw.TensorLoadToLDS))
        s2v_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        g2s_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.GatherToLDS)
        s2v_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Partition by N (B's spatial dimension, mirrored from M for A)
        g2s_b_0, g2s_b_1 = tkw.partition_by_dim(g2s_b, dim=N, num_partitions=2)
        g2s_b_scale_0, g2s_b_scale_1 = tkw.partition_by_dim(g2s_b_scale, dim=N, num_partitions=2)

        s2v_b_0, s2v_b_1 = tkw.partition_by_dim(s2v_b, dim=N, num_partitions=2)
        s2v_b_scale_0, s2v_b_scale_1 = tkw.partition_by_dim(
            s2v_b_scale, dim=N, num_partitions=2
        )

        # Matrix A data and A scale - Global to Vector (direct, preshuffled)
        g2v_a = tkw.get_node_by_tag("read_a")
        g2v_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop, eliminate_epilogue=eliminate_epilogue)

        pipeline_loop.multi_buffer_count = 2
        pipeline_loop.unroll_factor = 2

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        g2s_b_0,
                        g2s_b_scale_0,
                        g2s_b_1,
                        g2s_b_scale_1,
                    ),
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        g2v_a,
                        g2v_a_scale,
                    ),
                    (
                        s2v_b_0,
                        s2v_b_scale_0,
                    ),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        s2v_b_1,
                        s2v_b_scale_1,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        num_n_partitions = (
            2  # we are dividing the N dimension into 2 partitions per loop iteration
        )
        num_pf_iters = (
            2  # prefetch depth of B and B_scale is 2 iterations (triple buffer)
        )

        if is_ascale_shuffled:
            a_scale_shuffling_factor = 4
        else:
            a_scale_shuffling_factor = 1

        # =====================================================================
        # Prologue: G2S_B + G2S_B_scale + G2V_A + G2V_A_scale + vmcnt + s2v_b_0 + s2v_b_scale_0
        # =====================================================================
        prologue_g2s_b_0 = tkw.filter_nodes(g2s_b_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_b_1 = tkw.filter_nodes(g2s_b_1, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_b_scale_0 = tkw.filter_nodes(
            g2s_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_scale_1 = tkw.filter_nodes(
            g2s_b_scale_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_a = tkw.filter_nodes(g2v_a, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_a_scale = tkw.filter_nodes(
            g2v_a_scale, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_b_0 = tkw.filter_nodes(s2v_b_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_s2v_b_scale_0 = tkw.filter_nodes(
            s2v_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )

        # B is prefetched twice in the prologue, we want to wait for just the first prefetch
        B_g2s_total = len(prologue_g2s_b_0) + len(prologue_g2s_b_scale_0) + len(prologue_g2s_b_1) + len(prologue_g2s_b_scale_1)
        B_g2s_per_iter = B_g2s_total // num_pf_iters
        A_g2v_prologue = len(prologue_g2v_a) + (
            len(prologue_g2v_a_scale) // a_scale_shuffling_factor
        )

        # Sort prologue G2S by multi_buffer target so the buffer that
        # s2v_b_0 reads from (multi_buffer_0) is issued first.  This
        # allows vmcnt(N) to skip waiting for the other buffer's ops.
        from wave_lang.kernel.ops.wave_ops import get_custom

        def _get_dst_name(n):
            """Get the destination buffer name from a G2S node."""
            c = get_custom(n)
            dst = c.dst
            if hasattr(dst, "name"):
                return dst.name
            if hasattr(dst, "__name__"):
                return dst.__name__
            return str(dst)

        def _sort_by_dst_buffer(nodes, target_first="multi_buffer_0"):
            """Sort G2S nodes: target buffer first, then the rest."""
            first = [n for n in nodes if target_first in _get_dst_name(n)]
            rest = [n for n in nodes if target_first not in _get_dst_name(n)]
            return first + rest

        sorted_g2s_b_0 = _sort_by_dst_buffer(prologue_g2s_b_0)
        sorted_g2s_b_1 = _sort_by_dst_buffer(prologue_g2s_b_1)
        sorted_g2s_b_scale_0 = _sort_by_dst_buffer(prologue_g2s_b_scale_0)
        sorted_g2s_b_scale_1 = _sort_by_dst_buffer(prologue_g2s_b_scale_1)

        # Split data ops in half across two interleaved groups, with one
        # scale op per group per N-partition.
        # s2v_b_0 reads from multi_buffer_0, so group 0 must complete first.
        half_b_0 = len(sorted_g2s_b_0) // 2
        half_b_1 = len(sorted_g2s_b_1) // 2
        assert len(sorted_g2s_b_scale_0) % 2 == 0, (
            f"B-scale ops for partition 0 not divisible by 2: {len(sorted_g2s_b_scale_0)}"
        )
        assert len(sorted_g2s_b_scale_1) % 2 == 0, (
            f"B-scale ops for partition 1 not divisible by 2: {len(sorted_g2s_b_scale_1)}"
        )
        half_bs_0 = len(sorted_g2s_b_scale_0) // 2
        half_bs_1 = len(sorted_g2s_b_scale_1) // 2
        prologue_g2s_interleaved_0 = (
            sorted_g2s_b_0[:half_b_0] + sorted_g2s_b_scale_0[:half_bs_0]
            + sorted_g2s_b_1[:half_b_1] + sorted_g2s_b_scale_1[:half_bs_1]
        )
        prologue_g2s_interleaved_1 = (
            sorted_g2s_b_0[half_b_0:] + sorted_g2s_b_scale_0[half_bs_0:]
            + sorted_g2s_b_1[half_b_1:] + sorted_g2s_b_scale_1[half_bs_1:]
        )

        prologue_load_count = (
            len(prologue_g2s_interleaved_0)
            + len(prologue_g2v_a)
            + len(prologue_g2v_a_scale)
            + len(prologue_g2s_interleaved_1)
        ) - half_b_0 - half_bs_0
        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_g2s_interleaved_0,
                    prologue_g2v_a,
                    prologue_g2v_a_scale,
                    prologue_g2s_interleaved_1,
                    tkw.MemoryCounterWaitBarrier(load=prologue_load_count),
                    tkw.SchedulingBarrier([]),
                    prologue_s2v_b_0,
                    prologue_s2v_b_scale_0,
                ],
            )
        ]

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_g2s_b_0 = tkw.filter_nodes(g2s_b_0, subgraph=pipeline_loop.KERNEL)
        loop_g2s_b_1 = tkw.filter_nodes(g2s_b_1, subgraph=pipeline_loop.KERNEL)
        loop_g2s_b_scale_0 = tkw.filter_nodes(g2s_b_scale_0, subgraph=pipeline_loop.KERNEL)
        loop_g2s_b_scale_1 = tkw.filter_nodes(g2s_b_scale_1, subgraph=pipeline_loop.KERNEL)

        loop_g2v_a = tkw.filter_nodes(g2v_a, subgraph=pipeline_loop.KERNEL)
        loop_g2v_a_scale = tkw.filter_nodes(g2v_a_scale, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_b_0 = tkw.filter_nodes(s2v_b_0, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_b_scale_0 = tkw.filter_nodes(
            s2v_b_scale_0, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b_1 = tkw.filter_nodes(s2v_b_1, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_b_scale_1 = tkw.filter_nodes(
            s2v_b_scale_1, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)
        # Partition MFMAs and bitcasts by N for interleaving compute with
        # memory ops.  With odd N-tile counts (e.g. 7) the partitions will
        # be unequal (4+3); interleave_operations handles this via offset
        # clamping and tail flush.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=N, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=N, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=N, num_partitions=2
        )

        # Interleave MFMAs with memory ops (matching aiter f4gemm pattern).
        # Clamp start_offsets so they fit within each partition when the N
        # tile count is odd (e.g. 7 tiles split into 4+3).
        def _clamp_offsets(n, offsets):
            return [min(o, max(0, n - 1)) for o in offsets]

        # Partition 0: A-data (g2v_a) at interval 2, B-data prefetch at 4,
        # B-scale spread evenly, A-scale at 6 staggered from A-data.
        n_mma_0 = len(loop_scaled_mma_0)
        n_bscale_reads_1 = len(loop_shared_load_b_scale_1)
        bscale_interval_0 = max(1, n_mma_0 // (n_bscale_reads_1 + 1))
        interleaved_mma_0 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_0,
            interleaved_ops=[
                loop_g2v_a,
                loop_shared_load_b_1,
                loop_shared_load_b_scale_1,
                loop_g2v_a_scale,
            ],
            intervals=[2, 4, bscale_interval_0, 6],
            start_offsets=_clamp_offsets(n_mma_0, [0, 1, 2, 3]),
            start_after_groups=[[], [], [], []],
        )

        # Partition 1: G2S, B-data ds_reads, and B-scale ds_reads as
        # separate groups so B-scale reads spread evenly across MFMAs
        # instead of clustering at the tail.
        n_mma_1 = len(loop_scaled_mma_1)
        n_bscale_reads = len(loop_shared_load_b_scale_0)
        bscale_interval = max(1, n_mma_1 // (n_bscale_reads + 1))
        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                loop_g2s_b_0 + [loop_g2s_b_scale_0[0]] + loop_g2s_b_1 + [loop_g2s_b_scale_1[0]],
                loop_shared_load_b_0,
                loop_shared_load_b_scale_0,
            ],
            intervals=[3, 3, bscale_interval],
            start_offsets=_clamp_offsets(n_mma_1, [0, 1, 2]),
            start_after_groups=[[], [], []],
        )

        loop_A_g2v_bs = len(loop_g2v_a) + (
            len(loop_g2v_a_scale) // a_scale_shuffling_factor
        )
        loop_B_s2v_bs = len(loop_g2s_b_0) + len(loop_g2s_b_scale_0) + len(loop_g2s_b_1) + len(loop_g2s_b_scale_1)
        clusters = [
            tkw.cluster(
                [
                    loop_bitcast_b_0,
                    loop_bitcast_b_scale_0,
                    loop_bitcast_a,
                    loop_bitcast_a_scale,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_0,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=loop_A_g2v_bs + half_b_0 + half_bs_0, ds=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=loop_B_s2v_bs, ds=0),
                    tkw.SchedulingBarrier([]),
                ]
            ),
        ]

        if eliminate_epilogue:
            tkw.reorder_graph(pipeline_loop.PROLOGUE, prologue_clusters)
            tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        else:
            epilogue_g2v_a = tkw.filter_nodes(g2v_a, subgraph=pipeline_loop.EPILOGUE)
            epilogue_g2v_a_scale = tkw.filter_nodes(
                g2v_a_scale, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_b_0 = tkw.filter_nodes(
                s2v_b_0, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_b_scale_0 = tkw.filter_nodes(
                s2v_b_scale_0, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_b_1 = tkw.filter_nodes(
                s2v_b_1, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_s2v_b_scale_1 = tkw.filter_nodes(
                s2v_b_scale_1, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_a = tkw.filter_nodes(
                bitcast_a, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_a_scale = tkw.filter_nodes(
                bitcast_a_scale, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_b = tkw.filter_nodes(
                bitcast_b, subgraph=pipeline_loop.EPILOGUE
            )
            epilogue_bitcast_b_scale = tkw.filter_nodes(
                bitcast_b_scale, subgraph=pipeline_loop.EPILOGUE
            )

            epilogue_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.EPILOGUE)

            def split_by_iteration(nodes, key="name"):
                itr0 = []
                itr1 = []
                for node in nodes:
                    value = getattr(node, key)
                    if "1_2" in value:
                        itr0.append(node)
                    elif "2_2" in value:
                        itr1.append(node)
                    else:
                        raise ValueError(f"Unknown {key} for node: {value}")
                return itr0, itr1

            epilogue_mma_itr0, epilogue_mma_itr1 = split_by_iteration(epilogue_mma)
            epilogue_s2v_b_1_itr0, epilogue_s2v_b_1_itr1 = split_by_iteration(
                epilogue_s2v_b_1
            )
            (
                epilogue_s2v_b_scale_1_itr0,
                epilogue_s2v_b_scale_1_itr1,
            ) = split_by_iteration(epilogue_s2v_b_scale_1)
            epilogue_bitcast_a_itr0, epilogue_bitcast_a_itr1 = split_by_iteration(
                epilogue_bitcast_a
            )
            epilogue_bitcast_a_scale_itr0, epilogue_bitcast_a_scale_itr1 = (
                split_by_iteration(epilogue_bitcast_a_scale)
            )
            epilogue_bitcast_b_itr0, epilogue_bitcast_b_itr1 = split_by_iteration(
                epilogue_bitcast_b
            )
            epilogue_bitcast_b_scale_itr0, epilogue_bitcast_b_scale_itr1 = (
                split_by_iteration(epilogue_bitcast_b_scale)
            )

            epilogue_mma_itr0_0, epilogue_mma_itr0_1 = tkw.partition_by_dim(
                epilogue_mma_itr0, dim=N, num_partitions=2
            )
            epilogue_bitcast_b_itr0_0, epilogue_bitcast_b_itr0_1 = tkw.partition_by_dim(
                epilogue_bitcast_b_itr0, dim=N, num_partitions=2
            )
            epilogue_bitcast_b_scale_itr0_0, epilogue_bitcast_b_scale_itr0_1 = (
                tkw.partition_by_dim(
                    epilogue_bitcast_b_scale_itr0, dim=N, num_partitions=2
                )
            )

            epilogue_mma_itr1_0, epilogue_mma_itr1_1 = tkw.partition_by_dim(
                epilogue_mma_itr1, dim=N, num_partitions=2
            )
            epilogue_bitcast_b_itr1_0, epilogue_bitcast_b_itr1_1 = tkw.partition_by_dim(
                epilogue_bitcast_b_itr1, dim=N, num_partitions=2
            )
            epilogue_bitcast_b_scale_itr1_0, epilogue_bitcast_b_scale_itr1_1 = (
                tkw.partition_by_dim(
                    epilogue_bitcast_b_scale_itr1, dim=N, num_partitions=2
                )
            )

            epilogue_clusters_itr0 = [
                tkw.cluster(
                    [
                        epilogue_bitcast_b_itr0_0,
                        epilogue_bitcast_b_scale_itr0_0,
                        epilogue_bitcast_a_itr0,
                        epilogue_bitcast_a_scale_itr0,
                        tkw.SchedulingBarrier([]),
                        epilogue_mma_itr0_0,
                        epilogue_g2v_a,
                        epilogue_s2v_b_1_itr0,
                        epilogue_g2v_a_scale,
                        epilogue_s2v_b_scale_1_itr0,
                        epilogue_bitcast_b_itr0_1,
                        epilogue_bitcast_b_scale_itr0_1,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_mma_itr0_1,
                        tkw.SchedulingBarrier([]),
                        epilogue_s2v_b_0,
                        epilogue_s2v_b_scale_0,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_bitcast_b_itr1_0,
                        epilogue_bitcast_b_scale_itr1_0,
                        epilogue_bitcast_a_itr1,
                        epilogue_bitcast_a_scale_itr1,
                        tkw.SchedulingBarrier([]),
                        epilogue_mma_itr1_0,
                        epilogue_s2v_b_1_itr1,
                        epilogue_s2v_b_scale_1_itr1,
                    ],
                ),
                tkw.cluster(
                    [
                        epilogue_bitcast_b_itr1_1,
                        epilogue_bitcast_b_scale_itr1_1,
                        epilogue_mma_itr1_1,
                    ],
                ),
            ]

            tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

        tkw.insert_at_start(
            pipeline_loop.KERNEL,
            tkw.MemoryCounterWaitBarrier(load=B_g2s_per_iter, ds=0),
        )

    return mxfp4_dbuf_schedule


def get_mxfp4_asymmetric_schedule_mirrored_3phase_experimental(
    eliminate_epilogue: bool = False, is_ascale_shuffled: bool = False
):
    """Experimental mirrored schedule that starts from the 2-cluster baseline.

    Goal: match the AITER mirrored kernel's assembly exactly. This schedule
    keeps the mirrored 2-cluster loop body as the baseline and only reshapes
    the prologue first, so we can tune buffer-load / ds-read ordering and
    waitcnt placement before taking on loop-body correctness.

    Mirrored mapping:
      - AITER A->LDS maps to B->LDS here.
      - AITER B->VGPR maps to A->VGPR here.
      - AITER ds_read stream maps to the B LDS read stream here.
    """
    if not eliminate_epilogue:
        return get_mxfp4_asymmetric_schedule_mirrored(
            eliminate_epilogue=eliminate_epilogue,
            is_ascale_shuffled=is_ascale_shuffled,
        )

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule_experimental():
        # =====================================================================
        # Get tagged nodes from the kernel.
        # =====================================================================
        from wave_lang.kernel.ops.wave_ops import get_custom

        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix B data: 6 GatherToLDS (global→LDS) + 24 Read (LDS→VGPR).
        all_read_b = tkw.get_node_by_tag("read_b")
        g2s_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        s2v_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        g2s_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.GatherToLDS)
        s2v_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Partition by N (B's spatial dimension, mirrored from M for A).
        assert len(g2s_b) == 6, f"Expected 6 B->LDS data gathers, got {len(g2s_b)}"
        assert len(g2s_b_scale) == 2, (
            f"Expected 2 B->LDS scale gathers, got {len(g2s_b_scale)}"
        )
        g2s_b_0, g2s_b_1 = g2s_b[:3], g2s_b[3:]
        g2s_b_scale_0, g2s_b_scale_1 = g2s_b_scale[:1], g2s_b_scale[1:]
        s2v_b_0, s2v_b_1, s2v_b_2 = tkw.partition_by_dim(s2v_b, dim=N, num_partitions=3)
        s2v_b_scale_0, s2v_b_scale_1, s2v_b_scale_2 = tkw.partition_by_dim(
            s2v_b_scale, dim=N, num_partitions=3
        )

        # Matrix A data and A scale - Global to Vector (direct, preshuffled).
        # Partition by M (A's spatial dimension, as a mirror to N for B).
        g2v_a = tkw.get_node_by_tag("read_a")
        g2v_a_0, g2v_a_1 = tkw.partition_by_dim(g2v_a, dim=M, num_partitions=2)

        g2v_a_scale = tkw.get_node_by_tag("read_a_scale")
        g2v_a_scale_0, g2v_a_scale_1 = tkw.partition_by_dim(g2v_a_scale, dim=M, num_partitions=2)

        # Bitcast operations (needed alongside compute).
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering).
        # =====================================================================
        pipeline_loop = tkw.pipeline(
            k_loop, eliminate_epilogue=eliminate_epilogue
        )
        pipeline_loop.multi_buffer_count = 2
        pipeline_loop.unroll_factor = 2

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        g2s_b_0,
                        g2s_b_scale_0,
                        g2s_b_1,
                        g2s_b_scale_1,
                    ),
                    (
                        g2v_a_0,
                        g2v_a_scale_0,
                    ),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        g2v_a_1,
                        g2v_a_scale_1,
                    ),
                    (
                        s2v_b_0,
                        s2v_b_scale_0,
                    ),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (
                        s2v_b_1,
                        s2v_b_scale_1,
                        s2v_b_2,
                        s2v_b_scale_2,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        num_pf_iters = 2
        a_scale_shuffling_factor = 4 if is_ascale_shuffled else 1

        # =====================================================================
        # Prologue: explicit packetization for AITER assembly matching.
        # Target shape:
        #   4+1, 4+1, 6+2, 4+1, 4+1, tail, vmcnt(27), barrier, 4+1, 4+1
        # with mirrored roles (B->LDS, A->VGPR, B DS reads).
        # =====================================================================
        prologue_g2s_b_0 = tkw.filter_nodes(
            g2s_b_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_1 = tkw.filter_nodes(
            g2s_b_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_scale_0 = tkw.filter_nodes(
            g2s_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_scale_1 = tkw.filter_nodes(
            g2s_b_scale_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_a_0 = tkw.filter_nodes(g2v_a_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_a_scale_0 = tkw.filter_nodes(
            g2v_a_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_a_1 = tkw.filter_nodes(g2v_a_1, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_a_scale_1 = tkw.filter_nodes(
            g2v_a_scale_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_b_0 = tkw.filter_nodes(
            s2v_b_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_b_scale_0 = tkw.filter_nodes(
            s2v_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )

        B_g2s_total = (
            len(prologue_g2s_b_0)
            + len(prologue_g2s_b_scale_0)
            + len(prologue_g2s_b_1)
            + len(prologue_g2s_b_scale_1)
        )
        B_g2s_per_iter = B_g2s_total // num_pf_iters
        def _get_dst_name(n):
            c = get_custom(n)
            dst = getattr(c, "dst", None)
            if dst is None:
                dst = getattr(c, "memory", None)
            if hasattr(dst, "name"):
                return dst.name
            if hasattr(dst, "__name__"):
                return dst.__name__
            return str(dst)

        def _expanded_dim_sort_key(node, *dims):
            expanded_dims = get_custom(node).expanded_dims or {}
            return tuple(expanded_dims.get(dim, -1) for dim in dims) + (
                node.name,
            )

        def _sort_nodes_by_dims(nodes, *dims):
            return sorted(nodes, key=lambda n: _expanded_dim_sort_key(n, *dims))

        def _sort_by_dst_buffer(nodes, *dims, target_first="multi_buffer_0"):
            return sorted(
                nodes,
                key=lambda n: (
                    0 if target_first in _get_dst_name(n) else 1,
                )
                + _expanded_dim_sort_key(n, *dims),
            )

        def _append_if_any(ops, group):
            if group:
                ops.append(group)

        def _split_stage0_fill_copies(nodes):
            first = [n for n in nodes if n.name.endswith("_mapped_0_0")]
            second = [n for n in nodes if n.name.endswith("_mapped_1_0")]
            assert first and second and len(first) + len(second) == len(nodes), (
                "Expected stage-0 prologue copies mapped_0_0 and mapped_1_0, "
                f"got {[n.name for n in nodes]}"
            )
            return first, second

        (
            prologue_g2v_a_0_first,
            prologue_g2v_a_0_second,
        ) = _split_stage0_fill_copies(prologue_g2v_a_0)
        (
            prologue_g2v_a_scale_0_first,
            prologue_g2v_a_scale_0_second,
        ) = _split_stage0_fill_copies(prologue_g2v_a_scale_0)

        prologue_g2v_a_0_first = _sort_nodes_by_dims(prologue_g2v_a_0_first, M, K)
        prologue_g2v_a_scale_0_first = _sort_nodes_by_dims(
            prologue_g2v_a_scale_0_first, M, K
        )
        prologue_g2v_a_0_second = _sort_nodes_by_dims(
            prologue_g2v_a_0_second, M, K
        )
        prologue_g2v_a_scale_0_second = _sort_nodes_by_dims(
            prologue_g2v_a_scale_0_second, M, K
        )
        prologue_g2v_a_1 = _sort_nodes_by_dims(prologue_g2v_a_1, M, K)
        prologue_g2v_a_scale_1 = _sort_nodes_by_dims(
            prologue_g2v_a_scale_1, M, K
        )
        prologue_s2v_b_0 = _sort_nodes_by_dims(prologue_s2v_b_0, N, K)
        prologue_s2v_b_scale_0 = _sort_nodes_by_dims(
            prologue_s2v_b_scale_0, N, K
        )
        ds_packet_width = 4
        assert len(prologue_s2v_b_0) == 2 * ds_packet_width, (
            f"Expected 8 prologue B LDS reads, got {len(prologue_s2v_b_0)}"
        )
        assert len(prologue_s2v_b_scale_0) == 2 * ds_packet_width, (
            f"Expected 8 prologue B-scale LDS reads, got "
            f"{len(prologue_s2v_b_scale_0)}"
        )

        sorted_g2s_b_0 = _sort_by_dst_buffer(prologue_g2s_b_0, N, K)
        sorted_g2s_b_1 = _sort_by_dst_buffer(prologue_g2s_b_1, N, K)
        sorted_g2s_b_scale_0 = _sort_by_dst_buffer(
            prologue_g2s_b_scale_0, N, K
        )
        sorted_g2s_b_scale_1 = _sort_by_dst_buffer(
            prologue_g2s_b_scale_1, N, K
        )

        half_b_0 = len(sorted_g2s_b_0) // 2
        half_b_1 = len(sorted_g2s_b_1) // 2
        assert len(sorted_g2s_b_scale_0) % 2 == 0, (
            f"B-scale ops for partition 0 not divisible by 2: "
            f"{len(sorted_g2s_b_scale_0)}"
        )
        assert len(sorted_g2s_b_scale_1) % 2 == 0, (
            f"B-scale ops for partition 1 not divisible by 2: "
            f"{len(sorted_g2s_b_scale_1)}"
        )
        half_bs_0 = len(sorted_g2s_b_scale_0) // 2
        half_bs_1 = len(sorted_g2s_b_scale_1) // 2

        prologue_g2s_scale = sorted_g2s_b_scale_0[:1] + sorted_g2s_b_scale_1[:1]
        prologue_g2s_scale_tail = (
            sorted_g2s_b_scale_0[1:2] + sorted_g2s_b_scale_1[1:2]
        )
        prologue_g2s_data = sorted_g2s_b_0[:half_b_0] + sorted_g2s_b_1[:half_b_1]
        prologue_g2s_data_tail = (
            sorted_g2s_b_0[half_b_0:] + sorted_g2s_b_1[half_b_1:]
        )

        # Hardcode the packet slices for the current mirrored AITER bringup.
        # These are temporary schedule slices used only to shape the emitted
        # packet order while we compare the live gemm.rocmasm against AITER.
        # Build three explicit A->VGPR packets from the 3-stage pipeline fill.
        prologue_a_first = prologue_g2v_a_0_first + prologue_g2v_a_scale_0_first
        prologue_a_second = prologue_g2v_a_0_second + prologue_g2v_a_scale_0_second
        prologue_a_third = prologue_g2v_a_1 + prologue_g2v_a_scale_1

        # Four B-scale byte reads coalesce into one ds_read_b32, so these two
        # packets lower to 4+1 and 4+1 in the emitted assembly.
        prologue_ds_first = (
            prologue_s2v_b_0[:ds_packet_width]
            + prologue_s2v_b_scale_0[:ds_packet_width]
        )
        prologue_ds_second = (
            prologue_s2v_b_0[ds_packet_width : 2 * ds_packet_width]
            + prologue_s2v_b_scale_0[ds_packet_width : 2 * ds_packet_width]
        )

        prologue_ops = []
        _append_if_any(prologue_ops, prologue_g2s_scale)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_g2s_data)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_a_first)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_a_second)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_g2s_data_tail)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_g2s_scale_tail)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_a_third)
        prologue_ops.extend(
            [
                tkw.MemoryCounterWaitBarrier(load=27),
                tkw.SchedulingBarrier([]),
            ]
        )
        _append_if_any(prologue_ops, prologue_ds_first)
        prologue_ops.append(tkw.SchedulingBarrier([]))
        _append_if_any(prologue_ops, prologue_ds_second)

        # =====================================================================
        # KERNEL: explicit 12-cluster loop body skeleton.
        # Keep packet boundaries explicit so we can tune one cluster at a time.
        # =====================================================================
        loop_g2s_b_0 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_1 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_1, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_scale_0 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_scale_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_scale_1 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_scale_1, subgraph=pipeline_loop.KERNEL), N, K
        )

        loop_g2v_a = _sort_nodes_by_dims(
            tkw.filter_nodes(g2v_a, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_g2v_a_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(g2v_a_scale, subgraph=pipeline_loop.KERNEL), M, K
        )

        loop_s2v_b_0 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_1 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_1, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_2 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_2, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_scale_0 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_scale_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_scale_1 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_scale_1, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_scale_2 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_scale_2, subgraph=pipeline_loop.KERNEL), N, K
        )

        loop_bitcast_a = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_bitcast_a_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_a_scale, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_bitcast_b = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_bitcast_b_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_b_scale, subgraph=pipeline_loop.KERNEL), N, K
        )

        def _mma_cluster_sort_key(node):
            ed = get_custom(node).expanded_dims or {}
            m_idx = ed.get(M, 0)
            n_idx = ed.get(N, 0)
            k_idx = ed.get(K, 0)
            return (m_idx // 2, n_idx // 2, m_idx % 2, n_idx % 2, k_idx, node.name)

        loop_scaled_mma = sorted(
            tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL),
            key=_mma_cluster_sort_key,
        )

        # Partition by M-half (mirrors AITER's B-half split),
        # then re-sort each partition since partition_by_dim re-sorts by
        # the partition dimension and destroys the custom key.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=M, num_partitions=2
        )
        loop_scaled_mma_0 = sorted(loop_scaled_mma_0, key=_mma_cluster_sort_key)
        loop_scaled_mma_1 = sorted(loop_scaled_mma_1, key=_mma_cluster_sort_key)

        def _clamp_offsets(n, offsets):
            return [min(o, max(0, n - 1)) for o in offsets]

        def _chunk_ops(nodes, chunk_size, label):
            assert len(nodes) % chunk_size == 0, (
                f"{label} count {len(nodes)} is not divisible by {chunk_size}"
            )
            return [
                nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)
            ]

        def _packet_or_none(packets, index):
            return packets[index] if index < len(packets) else None

        def _interleave_cluster_body(
            mma_ops, ds_packet=None, aux_packet=None, aux_interval=4, aux_offset=1
        ):
            interleaved_ops = []
            intervals = []
            offsets = []
            if ds_packet:
                interleaved_ops.append(ds_packet)
                intervals.append(2)
                offsets.append(0)
            if aux_packet:
                interleaved_ops.append(aux_packet)
                intervals.append(aux_interval)
                offsets.append(aux_offset)
            if not interleaved_ops:
                return mma_ops
            return tkw.interleave_operations(
                base_ops=mma_ops,
                interleaved_ops=interleaved_ops,
                intervals=intervals,
                start_offsets=_clamp_offsets(len(mma_ops), offsets),
                start_after_groups=[[] for _ in interleaved_ops],
            )

        mma_clusters_0 = _chunk_ops(loop_scaled_mma_0, 8, "M-half-0 MMA")
        mma_clusters_1 = _chunk_ops(loop_scaled_mma_1, 8, "M-half-1 MMA")
        assert len(mma_clusters_0) == 6, (
            f"Expected 6 M-half-0 MMA clusters, got {len(mma_clusters_0)}"
        )
        assert len(mma_clusters_1) == 6, (
            f"Expected 6 M-half-1 MMA clusters, got {len(mma_clusters_1)}"
        )

        # B bitcasts: 6 N-pairs x 4 (2 N-tiles x 2 K-steps).
        bc_b = _chunk_ops(loop_bitcast_b, 4, "B bitcast per N-pair")
        bc_bs = _chunk_ops(loop_bitcast_b_scale, 4, "B-scale bitcast per N-pair")
        assert len(bc_b) == 6, f"Expected 6 B-bitcast N-pairs, got {len(bc_b)}"
        assert len(bc_bs) == 6, f"Expected 6 B-scale N-pairs, got {len(bc_bs)}"

        # A bitcasts: 2 M-halves x 4 (2 M-tiles x 2 K-steps).
        bc_a = _chunk_ops(loop_bitcast_a, 4, "A bitcast per M-half")
        bc_as = _chunk_ops(loop_bitcast_a_scale, 4, "A-scale bitcast per M-half")
        assert len(bc_a) == 2, f"Expected 2 A-bitcast M-halves, got {len(bc_a)}"
        assert len(bc_as) == 2, f"Expected 2 A-scale M-halves, got {len(bc_as)}"

        # DS prefetch packets: each cluster prefetches for cluster+2.
        # s2v_b_1 (N:4-7, stage 2) -> clusters 0-1 prefetch for clusters 2-3.
        ds_pf_data_01 = _chunk_ops(loop_s2v_b_1, ds_packet_width, "DS pf s2v_b_1 data")
        ds_pf_scale_01 = _chunk_ops(loop_s2v_b_scale_1, ds_packet_width, "DS pf s2v_b_1 scale")
        # s2v_b_2 (N:8-11, stage 2) -> clusters 2-3 prefetch for clusters 4-5.
        ds_pf_data_23 = _chunk_ops(loop_s2v_b_2, ds_packet_width, "DS pf s2v_b_2 data")
        ds_pf_scale_23 = _chunk_ops(loop_s2v_b_scale_2, ds_packet_width, "DS pf s2v_b_2 scale")
        # s2v_b_0 (N:0-3, stage 1) -> clusters 10-11 for next iteration.
        ds_pf_data_1011 = _chunk_ops(loop_s2v_b_0, ds_packet_width, "DS pf s2v_b_0 data")
        ds_pf_scale_1011 = _chunk_ops(loop_s2v_b_scale_0, ds_packet_width, "DS pf s2v_b_0 scale")

        loop_a_data_packets = (
            _chunk_ops(loop_g2v_a, 2, "loop A data") if loop_g2v_a else []
        )
        loop_a_scale_packets = (
            _chunk_ops(loop_g2v_a_scale, 4, "loop A scale")
            if loop_g2v_a_scale
            else []
        )

        assert len(loop_g2s_b_0) == 3, (
            f"Expected 3 loop B->LDS packets in partition 0, got {len(loop_g2s_b_0)}"
        )
        assert len(loop_g2s_b_1) == 3, (
            f"Expected 3 loop B->LDS packets in partition 1, got {len(loop_g2s_b_1)}"
        )
        assert len(loop_g2s_b_scale_0) == 1, (
            f"Expected 1 loop B-scale->LDS packet in partition 0, "
            f"got {len(loop_g2s_b_scale_0)}"
        )
        assert len(loop_g2s_b_scale_1) == 1, (
            f"Expected 1 loop B-scale->LDS packet in partition 1, "
            f"got {len(loop_g2s_b_scale_1)}"
        )
        loop_g2s_scale_packet = [loop_g2s_b_scale_0[0], loop_g2s_b_scale_1[0]]
        loop_g2s_data_packets = [
            [loop_g2s_b_0[i], loop_g2s_b_1[i]] for i in range(len(loop_g2s_b_0))
        ]

        # =================================================================
        # Cluster definitions: AITER-mirrored 2N x 2M x 2K structure.
        # Each cluster's 8 MFMAs consume one DS chunk (4+1) of B-data.
        # Clusters 0-1 consume prologue data (rotating regs).
        # Each cluster prefetches DS data for cluster+2.
        # =================================================================

        # --- First M-half (M:0-1), clusters 0-5 ---

        cluster_0 = tkw.cluster(
            [
                bc_a[0],
                bc_as[0],
                bc_b[0],
                bc_bs[0],
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=5),
                tkw.SchedulingBarrier([]),
                _interleave_cluster_body(
                    mma_clusters_0[0],
                    ds_packet=ds_pf_data_01[0],
                    aux_packet=_packet_or_none(loop_a_data_packets, 0),
                ),
                ds_pf_scale_01[0],
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=5),
            ]
        )
        cluster_1 = tkw.cluster(
            [
                bc_b[1],
                bc_bs[1],
                _interleave_cluster_body(
                    mma_clusters_0[1],
                    ds_packet=ds_pf_data_01[1],
                    aux_packet=_packet_or_none(loop_a_data_packets, 1),
                ),
                ds_pf_scale_01[1],
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=5),
            ]
        )
        cluster_2 = tkw.cluster(
            [
                bc_b[2],
                bc_bs[2],
                _interleave_cluster_body(
                    mma_clusters_0[2],
                    ds_packet=ds_pf_data_23[0],
                    aux_packet=_packet_or_none(loop_a_scale_packets, 0),
                    aux_interval=6,
                    aux_offset=2,
                ),
                ds_pf_scale_23[0],
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=5),
            ]
        )
        cluster_3 = tkw.cluster(
            [
                bc_b[3],
                bc_bs[3],
                _interleave_cluster_body(
                    mma_clusters_0[3],
                    ds_packet=ds_pf_data_23[1],
                ),
                ds_pf_scale_23[1],
                tkw.SchedulingBarrier([]),
                tkw.WorkgroupBarrier(),
                tkw.MemoryCounterWait(ds=5),
            ]
        )
        cluster_4 = tkw.cluster(
            [
                bc_b[4],
                bc_bs[4],
                _interleave_cluster_body(
                    mma_clusters_0[4],
                    aux_packet=loop_g2s_scale_packet,
                    aux_interval=4,
                    aux_offset=1,
                ),
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=0),
            ]
        )
        cluster_5 = tkw.cluster(
            [
                bc_b[5],
                bc_bs[5],
                _interleave_cluster_body(
                    mma_clusters_0[5],
                    aux_packet=loop_g2s_data_packets[0],
                    aux_interval=4,
                    aux_offset=1,
                ),
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWaitBarrier(load=18),
            ]
        )

        # --- Second M-half (M:2-3), clusters 6-11 ---

        cluster_6 = tkw.cluster(
            [
                bc_a[1],
                bc_as[1],
                _interleave_cluster_body(
                    mma_clusters_1[0],
                    aux_packet=loop_g2s_data_packets[1],
                    aux_interval=4,
                    aux_offset=1,
                ),
            ]
        )
        cluster_7 = tkw.cluster(
            [
                _interleave_cluster_body(
                    mma_clusters_1[1],
                    aux_packet=loop_g2s_data_packets[2],
                    aux_interval=4,
                    aux_offset=1,
                ),
            ]
        )
        cluster_8 = tkw.cluster(
            [
                _interleave_cluster_body(
                    mma_clusters_1[2],
                    aux_packet=_packet_or_none(loop_a_data_packets, 2),
                    aux_interval=4,
                    aux_offset=1,
                ),
            ]
        )
        cluster_9 = tkw.cluster(
            [
                _interleave_cluster_body(
                    mma_clusters_1[3],
                    aux_packet=_packet_or_none(loop_a_data_packets, 3),
                    aux_interval=4,
                    aux_offset=1,
                ),
            ]
        )
        cluster_10 = tkw.cluster(
            [
                _interleave_cluster_body(
                    mma_clusters_1[4],
                    ds_packet=ds_pf_data_1011[0],
                    aux_packet=_packet_or_none(loop_a_scale_packets, 1),
                    aux_interval=6,
                    aux_offset=2,
                ),
                ds_pf_scale_1011[0],
            ]
        )
        cluster_11 = tkw.cluster(
            [
                _interleave_cluster_body(
                    mma_clusters_1[5],
                    ds_packet=ds_pf_data_1011[1],
                ),
                ds_pf_scale_1011[1],
            ]
        )

        tkw.reorder_graph(pipeline_loop.PROLOGUE, [tkw.cluster(prologue_ops)])

        tkw.reorder_graph(
            pipeline_loop.KERNEL,
            [
                cluster_0,
                cluster_1,
                cluster_2,
                cluster_3,
                cluster_4,
                cluster_5,
                cluster_6,
                cluster_7,
                cluster_8,
                cluster_9,
                cluster_10,
                cluster_11,
            ],
        )
        tkw.unroll(pipeline_loop.KERNEL, 2)
        tkw.insert_at_start(
            pipeline_loop.KERNEL,
            tkw.MemoryCounterWaitBarrier(load=18),
        )

    return mxfp4_dbuf_schedule_experimental


def get_mxfp4_asymmetric_schedule_mirrored_3phase(
    eliminate_epilogue: bool = False, is_ascale_shuffled: bool = False
):
    """Return a 3-phase mirrored asymmetric-prefetch MXFP4 schedule.

    Matches the aiter f4gemm_bf16 kernel's 3-phase structure after mirroring:
      - AITER A->LDS becomes B->LDS here.
      - AITER B->VGPR becomes A->VGPR here.
      - AITER ds_read stream becomes the B LDS read stream here.
      - Phase A (32 MFMAs): compute + ds_reads (consume current B from LDS)
                            + A VGPR loads
      - Phase B (16 MFMAs): accumulator drain + LDS refill (B G2S, no ds_reads)
      - Phase C (48 MFMAs): compute + LDS refill + A VGPR loads + ds_read
                            prefetch

    Data paths (same as mirrored schedule):
      - B (data + scale): global -> LDS -> VGPRs, prefetch depth 2.
      - A (data + scale): global -> VGPRs directly (preshuffled).

    The key structural difference from the 2-phase mirrored schedule is
    separating the LDS read phase (Phase A) from the LDS write phase
    (Phase B), avoiding contention between ds_read and buffer_load_lds
    on the LDS bus.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_3phase_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix B data - GatherToLDS (global->shared) + Read (shared load)
        all_read_b = tkw.get_node_by_tag("read_b")
        g2s_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        s2v_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        g2s_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        s2v_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # 2-way N partition for ds_reads (same as working 2-phase)
        s2v_b_0, s2v_b_1 = tkw.partition_by_dim(
            s2v_b, dim=N, num_partitions=2
        )
        s2v_b_scale_0, s2v_b_scale_1 = tkw.partition_by_dim(
            s2v_b_scale, dim=N, num_partitions=2
        )

        # 2-way N partition for G2S (split across Phase B + Phase C Region 1)
        g2s_b_0, g2s_b_1 = tkw.partition_by_dim(
            g2s_b, dim=N, num_partitions=2
        )
        g2s_b_scale_0, g2s_b_scale_1 = tkw.partition_by_dim(
            g2s_b_scale, dim=N, num_partitions=2
        )

        # Matrix A data and A scale - Global to Vector (direct, preshuffled)
        g2v_a = tkw.get_node_by_tag("read_a")
        g2v_a_scale = tkw.get_node_by_tag("read_a_scale")

        # Bitcast operations
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(
            k_loop, eliminate_epilogue=eliminate_epilogue
        )
        pipeline_loop.multi_buffer_count = 2
        pipeline_loop.unroll_factor = 2

        with pipeline_loop as pl:
            # Stage 0: Async G2S prefetch for B and B_scale
            pl.set_stage(
                [
                    (
                        g2s_b_0,
                        g2s_b_scale_0,
                        g2s_b_1,
                        g2s_b_scale_1,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: A VGPR loads + first N-partition ds_reads
            pl.set_stage(
                [
                    (g2v_a, g2v_a_scale),
                    (s2v_b_0, s2v_b_scale_0),
                    (),
                ],
            )
            # Stage 2: second N-partition ds_reads + bitcasts + MMA
            pl.set_stage(
                [
                    (s2v_b_1, s2v_b_scale_1),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # Prologue: G2S + G2V_A + vmcnt + s_barrier + ds_reads for Phase A
        # =====================================================================
        prologue_g2s_b_0 = tkw.filter_nodes(
            g2s_b_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_1 = tkw.filter_nodes(
            g2s_b_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_scale_0 = tkw.filter_nodes(
            g2s_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2s_b_scale_1 = tkw.filter_nodes(
            g2s_b_scale_1, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_a = tkw.filter_nodes(
            g2v_a, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_a_scale = tkw.filter_nodes(
            g2v_a_scale, subgraph=pipeline_loop.PROLOGUE
        )

        # Prologue ds_reads: first N-partition (same as 2-phase)
        prologue_s2v_b_0 = tkw.filter_nodes(
            s2v_b_0, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_b_scale_0 = tkw.filter_nodes(
            s2v_b_scale_0, subgraph=pipeline_loop.PROLOGUE
        )

        # Sort prologue G2S by destination buffer so multi_buffer_0 ops
        # (read by s2v_b_0) are issued first.
        from wave_lang.kernel.ops.wave_ops import get_custom

        def _get_dst_name(n):
            c = get_custom(n)
            dst = c.dst
            if hasattr(dst, "name"):
                return dst.name
            if hasattr(dst, "__name__"):
                return dst.__name__
            return str(dst)

        def _expanded_dim_sort_key(node, *dims):
            expanded_dims = get_custom(node).expanded_dims or {}
            return tuple(expanded_dims.get(dim, -1) for dim in dims) + (node.name,)

        def _sort_nodes_by_dims(nodes, *dims):
            return sorted(nodes, key=lambda n: _expanded_dim_sort_key(n, *dims))

        def _sort_by_dst_buffer(nodes, *dims, target_first="multi_buffer_0"):
            return sorted(
                nodes,
                key=lambda n: (
                    0 if target_first in _get_dst_name(n) else 1,
                )
                + _expanded_dim_sort_key(n, *dims),
            )

        prologue_g2v_a = _sort_nodes_by_dims(prologue_g2v_a, M, K)
        prologue_g2v_a_scale = _sort_nodes_by_dims(prologue_g2v_a_scale, M, K)
        prologue_s2v_b_0 = _sort_nodes_by_dims(prologue_s2v_b_0, N, K)
        prologue_s2v_b_scale_0 = _sort_nodes_by_dims(prologue_s2v_b_scale_0, N, K)

        sorted_g2s_b_0 = _sort_by_dst_buffer(prologue_g2s_b_0, N, K)
        sorted_g2s_b_1 = _sort_by_dst_buffer(prologue_g2s_b_1, N, K)
        sorted_g2s_b_scale_0 = _sort_by_dst_buffer(prologue_g2s_b_scale_0, N, K)
        sorted_g2s_b_scale_1 = _sort_by_dst_buffer(prologue_g2s_b_scale_1, N, K)

        # Interleave G2S: buf0 ops first, then buf1.
        prologue_g2s_interleaved_0 = (
            sorted_g2s_b_0[:4]
            + [sorted_g2s_b_scale_0[0]]
            + sorted_g2s_b_1[:4]
            + [sorted_g2s_b_scale_1[0]]
        )
        prologue_g2s_interleaved_1 = (
            sorted_g2s_b_0[4:8]
            + [sorted_g2s_b_scale_0[1]]
            + sorted_g2s_b_1[4:8]
            + [sorted_g2s_b_scale_1[1]]
        )

        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_g2s_interleaved_0,
                    prologue_g2v_a,
                    prologue_g2v_a_scale,
                    prologue_g2s_interleaved_1,
                    tkw.MemoryCounterWaitBarrier(load=27),
                    tkw.SchedulingBarrier([]),
                    prologue_s2v_b_0,
                    prologue_s2v_b_scale_0,
                ],
            )
        ]

        # =====================================================================
        # KERNEL: 3-phase cluster structure (Phase A / B / C)
        # =====================================================================

        # Filter all nodes for the KERNEL subgraph
        loop_g2s_b_0 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_1 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_1, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_scale_0 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_scale_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_g2s_b_scale_1 = _sort_by_dst_buffer(
            tkw.filter_nodes(g2s_b_scale_1, subgraph=pipeline_loop.KERNEL), N, K
        )

        loop_g2v_a = _sort_nodes_by_dims(
            tkw.filter_nodes(g2v_a, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_g2v_a_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(g2v_a_scale, subgraph=pipeline_loop.KERNEL), M, K
        )

        loop_s2v_b_0 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_1 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_1, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_scale_0 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_scale_0, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_s2v_b_scale_1 = _sort_nodes_by_dims(
            tkw.filter_nodes(s2v_b_scale_1, subgraph=pipeline_loop.KERNEL), N, K
        )

        loop_bitcast_a = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_bitcast_a_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_a_scale, subgraph=pipeline_loop.KERNEL), M, K
        )
        loop_bitcast_b = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_bitcast_b_scale = _sort_nodes_by_dims(
            tkw.filter_nodes(bitcast_b_scale, subgraph=pipeline_loop.KERNEL), N, K
        )
        loop_scaled_mma = _sort_nodes_by_dims(
            tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL), M, N, K
        )

        # Use the SAME 2-way N partition as the working 2-phase schedule
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=N, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=N, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = (
            tkw.partition_by_dim(
                loop_bitcast_b_scale, dim=N, num_partitions=2
            )
        )

        def _clamp_offsets(n, offsets):
            return [min(o, max(0, n - 1)) for o in offsets]

        # ----- Split A VGPR loads: ~5 for Phase A, ~5 for Phase C -----
        half_a = len(loop_g2v_a) // 2
        phase_a_g2v = loop_g2v_a[:half_a]
        phase_c_g2v = loop_g2v_a[half_a:]
        a_scale_half = max(1, len(loop_g2v_a_scale) // 2)
        phase_a_g2v_scale = loop_g2v_a_scale[:a_scale_half]
        phase_c_g2v_scale = loop_g2v_a_scale[a_scale_half:]

        # ----- Split G2S: 4 for Phase B, 4 for Phase C -----
        # G2S depends on partition-0 MFMAs (Phase A/B). Placing G2S
        # after Phase A's 32 MFMAs satisfies the dependency.
        all_g2s = (
            loop_g2s_b_0
            + [loop_g2s_b_scale_0[0]]
            + loop_g2s_b_1
            + [loop_g2s_b_scale_1[0]]
        )
        half_g2s = len(all_g2s) // 2
        phase_b_g2s = all_g2s[:half_g2s]
        phase_c_g2s = all_g2s[half_g2s:]

        # =============================================================
        # Phase A: first 32 MFMAs of partition 0
        # Interleaved with: ds_reads (partition 1) + A VGPR loads (half)
        # Per-scale-group interleaving for tighter MFMA/memory mixing.
        # =============================================================
        phase_a_mma = loop_scaled_mma_0[:32]
        phase_b_mma = loop_scaled_mma_0[32:]

        # Redistribute ds_reads: move 1/3 of partition-0 reads from
        # Phase C to Phase A to match aiter's 20+10 distribution.
        # Partition-0 reads are Stage 1 (rotating regs) so they have
        # no intra-iteration dependencies and can be placed anywhere.
        # Pull a small mixed set of partition-0 LDS reads into Phase A.
        # Moving only the leading data reads leaves one extra scalar
        # scale read in Phase C versus AITER. A 3-data + 1-scale split
        # keeps the first half at 20 ds_reads and trims the second half.
        phase_a_extra_ds = loop_s2v_b_0[:3] + loop_s2v_b_scale_0[:1]
        phase_c_ds = loop_s2v_b_0[3:] + loop_s2v_b_scale_0[1:]

        # Build per-scale-group interleaving for tighter mixing:
        # 4 groups of 8 MFMAs, each with ~5 ds_reads + ~2 VGPR loads.
        sg_size = 8
        num_sg = len(phase_a_mma) // sg_size
        all_ds = loop_s2v_b_1 + loop_s2v_b_scale_1 + phase_a_extra_ds
        all_vgpr = phase_a_g2v + phase_a_g2v_scale
        ds_per_sg = len(all_ds) // num_sg
        ds_remainder = len(all_ds) % num_sg
        vgpr_per_sg = len(all_vgpr) // num_sg
        vgpr_remainder = len(all_vgpr) % num_sg

        phase_a_interleaved = []
        sg_groups = []
        ds_idx = 0
        vgpr_idx = 0
        for sg in range(num_sg):
            sg_mma = phase_a_mma[sg * sg_size : (sg + 1) * sg_size]
            n_ds = ds_per_sg + (1 if sg < ds_remainder else 0)
            sg_ds = all_ds[ds_idx : ds_idx + n_ds]
            ds_idx += n_ds
            n_vgpr = vgpr_per_sg + (1 if sg < vgpr_remainder else 0)
            sg_vgpr = all_vgpr[vgpr_idx : vgpr_idx + n_vgpr]
            vgpr_idx += n_vgpr

            interleave_ops = []
            interleave_intervals = []
            interleave_offsets = []
            if sg_ds:
                interleave_ops.append(sg_ds)
                interleave_intervals.append(2)
                interleave_offsets.append(0)
            if sg_vgpr:
                interleave_ops.append(sg_vgpr)
                interleave_intervals.append(
                    max(2, sg_size // max(1, n_vgpr))
                )
                interleave_offsets.append(1)

            if interleave_ops:
                interleaved_sg = tkw.interleave_operations(
                    base_ops=sg_mma,
                    interleaved_ops=interleave_ops,
                    intervals=interleave_intervals,
                    start_offsets=_clamp_offsets(
                        len(sg_mma), interleave_offsets
                    ),
                )
                phase_a_interleaved.extend(interleaved_sg)
                sg_groups.append(interleaved_sg)
            else:
                phase_a_interleaved.extend(sg_mma)
                sg_groups.append(sg_mma)

        # Build cluster_a ops list with lgkmcnt(5) waits between
        # scale groups. Each scale group is a separate list entry
        # in the cluster so MemoryCounterWait appears at the top
        # level (cluster() handles non-node items at top level only).
        cluster_a_ops = [
            loop_bitcast_b_0,
            loop_bitcast_b_scale_0,
            loop_bitcast_a,
            loop_bitcast_a_scale,
            tkw.SchedulingBarrier([]),
        ]

        # Insert lgkmcnt(5) + sched_barrier between scale groups.
        # The sched_barrier prevents LLVM from reordering across
        # group boundaries, preserving the tight MFMA/ds_read
        # interleaving.
        for sg_idx, sg_items in enumerate(sg_groups):
            if sg_idx < num_sg - 1:
                cluster_a_ops.append(tkw.MemoryCounterWait(ds=5))
            cluster_a_ops.append(sg_items)
            if sg_idx < num_sg - 1:
                cluster_a_ops.append(tkw.SchedulingBarrier([]))

        cluster_a_ops.append(tkw.MemoryCounterWait(ds=5))
        cluster_a_ops.extend([
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            tkw.MemoryCounterWait(ds=5),
        ])

        cluster_a = tkw.cluster(cluster_a_ops)

        # =============================================================
        # Phase B: last 16 MFMAs of partition 0 + 4 G2S
        # Accumulator drain + start LDS refill. NO ds_reads.
        # =============================================================
        interleaved_phase_b = tkw.interleave_operations(
            base_ops=phase_b_mma,
            interleaved_ops=[phase_b_g2s],
            intervals=[4],
            start_offsets=_clamp_offsets(len(phase_b_mma), [1]),
        )

        # Match AITER's phase boundary shape: barrier at the end of Phase A,
        # then lgkmcnt(5) at the start of Phase B before the refill work.
        cluster_b = tkw.cluster(
            [
                interleaved_phase_b,
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWait(ds=0),
                tkw.MemoryCounterWaitBarrier(load=18),
            ],
        )

        # =============================================================
        # Phase C: 48 MFMAs (partition 1)
        # Interleaved with: 4 G2S + ds_reads (partition 0) + VGPR loads
        # =============================================================
        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                phase_c_g2s,
                phase_c_ds,
                phase_c_g2v + phase_c_g2v_scale,
            ],
            intervals=[3, 3, 6],
            start_offsets=_clamp_offsets(
                len(loop_scaled_mma_1), [0, 1, 2]
            ),
            start_after_groups=[[], [], []],
        )

        cluster_1 = tkw.cluster(
            [
                loop_bitcast_b_1,
                loop_bitcast_b_scale_1,
                tkw.SchedulingBarrier([]),
                interleaved_mma_1,
                tkw.SchedulingBarrier([]),
                tkw.MemoryCounterWaitBarrier(load=18),
            ]
        )

        # =============================================================
        # Assemble and reorder
        # =============================================================
        # 3-phase: Phase A (32 MFMAs) + Phase B (16 MFMAs) + Phase C (48 MFMAs)
        clusters = [cluster_a, cluster_b, cluster_1]

        tkw.reorder_graph(pipeline_loop.PROLOGUE, prologue_clusters)
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

        tkw.insert_at_start(
            pipeline_loop.KERNEL,
            tkw.MemoryCounterWaitBarrier(load=18),
        )

    return mxfp4_dbuf_3phase_schedule