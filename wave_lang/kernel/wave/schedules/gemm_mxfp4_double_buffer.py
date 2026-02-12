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


def get_mxfp4_dbuf_hipblaslt_schedule():
    """Return a double-buffered MXFP4 schedule for wave_compile()."""
    M = tkl.sym.M

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
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        g2s_a,
                        g2s_a_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
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
            # Stage 1: Shared memory loads + bitcasts + compute
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

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_g2s_a = tkw.filter_nodes(g2s_a, subgraph=pipeline_loop.KERNEL)
        loop_g2s_a_scale = tkw.filter_nodes(g2s_a_scale, subgraph=pipeline_loop.KERNEL)

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
        # Partition by K dimension for interleaving compute with memory ops.
        # NOTE: Bitcasts MUST also be partitioned by K to match their producer
        # shared loads, otherwise reorder_graph fails with
        # "Cannot find producer(s)" because bitcasts in an earlier cluster
        # would depend on shared loads in a later cluster.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=M, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=M, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=M, num_partitions=2
        )

        # Interleave elements from loop_scaled_mma_1, loop_g2s_a, and loop_g2s_a_scale as described,
        # but any of these lists may be longer than others.
        # Always take one element from loop_scaled_mma_1,
        # then if available one from loop_g2s_a, else if available one from loop_g2s_a_scale,
        # else nothing.
        new_list = []
        i, j, k = 0, 0, 0
        n_mma = len(loop_scaled_mma_1)
        n_a = len(loop_g2s_a)
        n_a_scale = len(loop_g2s_a_scale)
        # We keep looping as long as there are any remaining elements in any list.
        while i < n_mma or j < n_a or k < n_a_scale:
            # Always add from loop_scaled_mma_1 if available
            if i < n_mma:
                new_list.append(loop_scaled_mma_1[i])
                i += 1
            # Try to add from loop_g2s_a if available
            if j < n_a:
                new_list.append(loop_g2s_a[j])
                j += 1
            # If loop_g2s_a is exhausted, add from loop_g2s_a_scale if available
            elif k < n_a_scale:
                new_list.append(loop_g2s_a_scale[k])
                k += 1
        # After this, if any elements remain in loop_g2s_a_scale, append them
        while k < n_a_scale:
            new_list.append(loop_g2s_a_scale[k])
            k += 1
        clusters = [
            # Cluster 0: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b,
                    loop_bitcast_b_scale,
                    loop_scaled_mma_0,
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_g2v_b,
                    loop_g2v_b_scale,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            # tkw.cluster(
            #     [
            #         loop_scaled_mma_1,
            #         loop_g2s_a,
            #         loop_g2s_a_scale,
            #     ],
            # ),
            tkw.cluster(new_list),
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(
                [
                    loop_shared_load_a_0,
                    loop_shared_load_a_scale_0,
                    tkw.SchedulingBarrier([]),
                ]
            ),
        ]

        # Insert shared memory barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

    return mxfp4_dbuf_schedule


def get_mxfp4_aiter_style_schedule():
    """Return an aiter-emulating MXFP4 schedule for 4-wave configs.

    This schedule targets the pipeline structure of the hand-tuned aiter
    128x128 MXFP4 kernel:
    - 2 K-partitions x 2 M-partitions = 4 MMA groups (~32 MFMAs each)
    - 8 clusters alternating shared loads with MFMA compute
    - No wave staggering (not beneficial at 4-wave occupancy)
    - High wave priority around all MFMA clusters
    - SharedMemoryBarrier at loop boundaries

    The K dimension only has 2 unique IDs (from the 2-stage pipeline), so
    we use M-dimension sub-partitioning within each K-half to create 4
    total MMA groups. This interleaves memory and compute more finely than
    the default 2-cluster schedule.
    """
    K = tkl.sym.K
    M = tkl.sym.M

    @wave_schedule.wave_schedule()
    def mxfp4_aiter_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel (same as double-buffer schedule)
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
        # Create 2-stage pipeline (double buffering, same as base schedule)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS
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
        # KERNEL: Main loop body with K x M partitioning (aiter-style)
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

        # =================================================================
        # Partition by K (2-way), then sub-partition by M (2-way) to get
        # 4 MMA groups of ~32 MFMAs each.
        #
        # The K dimension only has 2 unique IDs (from the 2-stage pipeline).
        # Within each K-half, we sub-partition by M to further interleave
        # loads and compute, reducing live register pressure at each phase.
        # =================================================================

        # First: 2-way K partition (same as base schedule)
        loop_scaled_mma_k0, loop_scaled_mma_k1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_k0, loop_shared_load_a_k1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_k0, loop_shared_load_b_k1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_k0, loop_shared_load_a_scale_k1 = (
            tkw.partition_by_dim(loop_shared_load_a_scale, dim=K, num_partitions=2)
        )
        loop_shared_load_b_scale_k0, loop_shared_load_b_scale_k1 = (
            tkw.partition_by_dim(loop_shared_load_b_scale, dim=K, num_partitions=2)
        )
        loop_bitcast_a_k0, loop_bitcast_a_k1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_k0, loop_bitcast_a_scale_k1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_k0, loop_bitcast_b_k1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_k0, loop_bitcast_b_scale_k1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        # Second: 2-way M sub-partition within each K-half for MFMAs.
        # A loads and A scale loads are partitioned by M (they depend on M).
        # B loads and B scale loads are NOT partitioned by M (shared across
        # M groups — all B data is needed for all M tile rows).
        mma_k0_m0, mma_k0_m1 = tkw.partition_by_dim(
            loop_scaled_mma_k0, dim=M, num_partitions=2
        )
        mma_k1_m0, mma_k1_m1 = tkw.partition_by_dim(
            loop_scaled_mma_k1, dim=M, num_partitions=2
        )

        # Partition A loads/bitcasts by M within each K-half
        a_load_k0_m0, a_load_k0_m1 = tkw.partition_by_dim(
            loop_shared_load_a_k0, dim=M, num_partitions=2
        )
        a_load_k1_m0, a_load_k1_m1 = tkw.partition_by_dim(
            loop_shared_load_a_k1, dim=M, num_partitions=2
        )
        a_scale_load_k0_m0, a_scale_load_k0_m1 = tkw.partition_by_dim(
            loop_shared_load_a_scale_k0, dim=M, num_partitions=2
        )
        a_scale_load_k1_m0, a_scale_load_k1_m1 = tkw.partition_by_dim(
            loop_shared_load_a_scale_k1, dim=M, num_partitions=2
        )
        bitcast_a_k0_m0, bitcast_a_k0_m1 = tkw.partition_by_dim(
            loop_bitcast_a_k0, dim=M, num_partitions=2
        )
        bitcast_a_k1_m0, bitcast_a_k1_m1 = tkw.partition_by_dim(
            loop_bitcast_a_k1, dim=M, num_partitions=2
        )
        bitcast_a_scale_k0_m0, bitcast_a_scale_k0_m1 = tkw.partition_by_dim(
            loop_bitcast_a_scale_k0, dim=M, num_partitions=2
        )
        bitcast_a_scale_k1_m0, bitcast_a_scale_k1_m1 = tkw.partition_by_dim(
            loop_bitcast_a_scale_k1, dim=M, num_partitions=2
        )

        independent_global_count = len(loop_global_to_shared)

        # =================================================================
        # Build 4 merged clusters.  Each cluster contains:
        #   1. Current-phase data loads + bitcasts (RAW dep: before MFMAs)
        #   2. SchedulingBarrier (LLVM: don't hoist MFMAs before loads)
        #   3. MFMAs for this phase
        #   4. Next-phase prefetch reads (safe: different VGPR dests)
        #   5. G2S loads for next K iteration (safe: writes other LDS bank)
        #   6. SchedulingBarrier (LLVM: boundary before next phase)
        #
        # The C++ backend ignores SchedulingBarrier (silently dropped)
        # and emits ops in source order, producing:
        #   [ds_reads] [MFMAs] [ds_reads_next] [G2S]
        # This overlaps next-phase reads and G2S with MFMA execution.
        #
        # LLVM gets freedom to reorder within each barrier-bounded
        # region while preserving the macro-level phase ordering.
        # =================================================================

        # Split G2S loads into 2 halves: first half with K0, second with K1
        g2s_all = list(loop_global_to_shared)
        g2s_half = max(1, len(g2s_all) // 2)
        g2s_h0 = g2s_all[:g2s_half]
        g2s_h1 = g2s_all[g2s_half:]

        clusters = [
            # --- Phase 0 (K0-M0): loads + G2S + MFMAs + K0-M1 prefetch ---
            # G2S loads go BEFORE MFMAs so they have time to complete before
            # the barrier at the end (vmcnt(N) must count completed G2S).
            tkw.cluster(
                [
                    # Current-phase data (must complete before MFMAs)
                    a_load_k0_m0,
                    a_scale_load_k0_m0,
                    loop_shared_load_b_k0,
                    loop_shared_load_b_scale_k0,
                    bitcast_a_k0_m0,
                    bitcast_a_scale_k0_m0,
                    loop_bitcast_b_k0,
                    loop_bitcast_b_scale_k0,
                    tkw.SchedulingBarrier([]),
                    # G2S loads (next-K prefetch, writes to other LDS bank)
                    loop_global_to_shared,
                    tkw.SchedulingBarrier([]),
                    # MFMAs
                    tkw.SetWavePrio(1),
                    mma_k0_m0,
                    tkw.SetWavePrio(0),
                    # Next-phase prefetch reads (different VGPR dests, safe)
                    a_load_k0_m1,
                    a_scale_load_k0_m1,
                    bitcast_a_k0_m1,
                    bitcast_a_scale_k0_m1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # --- Phase 1 (K0-M1): MFMAs + K1-M0 prefetch ---
            # K0-M1 data was prefetched during Phase 0 (after MFMAs).
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_k0_m1,
                    tkw.SetWavePrio(0),
                    # K1-M0 prefetch: B loads for K1 + A loads for K1-M0
                    a_load_k1_m0,
                    a_scale_load_k1_m0,
                    loop_shared_load_b_k1,
                    loop_shared_load_b_scale_k1,
                    bitcast_a_k1_m0,
                    bitcast_a_scale_k1_m0,
                    loop_bitcast_b_k1,
                    loop_bitcast_b_scale_k1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # --- Phase 2 (K1-M0): MFMAs + K1-M1 prefetch ---
            # K1-M0 data was prefetched during Phase 1 (after MFMAs).
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_k1_m0,
                    tkw.SetWavePrio(0),
                    # K1-M1 prefetch
                    a_load_k1_m1,
                    a_scale_load_k1_m1,
                    bitcast_a_k1_m1,
                    bitcast_a_scale_k1_m1,
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # --- Phase 3 (K1-M1): MFMAs only ---
            # K1-M1 data was prefetched during Phase 2 (after MFMAs).
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_k1_m1,
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

        # NO wave staggering — not beneficial at 4-wave occupancy

    return mxfp4_aiter_schedule


def get_mxfp4_aiter_faithful_schedule():
    """Return an AITER-faithful MXFP4 schedule for 4-wave configs.

    This schedule faithfully reproduces the pipeline structure documented
    in understanding_aiter_kernel.md / .tex using a true 3-stage pipeline:

    3-stage pipeline (num_stages=3, II=1):
      Stage 0 — A PREFETCH: GatherToLDS for A data + A scale     (earliest)
      Stage 1 — B PREFETCH: GatherToLDS for B data + B scale     (1 iter later)
      Stage 2 — COMPUTE:   shared loads + bitcasts + scaled_mma  (last to fill)

    The framework fills the prologue starting from stage 0, so:
      Stage 0 (A-loads) starts 2 iterations before compute
      Stage 1 (B-loads) starts 1 iteration before compute
      Stage 2 (compute) enters only at kernel steady state

    This gives 3 K-iterations in flight at steady state, matching AITER's
    "Pipeline depth = 5 stages, max 3 K-iters in flight" structure.

    Prologue fills 2 iterations:
      P.0: A-loads for iter 0             (stage 0 only)
      P.1: A-loads for iter 1 +           (stage 0)
           B-loads for iter 0             (stage 1)

    Kernel body (per iteration, iv=n):
      Stage 0: A-prefetch at iv+0         (G2S_A for iter n)
      Stage 1: B-prefetch at iv+1         (G2S_B for iter n+1)
      Stage 2: compute at iv+2            (shared loads + MMA for iter n+2)

    Epilogue drains 2 iterations:
      E.0: compute + B-loads (last B prefetch)
      E.1: compute only (final iteration)

    Within the kernel body, AITER-style fine-grained interleaving spreads
    individual load instructions between individual MFMA instructions:
      Phase 0 (K0-M0): K0-M0 data reads → MFMAs ⊕ K0-M1 prefetch ⊕ G2S_B
      Phase 1 (K0-M1): MFMAs ⊕ K1-M0 prefetch ⊕ G2S_B (cont'd)
      Phase 2 (K1-M0): MemWait → MFMAs ⊕ K1-M1 prefetch ⊕ G2S_A
      Phase 3 (K1-M1): MFMAs ⊕ G2S_A (cont'd)
    where ⊕ denotes interleaving at individual instruction granularity.

    No wave staggering (4-wave occupancy).
    """
    K = tkl.sym.K
    M = tkl.sym.M

    @wave_schedule.wave_schedule()
    def mxfp4_aiter_faithful():
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

        # Bitcast operations
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 3-stage pipeline matching AITER's pipeline depth
        #
        # The framework's prologue fill order is stage 0 first, stage N-1 last:
        #   Prologue step 0: [stage 0]
        #   Prologue step 1: [stage 0, stage 1]
        #   Kernel:          [stage 0, stage 1, stage 2]
        #
        # Therefore stage ordering must be:
        #   Stage 0 — A PREFETCH: GatherToLDS for A (starts 2 iters early)
        #   Stage 1 — B PREFETCH: GatherToLDS for B (starts 1 iter early)
        #   Stage 2 — COMPUTE:   shared loads + bitcasts + MMA (current iter)
        #
        # In the kernel body, stage i uses induction_variable + i:
        #   Stage 0 at iv+0: A-loads for current base index
        #   Stage 1 at iv+1: B-loads for next index
        #   Stage 2 at iv+2: compute for index 2 ahead
        #
        # The net effect: compute at index n uses A loaded at n-2 and B at n-1,
        # giving 3 K-iterations in flight, matching AITER exactly.
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: A prefetch (fills first in prologue)
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_a_scale),
                ],
            )
            # Stage 1: B prefetch (fills second in prologue)
            pl.set_stage(
                [
                    (global_to_shared_b, global_to_shared_b_scale),
                ],
            )
            # Stage 2: Compute (enters only at kernel steady state)
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                        shared_load_a_scale,
                        shared_load_b_scale,
                        bitcast_a,
                        bitcast_a_scale,
                        bitcast_b,
                        bitcast_b_scale,
                        scaled_mma,
                    ),
                ],
            )

        # =====================================================================
        # KERNEL: Cluster reordering for AITER-style interleaving
        #
        # After pipelining, the kernel body contains ops from all 3 stages:
        #   - Stage 0 ops: shared loads + bitcasts + MMA (iter n)
        #   - Stage 1 ops: G2S_B + G2S_B_scale (iter n+1)
        #   - Stage 2 ops: G2S_A + G2S_A_scale (iter n+2)
        #
        # We partition compute by K×M and interleave with prefetch to match:
        #   Slot 0 (AITER B0.S0): MFMA_s0 + LOAD_B[n+1]
        #   Slot 1 (AITER B0.S1): MFMA_s1 + LOAD_A[n+2]
        # =====================================================================

        # Filter KERNEL-stage nodes by original tag
        loop_g2s_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        loop_g2s_b_scale = tkw.filter_nodes(
            global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_g2s_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        loop_g2s_a_scale = tkw.filter_nodes(
            global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL
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

        # =================================================================
        # Partition compute ops by K (2-way), then M (2-way) → 4 MMA groups
        # =================================================================

        # K partition
        loop_scaled_mma_k0, loop_scaled_mma_k1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_k0, loop_shared_load_a_k1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_k0, loop_shared_load_b_k1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_k0, loop_shared_load_a_scale_k1 = (
            tkw.partition_by_dim(loop_shared_load_a_scale, dim=K, num_partitions=2)
        )
        loop_shared_load_b_scale_k0, loop_shared_load_b_scale_k1 = (
            tkw.partition_by_dim(loop_shared_load_b_scale, dim=K, num_partitions=2)
        )
        loop_bitcast_a_k0, loop_bitcast_a_k1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_k0, loop_bitcast_a_scale_k1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_k0, loop_bitcast_b_k1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_k0, loop_bitcast_b_scale_k1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        # M sub-partition within each K-half
        mma_k0_m0, mma_k0_m1 = tkw.partition_by_dim(
            loop_scaled_mma_k0, dim=M, num_partitions=2
        )
        mma_k1_m0, mma_k1_m1 = tkw.partition_by_dim(
            loop_scaled_mma_k1, dim=M, num_partitions=2
        )

        # A loads/bitcasts partitioned by M (B loads shared across M groups)
        a_load_k0_m0, a_load_k0_m1 = tkw.partition_by_dim(
            loop_shared_load_a_k0, dim=M, num_partitions=2
        )
        a_load_k1_m0, a_load_k1_m1 = tkw.partition_by_dim(
            loop_shared_load_a_k1, dim=M, num_partitions=2
        )
        a_scale_load_k0_m0, a_scale_load_k0_m1 = tkw.partition_by_dim(
            loop_shared_load_a_scale_k0, dim=M, num_partitions=2
        )
        a_scale_load_k1_m0, a_scale_load_k1_m1 = tkw.partition_by_dim(
            loop_shared_load_a_scale_k1, dim=M, num_partitions=2
        )
        bitcast_a_k0_m0, bitcast_a_k0_m1 = tkw.partition_by_dim(
            loop_bitcast_a_k0, dim=M, num_partitions=2
        )
        bitcast_a_k1_m0, bitcast_a_k1_m1 = tkw.partition_by_dim(
            loop_bitcast_a_k1, dim=M, num_partitions=2
        )
        bitcast_a_scale_k0_m0, bitcast_a_scale_k0_m1 = tkw.partition_by_dim(
            loop_bitcast_a_scale_k0, dim=M, num_partitions=2
        )
        bitcast_a_scale_k1_m0, bitcast_a_scale_k1_m1 = tkw.partition_by_dim(
            loop_bitcast_a_scale_k1, dim=M, num_partitions=2
        )

        # Combine B and A G2S ops for interleaving into slots
        g2s_b_all = loop_g2s_b + loop_g2s_b_scale  # Stage 1: B prefetch (iter n+1)
        g2s_a_all = loop_g2s_a + loop_g2s_a_scale  # Stage 2: A prefetch (iter n+2)

        # =================================================================
        # Build AITER-style interleaved clusters.
        #
        # AITER interleaves individual loads BETWEEN individual MFMAs
        # rather than issuing all loads in a block before all MFMAs.
        #
        # The pattern per MFMA phase:
        #   1. ds_reads for THIS phase's data (must complete before MFMAs)
        #   2. MFMAs interleaved with:
        #      - ds_reads for NEXT phase (prefetching next block's data)
        #      - G2S loads for next iteration (global->shared prefetch)
        #
        # G2S ops come from different pipeline stages:
        #   g2s_b_all: stage 1 ops, loading B for iter n+1
        #   g2s_a_all: stage 2 ops, loading A for iter n+2
        #
        # Phase structure (4 phases per K-iteration):
        #   Phase 0 (K0-M0): reads K0-M0 data, MFMAs + K0-M1 reads + G2S_B
        #   Phase 1 (K0-M1): MFMAs + K1-M0 reads + G2S_A first half
        #   Phase 2 (K1-M0): MFMAs + K1-M1 reads + G2S_A second half
        #   Phase 3 (K1-M1): MFMAs + remaining G2S
        # =================================================================

        # ----- Helper: spread side ops evenly among MFMAs -----
        def _spread(mma_nodes, side_ops):
            """Interleave side_ops among mma_nodes at even intervals.

            Each side_op can be a single fx.Node or a [node, ...] group.
            Returns a flat list suitable for tkw.cluster().
            """
            if not side_ops:
                return list(mma_nodes)

            result = []
            n_mma = len(mma_nodes)
            n_side = len(side_ops)
            # Compute interval: insert one side op every `step` MFMAs
            step = max(1, n_mma // n_side) if n_side > 0 else n_mma + 1
            side_idx = 0

            for i, mma in enumerate(mma_nodes):
                result.append(mma)
                if side_idx < n_side and (i + 1) % step == 0:
                    item = side_ops[side_idx]
                    if isinstance(item, (list, tuple)):
                        result.extend(item)
                    else:
                        result.append(item)
                    side_idx += 1

            # Drain any remaining side ops at the end
            while side_idx < n_side:
                item = side_ops[side_idx]
                if isinstance(item, (list, tuple)):
                    result.extend(item)
                else:
                    result.append(item)
                side_idx += 1

            return result

        def _round_robin(*groups):
            """Merge multiple lists in round-robin order."""
            merged = []
            iters = [iter(g) for g in groups if g]
            while iters:
                next_iters = []
                for it in iters:
                    try:
                        merged.append(next(it))
                    except StopIteration:
                        continue
                    else:
                        next_iters.append(it)
                iters = next_iters
            return merged

        # ----- Prepare interleaving operands -----

        # G2S loads split into halves for spreading across phases
        g2s_b_all = loop_g2s_b + loop_g2s_b_scale
        g2s_a_all = loop_g2s_a + loop_g2s_a_scale
        g2s_b_h1 = g2s_b_all[: len(g2s_b_all) // 2]
        g2s_b_h2 = g2s_b_all[len(g2s_b_all) // 2 :]
        g2s_a_h1 = g2s_a_all[: len(g2s_a_all) // 2]
        g2s_a_h2 = g2s_a_all[len(g2s_a_all) // 2 :]

        # Next-phase ds_reads to interleave during current phase MFMAs.
        # Bitcasts are paired with their reads (they don't generate
        # instructions but must follow their reads for data dependency).
        # Phase 0 prefetches: K0-M1 A-data reads (B-data shared, already loaded)
        phase0_prefetch = _round_robin(
            a_load_k0_m1, a_scale_load_k0_m1,
            bitcast_a_k0_m1, bitcast_a_scale_k0_m1,
        )

        # Phase 1 prefetches: K1-M0 data reads (A + B for second K-half)
        phase1_prefetch = _round_robin(
            a_load_k1_m0, a_scale_load_k1_m0,
            loop_shared_load_b_k1, loop_shared_load_b_scale_k1,
            bitcast_a_k1_m0, bitcast_a_scale_k1_m0,
            loop_bitcast_b_k1, loop_bitcast_b_scale_k1,
        )

        # Phase 2 prefetches: K1-M1 A-data reads
        phase2_prefetch = _round_robin(
            a_load_k1_m1, a_scale_load_k1_m1,
            bitcast_a_k1_m1, bitcast_a_scale_k1_m1,
        )

        # ----- Build interleaved clusters -----

        clusters = [
            # Phase 0 (K0-M0): Load K0-M0 data up front, then MFMAs
            # interleaved with K0-M1 prefetch reads + G2S_B first half
            tkw.cluster(
                [
                    # K0-M0 data reads (must complete before K0-M0 MFMAs)
                    a_load_k0_m0,
                    a_scale_load_k0_m0,
                    loop_shared_load_b_k0,
                    loop_shared_load_b_scale_k0,
                    bitcast_a_k0_m0,
                    bitcast_a_scale_k0_m0,
                    loop_bitcast_b_k0,
                    loop_bitcast_b_scale_k0,
                    # MFMAs interleaved with next-phase reads + G2S loads
                    *_spread(
                        mma_k0_m0,
                        _round_robin(phase0_prefetch, g2s_b_h1),
                    ),
                ],
            ),
            # Phase 1 (K0-M1): K0-M1 data was prefetched during Phase 0.
            # MFMAs interleaved with K1-M0 prefetch + G2S_B second half
            tkw.cluster(
                [
                    *_spread(
                        mma_k0_m1,
                        _round_robin(phase1_prefetch, g2s_b_h2),
                    ),
                ],
            ),
            # Phase 2 (K1-M0): K1-M0 data was prefetched during Phase 1.
            # MFMAs interleaved with K1-M1 prefetch + G2S_A first half
            # MemoryCounterWait ensures G2S loads from previous phases landed
            tkw.cluster(
                [
                    tkw.MemoryCounterWaitBarrier(load=0),
                    *_spread(
                        mma_k1_m0,
                        _round_robin(phase2_prefetch, g2s_a_h1),
                    ),
                ],
            ),
            # Phase 3 (K1-M1): K1-M1 data was prefetched during Phase 2.
            # MFMAs interleaved with remaining G2S_A loads
            tkw.cluster(
                [
                    *_spread(
                        mma_k1_m1,
                        g2s_a_h2,
                    ),
                ],
            ),
        ]

        # SharedMemoryBarrier at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # No wave staggering — not beneficial at 4-wave occupancy

    return mxfp4_aiter_faithful


def get_mxfp4_dbuf_hipblaslt_schedule():
    """Return an asymmetric MXFP4 schedule (HipBLASLt-style).

    - A data goes through LDS (GatherToLDS -> ds_read -> VGPR), triple-buffered
    - B data loaded directly from global to VGPRs (no LDS)
    - Prefetch depth of A is 2
    - Fixed/optimized waitcnts for barriers
    """
    M = tkl.sym.M

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        k_loop = tkw.get_node_by_tag("k_loop")

        all_read_a = tkw.get_node_by_tag("read_a")
        g2s_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        s2v_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        g2s_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.GatherToLDS)
        s2v_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        s2v_a_0, s2v_a_1 = tkw.partition_by_dim(s2v_a, dim=M, num_partitions=2)
        s2v_a_scale_0, s2v_a_scale_1 = tkw.partition_by_dim(
            s2v_a_scale, dim=M, num_partitions=2
        )

        # B data and B scale - Global to Vector (no LDS)
        g2v_b = tkw.get_node_by_tag("read_b")
        g2v_b_scale = tkw.get_node_by_tag("read_b_scale")

        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # 3-stage pipeline (triple buffer for A)
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (g2s_a, g2s_a_scale),
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (g2v_b, g2v_b_scale),
                    (s2v_a_0, s2v_a_scale_0),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (s2v_a_1, s2v_a_scale_1),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        num_m_partitions = 2
        num_pf_iters = 2

        # ======================= Prologue =======================
        prologue_g2s_a = tkw.filter_nodes(g2s_a, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_a_scale = tkw.filter_nodes(g2s_a_scale, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_b_scale = tkw.filter_nodes(g2v_b_scale, subgraph=pipeline_loop.PROLOGUE)
        prologue_s2v_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_s2v_a_scale_0 = tkw.filter_nodes(s2v_a_scale_0, subgraph=pipeline_loop.PROLOGUE)

        g2s_per_iter = (len(prologue_g2s_a) + len(prologue_g2s_a_scale)) // num_pf_iters
        must_complete = g2s_per_iter // num_m_partitions
        total_vmcnt = (
            len(prologue_g2s_a) + len(prologue_g2s_a_scale)
            + len(prologue_g2v_b) + len(prologue_g2v_b_scale)
        )
        prologue_vmcnt = total_vmcnt - must_complete

        prologue_clusters = [
            tkw.cluster([
                prologue_g2s_a, prologue_g2s_a_scale,
                prologue_g2v_b, prologue_g2v_b_scale,
                tkw.MemoryCounterWaitBarrier(load=prologue_vmcnt),
                prologue_s2v_a_0, prologue_s2v_a_scale_0,
            ])
        ]

        # ======================= Kernel =======================
        # Compute waitcnt values before unrolling (use pre-unroll node counts)
        loop_g2v_b_pre = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.KERNEL)
        loop_g2v_b_scale_pre = tkw.filter_nodes(g2v_b_scale, subgraph=pipeline_loop.KERNEL)
        n_b_loads = len(loop_g2v_b_pre) + len(loop_g2v_b_scale_pre)
        a_sub_block = g2s_per_iter // num_m_partitions
        total_prologue_loads = total_vmcnt
        prologue_wait_load = total_prologue_loads - a_sub_block
        loop_wait_load = prologue_wait_load - a_sub_block - n_b_loads

        tkw.insert_at_start(
            pipeline_loop.KERNEL, tkw.MemoryCounterWaitBarrier(load=loop_wait_load)
        )

        # Unroll FIRST, then reorder across iterations to reduce VGPR
        # pressure at the iteration boundary.
        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

        # Access nodes per unrolled iteration using tag + iteration.
        # Tags are preserved through unrolling; iteration 0 = original,
        # iteration 1 = unrolled copy.
        def get_iter(tag, it):
            return tkw.get_node_by_tag_and_iteration(tag, iteration=it)

        # Iter 0 nodes
        g2s_a_i0 = get_iter("read_a", 0)
        g2s_a_scale_i0 = get_iter("read_a_scale", 0)
        g2v_b_i0 = get_iter("read_b", 0)
        g2v_b_scale_i0 = get_iter("read_b_scale", 0)
        bitcast_a_i0 = get_iter("bitcast_a", 0)
        bitcast_a_scale_i0 = get_iter("bitcast_a_scale", 0)
        bitcast_b_i0 = get_iter("bitcast_b", 0)
        bitcast_b_scale_i0 = get_iter("bitcast_b_scale", 0)
        mma_i0 = get_iter("scaled_mma", 0)

        # Iter 1 nodes
        g2s_a_i1 = get_iter("read_a", 1)
        g2s_a_scale_i1 = get_iter("read_a_scale", 1)
        g2v_b_i1 = get_iter("read_b", 1)
        g2v_b_scale_i1 = get_iter("read_b_scale", 1)
        bitcast_a_i1 = get_iter("bitcast_a", 1)
        bitcast_a_scale_i1 = get_iter("bitcast_a_scale", 1)
        bitcast_b_i1 = get_iter("bitcast_b", 1)
        bitcast_b_scale_i1 = get_iter("bitcast_b_scale", 1)
        mma_i1 = get_iter("scaled_mma", 1)

        # Filter to keep only KERNEL-stage nodes (tags return all stages).
        # After unrolling, filter_nodes with subgraph still works since
        # unrolled nodes inherit pipeline_stage metadata.
        def keep_kernel(nodes):
            """Filter nodes to only those in the KERNEL pipeline stage."""
            return [n for n in nodes
                    if hasattr(n, 'meta') and n.meta.get('pipeline_stage') == pipeline_loop.KERNEL.stage]

        # The read_a tag includes both GatherToLDS (G2S) and Read (s2v) nodes.
        # Separate them by node type.
        g2s_a_i0 = tkw.filter_nodes(g2s_a_i0, node_type=tkw.GatherToLDS)
        g2s_a_scale_i0 = tkw.filter_nodes(g2s_a_scale_i0, node_type=tkw.GatherToLDS)
        s2v_a_i0 = tkw.filter_nodes(get_iter("read_a", 0), node_type=tkw.Read)
        s2v_a_scale_i0 = tkw.filter_nodes(get_iter("read_a_scale", 0), node_type=tkw.Read)

        g2s_a_i1 = tkw.filter_nodes(g2s_a_i1, node_type=tkw.GatherToLDS)
        g2s_a_scale_i1 = tkw.filter_nodes(g2s_a_scale_i1, node_type=tkw.GatherToLDS)
        s2v_a_i1 = tkw.filter_nodes(get_iter("read_a", 1), node_type=tkw.Read)
        s2v_a_scale_i1 = tkw.filter_nodes(get_iter("read_a_scale", 1), node_type=tkw.Read)

        # Partition s2v_a by M (each iter has M0 and M1 halves)
        s2v_a_i0_m0, s2v_a_i0_m1 = tkw.partition_by_dim(s2v_a_i0, dim=M, num_partitions=2)
        s2v_a_scale_i0_m0, s2v_a_scale_i0_m1 = tkw.partition_by_dim(s2v_a_scale_i0, dim=M, num_partitions=2)
        s2v_a_i1_m0, s2v_a_i1_m1 = tkw.partition_by_dim(s2v_a_i1, dim=M, num_partitions=2)
        s2v_a_scale_i1_m0, s2v_a_scale_i1_m1 = tkw.partition_by_dim(s2v_a_scale_i1, dim=M, num_partitions=2)

        # Partition MMA and bitcast_a by M
        mma_i0_m0, mma_i0_m1 = tkw.partition_by_dim(mma_i0, dim=M, num_partitions=2)
        mma_i1_m0, mma_i1_m1 = tkw.partition_by_dim(mma_i1, dim=M, num_partitions=2)
        bitcast_a_i0_m0, bitcast_a_i0_m1 = tkw.partition_by_dim(bitcast_a_i0, dim=M, num_partitions=2)
        bitcast_a_scale_i0_m0, bitcast_a_scale_i0_m1 = tkw.partition_by_dim(bitcast_a_scale_i0, dim=M, num_partitions=2)
        bitcast_a_i1_m0, bitcast_a_i1_m1 = tkw.partition_by_dim(bitcast_a_i1, dim=M, num_partitions=2)
        bitcast_a_scale_i1_m0, bitcast_a_scale_i1_m1 = tkw.partition_by_dim(bitcast_a_scale_i1, dim=M, num_partitions=2)

        # Build cross-iteration clusters.
        # Key idea: iter0 uses data already loaded (from prev iteration/prologue),
        # completes MFMAs and frees VGPRs BEFORE iter1 loads new data.
        # This reduces peak VGPR pressure at the iteration boundary.
        clusters = [
            # Iter 0: bitcasts + MMA_0, then load next-half data + MMA_1 + G2S
            tkw.cluster([
                bitcast_a_i0_m0, bitcast_a_scale_i0_m0,
                bitcast_b_i0, bitcast_b_scale_i0,
                mma_i0_m0,
                g2v_b_i0, s2v_a_i0_m1,
                g2v_b_scale_i0, s2v_a_scale_i0_m1,
                bitcast_a_i0_m1, bitcast_a_scale_i0_m1,
                tkw.SchedulingBarrier([]),
            ]),
            tkw.cluster([
                mma_i0_m1,
                g2s_a_i0, g2s_a_scale_i0,
                tkw.MemoryCounterWaitBarrier(load=n_b_loads, ds=0),
                s2v_a_i0_m0, s2v_a_scale_i0_m0,
                tkw.SchedulingBarrier([]),
            ]),
            # Iter 1: bitcasts + MMA_0, then load next-half data + MMA_1 + G2S
            tkw.cluster([
                bitcast_a_i1_m0, bitcast_a_scale_i1_m0,
                bitcast_b_i1, bitcast_b_scale_i1,
                mma_i1_m0,
                g2v_b_i1, s2v_a_i1_m1,
                g2v_b_scale_i1, s2v_a_scale_i1_m1,
                bitcast_a_i1_m1, bitcast_a_scale_i1_m1,
                tkw.SchedulingBarrier([]),
            ]),
            tkw.cluster([
                mma_i1_m1,
                g2s_a_i1, g2s_a_scale_i1,
                tkw.MemoryCounterWaitBarrier(load=n_b_loads, ds=0),
                s2v_a_i1_m0, s2v_a_scale_i1_m0,
                tkw.SchedulingBarrier([]),
            ]),
        ]

        # ======================= Epilogue =======================
        epilogue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_g2v_b_scale = tkw.filter_nodes(g2v_b_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_scale_0 = tkw.filter_nodes(s2v_a_scale_0, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_1 = tkw.filter_nodes(s2v_a_1, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_scale_1 = tkw.filter_nodes(s2v_a_scale_1, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_a_scale = tkw.filter_nodes(bitcast_a_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_bitcast_b_scale = tkw.filter_nodes(bitcast_b_scale, subgraph=pipeline_loop.EPILOGUE)
        epilogue_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.EPILOGUE)

        def split_by_iteration(nodes, key="name"):
            itr0, itr1 = [], []
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
        epilogue_s2v_a_1_itr0, epilogue_s2v_a_1_itr1 = split_by_iteration(epilogue_s2v_a_1)
        epilogue_s2v_a_scale_1_itr0, epilogue_s2v_a_scale_1_itr1 = split_by_iteration(epilogue_s2v_a_scale_1)
        epilogue_bitcast_a_itr0, epilogue_bitcast_a_itr1 = split_by_iteration(epilogue_bitcast_a)
        epilogue_bitcast_a_scale_itr0, epilogue_bitcast_a_scale_itr1 = split_by_iteration(epilogue_bitcast_a_scale)
        epilogue_bitcast_b_itr0, epilogue_bitcast_b_itr1 = split_by_iteration(epilogue_bitcast_b)
        epilogue_bitcast_b_scale_itr0, epilogue_bitcast_b_scale_itr1 = split_by_iteration(epilogue_bitcast_b_scale)

        epilogue_mma_itr0_0, epilogue_mma_itr0_1 = tkw.partition_by_dim(epilogue_mma_itr0, dim=M, num_partitions=2)
        epilogue_bitcast_a_itr0_0, epilogue_bitcast_a_itr0_1 = tkw.partition_by_dim(epilogue_bitcast_a_itr0, dim=M, num_partitions=2)
        epilogue_bitcast_a_scale_itr0_0, epilogue_bitcast_a_scale_itr0_1 = tkw.partition_by_dim(epilogue_bitcast_a_scale_itr0, dim=M, num_partitions=2)
        epilogue_mma_itr1_0, epilogue_mma_itr1_1 = tkw.partition_by_dim(epilogue_mma_itr1, dim=M, num_partitions=2)
        epilogue_bitcast_a_itr1_0, epilogue_bitcast_a_itr1_1 = tkw.partition_by_dim(epilogue_bitcast_a_itr1, dim=M, num_partitions=2)
        epilogue_bitcast_a_scale_itr1_0, epilogue_bitcast_a_scale_itr1_1 = tkw.partition_by_dim(epilogue_bitcast_a_scale_itr1, dim=M, num_partitions=2)

        epilogue_clusters_itr0 = [
            tkw.cluster([
                epilogue_bitcast_a_itr0_0, epilogue_bitcast_a_scale_itr0_0,
                epilogue_bitcast_b_itr0, epilogue_bitcast_b_scale_itr0,
                tkw.SchedulingBarrier([]),
                epilogue_mma_itr0_0,
                epilogue_g2v_b, epilogue_s2v_a_1_itr0,
                epilogue_g2v_b_scale, epilogue_s2v_a_scale_1_itr0,
                epilogue_bitcast_a_itr0_1, epilogue_bitcast_a_scale_itr0_1,
            ]),
            tkw.cluster([
                epilogue_mma_itr0_1,
                tkw.SchedulingBarrier([]),
                epilogue_s2v_a_0, epilogue_s2v_a_scale_0,
            ]),
            tkw.cluster([
                epilogue_bitcast_a_itr1_0, epilogue_bitcast_a_scale_itr1_0,
                epilogue_bitcast_b_itr1, epilogue_bitcast_b_scale_itr1,
                tkw.SchedulingBarrier([]),
                epilogue_mma_itr1_0,
                epilogue_s2v_a_1_itr1, epilogue_s2v_a_scale_1_itr1,
            ]),
            tkw.cluster([
                epilogue_bitcast_a_itr1_1, epilogue_bitcast_a_scale_itr1_1,
                epilogue_mma_itr1_1,
            ]),
        ]

        # Reorder the KERNEL with cross-iteration clusters
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Reorder the EPILOGUE (unchanged from original)
        epi_clusters = epilogue_clusters_itr0 + prologue_clusters
        tkw.reorder_graph(pipeline_loop.EPILOGUE, epi_clusters)

    return mxfp4_dbuf_schedule
