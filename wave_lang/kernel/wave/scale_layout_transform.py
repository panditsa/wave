# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Scale Layout Transformation Pass

This pass transforms scale tensor layouts for scaled_mma operations to enable
coalesced LDS access patterns.

Problem:
--------
In standard MXFP layout, scales are stored row-major [M, K/32] or [N, K/32].
When loading from LDS for WMMA, each lane needs scales from different rows,
causing scattered access with 16 separate ds_load_2addr_b32 operations per tile.

Solution:
---------
Transform scale layout in LDS to [M/16, K/32, 16] where the inner 16 scales
(one per WMMA tile row) are contiguous. This enables coalesced wide loads
(ds_load_b128) instead of scattered loads.

The transformation:
1. Identifies scale tensors feeding into scaled_mma operations
2. Marks them with a "scale_layout_transformed" flag
3. The constraint computation uses this flag to generate correct offsets

Note: This requires the input scales to be PRE-SHUFFLED on the host before
kernel launch. The shuffle transforms [M, K/32] -> [M/16, K/32, 16].
"""

from typing import Optional
import torch.fx as fx

from ...lang import sym
from ...lang.global_symbols import WMMA_TILE_SIZE, SCALE_GROUP_SIZE
from ...ops.wave_ops import (
    get_custom,
    CustomOp,
    ScaledMMA,
    Read,
    Write,
    Allocate,
    Placeholder,
)
from .utils.graph_utils import DCE


# Flag to indicate scale tensors that have been transformed
SCALE_LAYOUT_TRANSFORMED = sym.Symbol("SCALE_LAYOUT_TRANSFORMED")


def is_scale_source_for_scaled_mma(node: fx.Node) -> tuple[bool, Optional[str]]:
    """
    Check if a node is a scale source for a scaled_mma operation.
    
    Returns:
        (is_scale, scale_type) where scale_type is 'lhs' or 'rhs' or None
    """
    for user in node.users:
        custom = get_custom(user)
        if isinstance(custom, ScaledMMA):
            if user.args[1] == node:  # lhs_scale position
                return True, 'lhs'
            if user.args[3] == node:  # rhs_scale position
                return True, 'rhs'
    return False, None


def find_scale_allocations(trace) -> list[tuple[fx.Node, str]]:
    """
    Find all Allocate nodes that are used for scales in scaled_mma operations.
    
    Returns list of (allocation_node, scale_type) tuples.
    """
    scale_allocs = []
    
    def walk_graph(graph):
        for node in graph.nodes:
            custom = get_custom(node)
            
            # Check if this is an allocation used for scales
            if isinstance(custom, Allocate):
                # Walk users to see if any reads from this allocation
                # feed into scaled_mma as scales
                for user in node.users:
                    user_custom = get_custom(user)
                    if isinstance(user_custom, Read):
                        is_scale, scale_type = is_scale_source_for_scaled_mma(user)
                        if is_scale:
                            scale_allocs.append((node, scale_type))
                            break
            
            # Recurse into subgraphs
            if hasattr(custom, 'subgraph_name') and custom.subgraph_name:
                subgraph = trace.get_subgraph(custom.subgraph_name)
                if subgraph:
                    walk_graph(subgraph)
    
    walk_graph(trace.get_root_graph())
    return scale_allocs


def mark_scale_tensors(trace, constraints) -> bool:
    """
    Mark scale tensors for layout transformation.
    
    This pass identifies scale tensors feeding into scaled_mma and marks them
    so that the constraint computation can generate the correct offsets for
    the transformed layout.
    
    Returns True if any scales were marked.
    """
    scale_allocs = find_scale_allocations(trace)
    
    if not scale_allocs:
        return False
    
    for alloc_node, scale_type in scale_allocs:
        custom = get_custom(alloc_node)
        # Mark this allocation as having transformed scale layout
        if not hasattr(custom, 'scale_layout_transformed'):
            custom.scale_layout_transformed = True
            custom.scale_type = scale_type
    
    return len(scale_allocs) > 0


def transform_scale_layout(trace, constraints, options=None) -> bool:
    """
    Main entry point for scale layout transformation.
    
    This pass:
    1. Identifies scale tensors feeding into scaled_mma
    2. Marks them for transformed layout access
    3. The actual index transformation happens in constraints.py
    
    Args:
        trace: The captured trace
        constraints: List of constraints
        options: Compilation options (optional)
    
    Returns:
        True if any transformations were applied
    """
    # Check if scale layout transformation is enabled
    if options and hasattr(options, 'transform_scale_layout'):
        if not options.transform_scale_layout:
            return False
    
    return mark_scale_tensors(trace, constraints)


def get_scale_tile_size() -> int:
    """Get the WMMA tile size used for scale transformation (16 for 16x16 WMMA)."""
    return 16


def get_scale_group_size() -> int:
    """Get the number of elements per scale (32 for MXFP4)."""
    return 32
