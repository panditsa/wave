# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    get_custom,
    Write,
    Placeholder,
    DebugLogWrite,
)
from .._support.dtype import DataType
from .._support.indexing import IndexSymbol
from typing import TypedDict

from ..wave.constraints import Constraint, DeviceConstraint

def print_all_nodes(trace: CapturedTrace) -> None:
    """Print all nodes across all graphs using the walk method."""
    
    all_nodes = trace.walk()  # No filter = all nodes
    
    print(f"Total nodes across all graphs: {len(all_nodes)}")
    print("\n=== ALL NODES ===")
    
    for i, node in enumerate(all_nodes):
        print(f"{i}: {node.name} (op: {node.op}, target: {node.target})")
        if node.type:
            print(f"    Type: {node.type}")
        if node.meta:
            print(f"    Meta: {node.meta}")
        print()

def _substitute_recursive(obj, subs_dict):
    if isinstance(obj, (list, tuple)):
        new_obj = [_substitute_recursive(item, subs_dict) for item in obj]
        return tuple(new_obj) if isinstance(obj, tuple) else new_obj
    elif isinstance(obj, dict):
        return {key: _substitute_recursive(value, subs_dict) for key, value in obj.items()}
    elif hasattr(obj, 'symbolic_shape'):
        obj.symbolic_shape = _substitute_recursive(obj.symbolic_shape, subs_dict)
        return obj
    else:
        # Check if obj is in subs_dict only if it's hashable
        try:
            if obj in subs_dict:
                return subs_dict[obj]
        except TypeError:
            # obj is not hashable, skip this check
            pass
        
    return obj

def reshape_per_device(trace: CapturedTrace, constraints: list[Constraint]) -> None: 
    """
    Reshape the input and output tensors to match the device constraints
    """
    device_constraints = [x for x in constraints if isinstance(x, DeviceConstraint)]
    symbol_map = {}
    for constraint in device_constraints:
        symbol_map[constraint.dim] = constraint.tile_size
    
    for node in trace.walk():
        node.args = _substitute_recursive(node.args, symbol_map)
        node.kwargs = _substitute_recursive(node.kwargs, symbol_map)
        
        if node.type:
            _substitute_recursive(node.type, symbol_map)
        
        if node.meta:
            node.meta = _substitute_recursive(node.meta, symbol_map)

    #print_all_nodes(trace)     
    return