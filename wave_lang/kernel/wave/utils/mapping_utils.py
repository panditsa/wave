# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar
from copy import deepcopy
from math import gcd, lcm
import sympy
import torch.fx as fx

from ..._support.indexing import IndexingContext
from ...lang.wave_types import IndexMapping
from .general_utils import infer_dim, get_fastest_index
from .symbol_utils import IndexExpr, IndexSequence, IndexSymbol, simplify, subs_idxc
from ....support.indexing import piecewise_aware_subs
from ...compiler.utils import strides_from_symbolic_shape

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


def get_dict_with_updated_key(
    original_dict: dict[K, V], old_key: K, new_key: K
) -> dict[K, V]:
    """
    Update a key in a dictionary while preserving the original insertion order of values.

    Creates a new dictionary identical to the original except that the specified old key
    is replaced with the new key. All values and ordering remain unchanged.
    """
    if old_key not in original_dict:
        raise KeyError(f"Old key '{old_key}' not found in dictionary")
    if new_key in original_dict and new_key != old_key:
        raise KeyError(f"New key '{new_key}' already exists in dictionary")

    # Create a new dictionary with the same order but updated key
    new_dict = {}
    for key, value in original_dict.items():
        if key == old_key:
            new_dict[new_key] = value
        else:
            new_dict[key] = value

    return new_dict


def approximate_difference(
    expr: IndexExpr, vars: list[IndexSymbol], elements_per_thread: int
) -> bool:
    """
    During the contiguity check, we take a unit step in the fastest changing
    dimension (j -> j + 1) and we compute f(j + 1) - f(j) to see if it is 1.
    In general, we will end up with expressions of the form
    g(x + eps) - g(x) where x = h(j) and eps is a rational of the form 1/q.
    We can use q to determine if the mapping is contiguous as follows

    if q is divisible by elements_per_thread (dimensions where we have not applied the unit step), or
    if eps is 1 (corresponds to the dimension where we have applied the unit step)
    then the mapping is contiguous.

    The mapping function f(j) will be non-linear in general, and so the difference
    of 1 will be transformed to different constant values based on the function.
    But, if we recover a value of 1, we can assume that the function preserves
    the difference.

    In this function we do a pre-order traversal of the expression to obtain
    the value of the constant eps.
    """
    if expr.is_number:
        return expr
    new_vars, new_exprs = sympy.cse(expr)
    new_expr = new_exprs[0] if new_vars else expr
    new_vars = [x[0] for x in new_vars] if new_vars else vars
    for arg in sympy.preorder_traversal(new_expr):
        if isinstance(arg, sympy.Add):
            if all([x in arg.args for x in new_vars]):
                constant = [x for x in arg.args if x not in new_vars][0]
                if not isinstance(constant, sympy.Rational):
                    return expr
                if constant.p != 1:
                    return expr
                if constant.q == 1:
                    return 1
                return 0 if constant.q % elements_per_thread == 0 else expr
    return expr


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def check_is_dynamic_vals_broadcasted(nodes: list[fx.Node]) -> bool:
    """
    Check if dynamic values in a list of nodes are broadcasted.

    A dynamic value is considered broadcasted if its index has size 1 in all dimensions.
    This function checks all nodes in the list and returns True only if all dynamic values
    are broadcasted (size 1 in all dims).
    """
    for node in nodes:
        index = node.index
        assert index is not None, f"Node {node} has no index"
        if any(subs_idxc(i.size) > 1 for i in index.values()):
            return False
    return True


def check_is_mapping_contiguous(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    array_shape: tuple[IndexExpr, ...],
    index: tuple[IndexExpr, ...],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Check if mapping can be lowered to contiguous vector ops instead of gathers/scatters"""
    elements_per_thread = subs_idxc(elements_per_thread)
    if elements_per_thread == 1:
        return True

    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        symbolic_dims = [infer_dim(dim_size) for dim_size in symbolic_shape]
        index_mapping = mapping.map_input_indices(symbolic_dims)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolic_shape)

    index_mapping = tuple(subs_idxc(i) for i in index_mapping)
    iters = mapping.iters

    subs = {sym: sym + int(i == len(iters) - 1) for i, sym in enumerate(iters)}
    diff = [
        approximate_difference(
            piecewise_aware_subs(index_mapping[i], subs) - index_mapping[i],
            list(iters.keys())[-1:],
            elements_per_thread,
        )
        for i in range(len(index_mapping))
    ]

    expected_diff = [0] * len(index_mapping)
    expected_diff[-1] = 1
    if expected_diff == diff:
        return True

    # If the expected pattern is not found, check if the mapping still produces
    # contiguous memory accesses by computing offsets for each element.
    #
    # The check works by:
    # 1. Computing the linear memory offset symbolically for the first element
    # 2. For each subsequent element:
    #    - Updating the fastest dimension in the index for that element
    #    - Transforming through the mapping
    #    - Computing the new memory offset
    #    - Verifying the offset increased by exactly 1
    # 3. Returns True only if all elements are sequential
    fastest_dim = list(index.keys())[get_fastest_index(index)]

    idxc = IndexingContext.current()
    strides = strides_from_symbolic_shape(idxc, array_shape, allow_mixed_shapes=True)
    new_index = transform_index_on_mapping(
        mapping, symbolic_shape, index, is_read=is_read
    )
    prev_offset = _compute_offset(
        [new_index[infer_dim(d)] for d in symbolic_shape], strides
    )
    for i in range(1, elements_per_thread):
        new_index = deepcopy(index)
        new_index[fastest_dim].start += i
        new_index = transform_index_on_mapping(
            mapping, symbolic_shape, new_index, is_read=is_read
        )
        offset = _compute_offset(
            [new_index[infer_dim(d)] for d in symbolic_shape], strides
        )

        # When the sympy expressions are complex, we fallback to the aligned-base check.
        # Because sympy evaluates to a value, we cannot use the difference check. Hence, the check fails.
        if (offset - prev_offset) != 1:
            return _check_contiguous_with_aligned_base(
                mapping,
                symbolic_shape,
                array_shape,
                index,
                elements_per_thread,
                is_read,
            )

        prev_offset = offset

    return True


def _check_contiguous_with_aligned_base(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    array_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Aligned-base contiguity check for complex floor/Mod mappings.

    Thread indices in the fastest dimension are distributed in chunks of
    elements_per_thread, so the base is always a multiple of that value.
    Encoding this alignment (base = _aligned * elements_per_thread) lets
    sympy resolve sub-expressions such as Mod(16*k + i, 16) -> i and
    floor(k/2 + i/32) → floor(k/2) that the generic check cannot simplify.
    It is independent of the tensor shape and depends on the thread distributuion.
    """
    fastest_dim = list(index.keys())[get_fastest_index(index)]
    _aligned = sympy.Symbol("_aligned", integer=True, nonnegative=True)

    idxc = IndexingContext.current()
    strides = strides_from_symbolic_shape(idxc, array_shape, allow_mixed_shapes=True)

    def _make_aligned_index(offset_val: int) -> dict[IndexExpr, IndexSequence]:
        idx = deepcopy(index)
        idx[fastest_dim].start = _aligned * elements_per_thread + offset_val
        return idx

    new_index = transform_index_on_mapping(
        mapping,
        symbolic_shape,
        _make_aligned_index(0),
        is_read=is_read,
    )
    prev_offset = _compute_offset(
        [new_index[infer_dim(d)] for d in symbolic_shape],
        strides,
    )
    for i in range(1, elements_per_thread):
        new_index = transform_index_on_mapping(
            mapping,
            symbolic_shape,
            _make_aligned_index(i),
            is_read=is_read,
        )
        offset = _compute_offset(
            [new_index[infer_dim(d)] for d in symbolic_shape],
            strides,
        )
        diff_expr = simplify(offset - prev_offset)
        if diff_expr != 1:
            return False
        prev_offset = offset

    return True


_INDUCTION_PREFIX = "$ARG"


def _extract_integer_divisors(expr: sympy.Expr) -> set[int]:
    """Collect positive integer divisors from floor and Mod nodes in *expr*."""
    divisors: set[int] = set()
    for sub in sympy.preorder_traversal(expr):
        if isinstance(sub, sympy.Mod):
            d = sub.args[1]
            if d.is_Integer and int(d) > 0:
                divisors.add(int(d))
        elif isinstance(sub, sympy.floor):
            _, denom = sub.args[0].as_numer_denom()
            if denom.is_Integer and int(denom) > 1:
                divisors.add(int(denom))
    return divisors


def _compute_probe_depth(dim_exprs: list[IndexExpr], concrete_coeff: int) -> int:
    """Compute the minimum probe depth from mapping divisors and IV coefficient.

    For each floor(expr/D) or Mod(expr, D) with integer D, the IV contribution
    has period ``D // gcd(C, D)``.  The overall period is the LCM of all
    individual periods — this is the exact probe depth needed to detect
    constant or cyclic strides.
    """
    all_divisors: set[int] = set()
    for expr in dim_exprs:
        all_divisors |= _extract_integer_divisors(expr)
    if not all_divisors:
        return 1
    C = abs(concrete_coeff) if concrete_coeff != 0 else 1
    periods = [d // gcd(C, d) for d in all_divisors]
    return lcm(*periods) if periods else 1


def compute_iv_stride_through_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    is_read: bool = True,
    mem_strides: list[IndexExpr] | None = None,
) -> dict[sympy.Symbol, IndexExpr | list[IndexExpr]] | None:
    """Compute each IV symbol's linearized stride through a mapping.

    Uses numerical probing: evaluates the linearized address at iv=0,1,...,P,
    takes consecutive differences, and detects constant or cyclic stride
    patterns.  This handles symbolic divisors (e.g. ``floor(K/32)``) that
    defeat the old ``sympy.coeff(_iv)`` approach.

    Parameters
    ----------
    mem_strides : optional physical memory strides.  When provided these are
        used for linearization instead of strides derived from
        ``symbolic_shape``.  Callers with access to the physical memory
        layout should pass these to ensure floor/Mod cancellation.

    Returns ``{iv_sym: stride}`` (constant) or ``{iv_sym: [s0, s1, ...]}``
    (repeating cycle), or ``None`` when no IV is found.
    """
    iters = mapping.iters

    iv_info: dict[sympy.Symbol, tuple[sympy.Symbol, int]] = {}
    for dim_sym, iter_sym in mapping.output_mapping.items():
        seq = index.get(dim_sym)
        if seq is None:
            continue
        start = sympy.sympify(seq.start if isinstance(seq, IndexSequence) else seq)
        for sym in start.free_symbols:
            if not str(sym).startswith(_INDUCTION_PREFIX):
                continue
            coeff = sympy.expand(start).coeff(sym)
            concrete = subs_idxc(coeff)
            if not isinstance(concrete, (int, sympy.Integer)):
                continue
            iv_info[sym] = (iter_sym, int(concrete))

    if not iv_info:
        return None

    # Resolve mapping expressions.  Use the mapping's own dimension keys
    # so that the returned expressions are ordered by the mapping's
    # input/output declaration, independent of the caller's shape.
    map_dims = (
        mapping.input_shape if is_read else mapping.output_shape
    )
    raw_exprs = (
        mapping.map_input_indices(map_dims) if is_read
        else mapping.map_output_indices(map_dims)
    )

    idxc = IndexingContext.current()
    dim_exprs = [subs_idxc(e) for e in raw_exprs]

    if mem_strides is None:
        symbolic_shape_resolved = tuple(infer_dim(d) for d in symbolic_shape)
        mem_strides = strides_from_symbolic_shape(
            idxc, symbolic_shape_resolved, allow_mixed_shapes=True
        )

    result: dict[sympy.Symbol, IndexExpr | list[IndexExpr]] = {}

    for iv_sym, (iv_iter, concrete_coeff) in iv_info.items():
        stride_or_cycle = _probe_iv_stride(
            dim_exprs, mem_strides, iters, iv_iter, concrete_coeff
        )
        if stride_or_cycle is None:
            return None
        result[iv_sym] = stride_or_cycle

    return result


def _probe_iv_stride(
    dim_exprs: list[IndexExpr],
    mem_strides: list[IndexExpr],
    iters: dict,
    iv_iter: sympy.Symbol,
    concrete_coeff: int,
) -> IndexExpr | list[IndexExpr] | None:
    """Numerically probe the linearized address to extract the IV stride.

    The probe depth is derived from the mapping's floor/Mod divisors:
    for each integer divisor D, the IV has period D/gcd(C, D) where C is the
    IV coefficient.  The overall period P = LCM of all individual periods.
    We need P probes to detect constant or cyclic strides.

    Returns a single IndexExpr (constant stride), a list (repeating cycle),
    or None when no repeating pattern is found.
    """
    probe_depth = _compute_probe_depth(dim_exprs, concrete_coeff)

    def _linearized_addr(iv_val: int) -> IndexExpr:
        subs = {
            it: (concrete_coeff * iv_val if it == iv_iter else 0)
            for it in iters.keys()
        }
        addr = sympy.Integer(0)
        for dim_expr, stride in zip(dim_exprs, mem_strides):
            dim_val = subs_idxc(dim_expr.subs(subs))
            addr += simplify(dim_val) * stride
        return simplify(subs_idxc(addr))

    addrs = [_linearized_addr(i) for i in range(probe_depth + 1)]
    diffs = [simplify(subs_idxc(addrs[i + 1] - addrs[i]))
             for i in range(probe_depth)]

    if not diffs:
        return None

    # Constant stride (most common case).
    if all(simplify(d - diffs[0]) == 0 for d in diffs):
        return diffs[0]

    # Symbolic strides may contain floor(a/d)*d + Mod(a,d) patterns that
    # simplify can't resolve.  Substitute a concrete value for remaining
    # free symbols; use a value that avoids aliasing with any divisor.
    free = set()
    for d in diffs:
        free |= d.free_symbols
    if free:
        all_divs = set()
        for expr in dim_exprs:
            all_divs |= _extract_integer_divisors(expr)
        probe_val = lcm(*all_divs) * (probe_depth + 1) if all_divs else 8192
        numeric_subs = {s: probe_val for s in free}
        try:
            numeric_diffs = [int(d.subs(numeric_subs)) for d in diffs]
        except (TypeError, ValueError):
            return None
        if len(set(numeric_diffs)) == 1:
            return sympy.Integer(numeric_diffs[0])
        # Detect shortest repeating cycle from numeric diffs.
        for cycle_len in range(2, probe_depth // 2 + 1):
            if all(numeric_diffs[i] == numeric_diffs[i % cycle_len]
                   for i in range(probe_depth)):
                return [sympy.Integer(numeric_diffs[i]) for i in range(cycle_len)]

    # Detect shortest repeating cycle (symbolic).
    for cycle_len in range(2, probe_depth // 2 + 1):
        if all(simplify(diffs[i] - diffs[i % cycle_len]) == 0
               for i in range(probe_depth)):
            return list(diffs[:cycle_len])

    return None


def transform_index_on_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    is_read: bool = True,
) -> dict[IndexExpr, IndexSequence]:
    """Transforms the index according to the specified mapping"""
    symbolic_shape = tuple(infer_dim(d) for d in symbolic_shape)
    if is_read:
        index_mapping = mapping.map_input_indices(symbolic_shape)
    else:
        index_mapping = mapping.map_output_indices(symbolic_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(piecewise_aware_subs(i, idxc.subs) for i in index_mapping)
    iters = mapping.iters
    subs = dict(
        list(zip(iters.keys(), (expr.start for expr in index.values())))
        + list(idxc.subs.items())
    )
    transformed_index = {
        key: piecewise_aware_subs(m, subs)
        for key, m in zip(symbolic_shape, index_mapping)
    }

    return transformed_index
