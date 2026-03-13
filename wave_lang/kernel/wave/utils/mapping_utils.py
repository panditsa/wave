# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar
from copy import deepcopy
from math import gcd, lcm
import sympy
import torch.fx as fx

from collections.abc import Sequence

from ..._support.indexing import IndexingContext
from ...lang.wave_types import IndexMapping
from ..assumptions import get_divisibility_subs
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


def _expand_mod(expr: sympy.Expr) -> sympy.Expr:
    """Rewrite ``Mod(x, d)`` as ``x - d*floor(x/d)``.

    This lets ``floor(x/d)*d + Mod(x, d)`` cancel to ``x`` via normal
    algebraic simplification, which SymPy cannot do when ``Mod`` is
    kept as an opaque node.
    """
    if not expr.has(sympy.Mod):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sympy.Mod),
        lambda e: e.args[0] - e.args[1] * sympy.floor(e.args[0] / e.args[1]),
    )


def _infer_floor_to_exact(mem_strides: list[IndexExpr]) -> dict:
    """Infer ``floor(sym/n) → sym/n`` substitutions from memory strides.

    When a memory stride has the form ``sym/n`` (symbolic numerator over
    a positive integer denominator), the stride must be a non-negative
    integer for the layout to be physically valid.  That forces the
    numerator to be a multiple of the denominator, so
    ``floor(sym/n) == sym/n``.

    Substituting this into dimension expressions collapses
    ``floor(E / floor(sym/n)) * (sym/n) + Mod(E, floor(sym/n))`` back
    into ``E`` via the standard floor/Mod identity, eliminating the
    symbolic divisor that the probing cannot handle.
    """
    subs_map: dict[sympy.Expr, sympy.Expr] = {}
    for stride in mem_strides:
        stride = sympy.sympify(stride)
        numer, denom = stride.as_numer_denom()
        if denom.is_Integer and int(denom) > 1 and numer.free_symbols:
            exact_quot = numer / denom
            subs_map[sympy.floor(exact_quot)] = exact_quot
    return subs_map


def _extract_integer_divisors(expr: sympy.Expr) -> set[int]:
    """Collect positive integer divisors from floor and Mod nodes in *expr*."""
    divisors: set[int] = set()
    for sub in sympy.preorder_traversal(expr):
        if isinstance(sub, sympy.Mod):
            d = sub.args[1]
            if d.is_Integer and int(d) > 0:
                divisors.add(int(d))
                print(
                    f"  [divisor FOUND]  Mod({sub.args[0]}, {d})"
                    f"  -> integer divisor {int(d)}"
                )
            else:
                print(
                    f"  [divisor SKIPPED] Mod({sub.args[0]}, {d})"
                    f"  -> divisor {d} is symbolic"
                    f" (type={type(d).__name__},"
                    f" is_Integer={d.is_Integer},"
                    f" free={d.free_symbols})"
                )
        elif isinstance(sub, sympy.floor):
            inner = sub.args[0]
            _, denom = inner.as_numer_denom()
            if denom.is_Integer and int(denom) > 1:
                divisors.add(int(denom))
                print(
                    f"  [divisor FOUND]  floor({inner})"
                    f"  -> integer divisor {int(denom)}"
                )
            elif denom != 1:
                print(
                    f"  [divisor SKIPPED] floor({inner})"
                    f"  -> denom {denom} is symbolic"
                    f" (type={type(denom).__name__},"
                    f" is_Integer={denom.is_Integer},"
                    f" free={denom.free_symbols})"
                )
    return divisors


def _compute_probe_depth(dim_exprs: list[IndexExpr], concrete_coeff: int) -> int:
    """Compute the minimum probe depth from mapping divisors and IV coefficient.

    For each floor(expr/D) or Mod(expr, D) with integer D, the IV contribution
    has period ``D // gcd(C, D)``.  The overall period is the LCM of all
    individual periods — this is the exact probe depth needed to detect
    constant or cyclic strides.
    """
    print(f"_compute_probe_depth  coeff(C)={concrete_coeff}  exprs:")
    for i, expr in enumerate(dim_exprs):
        print(f"  dim_expr[{i}] = {expr}  (free={getattr(expr, 'free_symbols', set())})")
    all_divisors: set[int] = set()
    for expr in dim_exprs:
        all_divisors |= _extract_integer_divisors(expr)
    if not all_divisors:
        print(
            "  *** NO integer divisors found — probe_depth defaults to 1. "
            "If floor/Mod with symbolic divisors were skipped above, this "
            "is a single-sample estimate, NOT a proven stride."
        )
        return 1
    C = abs(concrete_coeff) if concrete_coeff != 0 else 1
    periods = [d // gcd(C, d) for d in all_divisors]
    depth = lcm(*periods) if periods else 1
    print(
        f"  integer divisors={sorted(all_divisors)}  C={C}"
        f"  periods={periods}  probe_depth={depth}"
    )
    return depth


def compute_iv_stride_through_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    is_read: bool = True,
    mem_strides: list[IndexExpr] | None = None,
    constraints: Sequence = (),
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
    constraints : constraint sequence (may include ``Assumption`` objects).
        Divisibility assumptions (e.g. ``Assumption(Eq(K % 256, 0))``)
        are used to simplify floor/Mod expressions before probing.

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
                print(
                    f"IV coeff for {sym} is non-concrete: coeff={coeff}"
                    f"  resolved={concrete} (type={type(concrete).__name__})"
                    f" — skipping this IV"
                )
                continue
            iv_info[sym] = (iter_sym, int(concrete))

    if not iv_info:
        return None

    print(f"=== compute_iv_stride_through_mapping  is_read={is_read} ===")
    print(f"  iters: {dict(iters)}")
    for iv_sym, (iv_iter, cc) in iv_info.items():
        print(f"  IV {iv_sym} -> iter={iv_iter}  coeff={cc}")

    map_dims = (
        mapping.input_shape if is_read else mapping.output_shape
    )
    raw_exprs = (
        mapping.map_input_indices(map_dims) if is_read
        else mapping.map_output_indices(map_dims)
    )

    idxc = IndexingContext.current()
    dim_exprs = [subs_idxc(e) for e in raw_exprs]

    for i, (raw, resolved) in enumerate(zip(raw_exprs, dim_exprs)):
        changed = str(raw) != str(resolved)
        print(
            f"  dim[{i}]  raw={raw}  ->  resolved={resolved}"
            f"{'  (CHANGED by subs_idxc)' if changed else ''}"
        )

    if mem_strides is None:
        symbolic_shape_resolved = tuple(infer_dim(d) for d in symbolic_shape)
        mem_strides = strides_from_symbolic_shape(
            idxc, symbolic_shape_resolved, allow_mixed_shapes=True
        )

    stride_free = set()
    for s in mem_strides:
        stride_free |= sympy.sympify(s).free_symbols
    print(
        f"  mem_strides={mem_strides}"
        f"  (symbolic={sorted(str(s) for s in stride_free) if stride_free else 'none'})"
    )

    div_fwd, div_bwd = get_divisibility_subs(constraints)
    if div_fwd:
        fwd_dict = dict(div_fwd)
        dim_exprs = [sympy.sympify(e).subs(fwd_dict) for e in dim_exprs]
        mem_strides = [sympy.sympify(s).subs(fwd_dict) for s in mem_strides]
        print(f"  divisibility fwd subs: {fwd_dict}")
        for i, e in enumerate(dim_exprs):
            print(f"  dim_after_div_subs[{i}] = {e}")
        print(f"  mem_strides_after_div_subs={mem_strides}")
    else:
        floor_subs = _infer_floor_to_exact(mem_strides)
        if floor_subs:
            dim_exprs = [sympy.sympify(e).subs(floor_subs) for e in dim_exprs]
            print(f"  floor_to_exact subs (fallback): {floor_subs}")
            for i, e in enumerate(dim_exprs):
                print(f"  dim_after_subs[{i}] = {e}")

    result: dict[sympy.Symbol, IndexExpr | list[IndexExpr]] = {}

    for iv_sym, (iv_iter, concrete_coeff) in iv_info.items():
        stride_or_cycle = _probe_iv_stride(
            dim_exprs, mem_strides, iters, iv_iter, concrete_coeff
        )
        if stride_or_cycle is None:
            print(
                f"  _probe_iv_stride returned None for IV {iv_sym}"
                f" — no pattern detected, returning None for entire mapping"
            )
            return None
        result[iv_sym] = stride_or_cycle

    if div_bwd:
        bwd_dict = dict(div_bwd)
        def _bwd(v):
            if isinstance(v, list):
                return [simplify(sympy.sympify(x).subs(bwd_dict)) for x in v]
            return simplify(sympy.sympify(v).subs(bwd_dict))
        result = {k: _bwd(v) for k, v in result.items()}

    for iv_sym, val in result.items():
        print(f"  RESULT  {iv_sym} -> {val}")

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

    print(
        f"_probe_iv_stride  iv_iter={iv_iter}  coeff={concrete_coeff}"
        f"  probe_depth={probe_depth}"
    )

    def _linearized_addr(iv_val: int) -> IndexExpr:
        subs = {
            it: (concrete_coeff * iv_val if it == iv_iter else 0)
            for it in iters.keys()
        }
        addr = sympy.Integer(0)
        for dim_expr, stride in zip(dim_exprs, mem_strides):
            dim_val = subs_idxc(dim_expr.subs(subs))
            addr += simplify(dim_val) * stride
        addr = _expand_mod(addr)
        return simplify(subs_idxc(addr))

    addrs = [_linearized_addr(i) for i in range(probe_depth + 1)]
    diffs = [simplify(subs_idxc(addrs[i + 1] - addrs[i]))
             for i in range(probe_depth)]

    for i, a in enumerate(addrs):
        print(f"  addr[iv={i}] = {a}  (free={getattr(a, 'free_symbols', set())})")
    for i, d in enumerate(diffs):
        print(f"  diff[{i}] = {d}  (free={getattr(d, 'free_symbols', set())})")

    if not diffs:
        print("  -> no diffs (probe_depth=0?), returning None")
        return None

    # Constant stride (most common case).
    if all(simplify(d - diffs[0]) == 0 for d in diffs):
        is_concrete = getattr(diffs[0], 'free_symbols', set()) == set()
        print(
            f"  -> CONSTANT stride = {diffs[0]}"
            f"  (concrete={is_concrete}, probe_depth={probe_depth})"
        )
        if probe_depth == 1:
            print(
                "  *** WARNING: stride determined from only 1 diff "
                "(probe_depth=1). If symbolic floor/Mod divisors were "
                "skipped, this may be a single-sample guess."
            )
        return diffs[0]

    print("  diffs are NOT all equal — checking for symbolic residuals / cycles")

    # Symbolic strides may contain floor(a/d)*d + Mod(a,d) patterns that
    # simplify can't resolve.  Substitute a concrete value for remaining
    # free symbols; use a value that avoids aliasing with any divisor.
    free = set()
    for d in diffs:
        free |= getattr(d, 'free_symbols', set())
    if free:
        all_divs = set()
        for expr in dim_exprs:
            all_divs |= _extract_integer_divisors(expr)
        probe_val = lcm(*all_divs) * (probe_depth + 1) if all_divs else 8192
        numeric_subs = {s: probe_val for s in free}

        print(
            f"  NUMERIC FALLBACK: free_symbols={sorted(str(s) for s in free)}"
            f"  probe_val={probe_val}  all_divs={sorted(all_divs)}"
        )
        print(
            f"  *** Substituting {probe_val} for ALL free symbols"
            f" — aliasing between distinct symbols is possible"
        )

        try:
            numeric_diffs = [int(d.subs(numeric_subs)) for d in diffs]
        except (TypeError, ValueError) as exc:
            print(f"  numeric substitution failed ({exc}), returning None")
            return None

        print(f"  numeric_diffs = {numeric_diffs}")

        if len(set(numeric_diffs)) == 1:
            print(
                f"  -> NUMERIC CONSTANT stride = {numeric_diffs[0]}"
                f"  (*** symbolic info LOST: original diff[0] was {diffs[0]}"
                f" with free symbols {getattr(diffs[0], 'free_symbols', set())})"
            )
            return sympy.Integer(numeric_diffs[0])
        # Detect shortest repeating cycle from numeric diffs.
        for cycle_len in range(2, probe_depth // 2 + 1):
            if all(numeric_diffs[i] == numeric_diffs[i % cycle_len]
                   for i in range(probe_depth)):
                print(
                    f"  -> NUMERIC CYCLE (len={cycle_len}):"
                    f" {numeric_diffs[:cycle_len]}"
                    f"  (*** symbolic info LOST)"
                )
                return [sympy.Integer(numeric_diffs[i]) for i in range(cycle_len)]

        print("  numeric fallback found no constant or cycle pattern")

    # Detect shortest repeating cycle (symbolic).
    for cycle_len in range(2, probe_depth // 2 + 1):
        if all(simplify(diffs[i] - diffs[i % cycle_len]) == 0
               for i in range(probe_depth)):
            print(f"  -> SYMBOLIC CYCLE (len={cycle_len}): {list(diffs[:cycle_len])}")
            return list(diffs[:cycle_len])

    print(
        f"  -> FAILED: no constant stride or repeating cycle detected."
        f"  diffs={diffs}  returning None"
    )
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
