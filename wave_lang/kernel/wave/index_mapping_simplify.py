# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simplify IndexMapping expressions by decomposing flat // D and flat % D.

Given an IndexMapping whose input_mapping contains paired expressions:
    dim1: flat_expr // D
    dim2: flat_expr % D

This pass tries to decompose flat_expr = quotient * D + remainder where
0 <= remainder < D, allowing the rewrite:
    dim1: quotient
    dim2: remainder

The decomposition uses:
  1. Algebraic factoring: terms in flat_expr that are provably multiples of D
     are separated into the quotient.
  2. Bounds analysis: if the remaining terms are provably bounded in [0, D),
     the dynamic floordiv/mod is eliminated entirely.
"""

import sympy
from functools import lru_cache

from ..lang.wave_types import IndexMapping
from .utils.symbol_utils import (
    _split_sum_by_divisibility,
    expr_bounds,
    IndexExpr,
    IndexSymbol,
    subs_idxc,
)


def _get_iterator_bounds(
    mapping: IndexMapping,
) -> dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]]:
    """Extract iterator bounds from the IndexMapping's iteration_shape.

    Returns {iterator_symbol: (0, upper_bound - 1)} for each iterator.
    """
    bounds = {}
    for sym, idx in mapping.iters.items():
        dim = mapping.iteration_shape[idx]
        if dim is not None:
            bounds[sym] = (sympy.Integer(0), dim - 1)
    return bounds


def _expr_bounds_with_iters(
    expr: sympy.Expr,
    iter_bounds: dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]],
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Compute expression bounds using iterator upper bounds.

    Extends expr_bounds by substituting known iterator ranges.
    """
    if expr.is_Integer or expr.is_Rational:
        return (expr, expr)
    if expr.is_Symbol:
        if expr in iter_bounds:
            return iter_bounds[expr]
        return (sympy.Integer(0), sympy.oo) if expr.is_nonnegative else None

    # For floor/Mod/Add/Mul, delegate to structural recursion.
    if isinstance(expr, sympy.Mod):
        p, q = expr.args
        if q.is_positive and q.is_number:
            return (sympy.Integer(0), q - 1)
        q_bounds = _expr_bounds_with_iters(q, iter_bounds)
        if q_bounds and q_bounds[0].is_positive:
            return (sympy.Integer(0), q_bounds[1] - 1)
        return None

    if isinstance(expr, sympy.floor):
        inner_bounds = _expr_bounds_with_iters(expr.args[0], iter_bounds)
        if inner_bounds:
            return (sympy.floor(inner_bounds[0]), sympy.floor(inner_bounds[1]))
        return None

    if isinstance(expr, sympy.Add):
        bounds = [_expr_bounds_with_iters(a, iter_bounds) for a in expr.args]
        if all(b is not None for b in bounds):
            return (sum(b[0] for b in bounds), sum(b[1] for b in bounds))
        return None

    if isinstance(expr, sympy.Mul):
        if not expr.args:
            return (sympy.Integer(1), sympy.Integer(1))
        bounds = [_expr_bounds_with_iters(a, iter_bounds) for a in expr.args]
        if all(b is not None for b in bounds):
            if any(sympy.oo in b or -sympy.oo in b for b in bounds):
                return None
            lo, hi = bounds[0]
            for b in bounds[1:]:
                corners = [lo * b[0], lo * b[1], hi * b[0], hi * b[1]]
                try:
                    lo, hi = min(corners), max(corners)
                except TypeError:
                    return None
            return (lo, hi)
        return None

    return None


def _find_floordiv_mod_pairs(
    input_mapping: dict[IndexSymbol, IndexExpr],
) -> list[tuple]:
    """Find paired floor/Mod expressions sharing the same divisor.

    Returns list of (dim_q, dim_r, numerator, divisor, addend) tuples where:
      dim_q has expression: addend + floor(numerator / divisor)
      dim_r has expression: Mod(something, divisor)

    Note: sympy auto-evaluates Mod(A*D + B, D) → Mod(B, D), so the Mod's
    first arg may not match the floor's numerator exactly.  We match on
    the divisor and verify compatibility.
    """
    floor_info: list[tuple] = []  # (dim, numerator, divisor, addend)
    mod_info: list[tuple] = []  # (dim, arg, divisor)

    for dim, expr in input_mapping.items():
        # Top-level Mod(E, D).
        if isinstance(expr, sympy.Mod):
            mod_info.append((dim, expr.args[0], expr.args[1]))
            continue

        # Top-level floor(E / D).
        if isinstance(expr, sympy.floor):
            inner = expr.args[0]
            numer, denom = inner.as_numer_denom()
            if denom != 1:
                floor_info.append((dim, numer, denom, sympy.Integer(0)))
                continue

        # A + floor(E / D) pattern.
        if isinstance(expr, sympy.Add):
            for arg in expr.args:
                if isinstance(arg, sympy.floor):
                    inner = arg.args[0]
                    numer, denom = inner.as_numer_denom()
                    if denom != 1:
                        addend = expr - arg
                        floor_info.append((dim, numer, denom, addend))
                        break

    # Match on divisor.
    pairs = []
    for dim_q, numer, divisor, addend in floor_info:
        for dim_r, mod_arg, mod_divisor in mod_info:
            if divisor == mod_divisor:
                pairs.append((dim_q, dim_r, numer, divisor, addend))
                break

    return pairs


def simplify_index_mapping(
    mapping: IndexMapping,
    constraints=(),
) -> tuple[IndexMapping, bool]:
    """Simplify flat // D and flat % D patterns in an IndexMapping.

    Returns (new_mapping, changed).
    """
    iter_bounds = _get_iterator_bounds(mapping)
    input_mapping = dict(mapping.input_mapping)
    changed = False

    pairs = _find_floordiv_mod_pairs(input_mapping)
    for dim_q, dim_r, flat_expr, divisor, addend in pairs:
        # Step 1: Factor out D-multiples from flat_expr.
        split = _split_sum_by_divisibility(flat_expr, divisor)
        if split is None:
            quotient = sympy.Integer(0)
            remainder = flat_expr
        else:
            quotient, remainder = split

        # Step 2: Check if remainder is bounded in [0, D).
        rem_bounds = _expr_bounds_with_iters(remainder, iter_bounds)
        if rem_bounds is None:
            continue

        lo, hi = rem_bounds
        if hi == sympy.oo:
            continue

        # Check lo >= 0.
        lo_nonneg = lo.is_nonnegative if hasattr(lo, "is_nonnegative") else None
        if lo_nonneg is None:
            lo_simplified = sympy.simplify(lo)
            lo_nonneg = lo_simplified.is_nonnegative
        if not lo_nonneg:
            continue

        # Check hi < divisor, i.e. (hi - divisor) is negative.
        diff = sympy.simplify(hi - divisor)
        if diff.is_negative is not True:
            continue

        # Remainder < D proven! Eliminate the dynamic floordiv/mod.
        input_mapping[dim_q] = addend + quotient
        input_mapping[dim_r] = remainder
        changed = True

    if not changed:
        return mapping, False

    return (
        IndexMapping(
            mapping.num_iterators,
            input_mapping,
            dict(mapping.output_mapping),
            dynamic_val_mappings=tuple(
                dict(dvm) for dvm in (mapping.dynamic_val_mappings or ())
            ),
        ),
        True,
    )
