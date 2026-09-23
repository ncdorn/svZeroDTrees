"""Closed-form DC (steady Poiseuille) resistance of a structured tree.

The recursion mirrors ``build_tree_soa`` and the ``poiseuille_network`` DC
mode of ``StructuredTree.compute_olufsen_impedance`` without allocating the
tree: the root is always split, a child with ``d < d_min`` is a collapsed
leaf that contributes only its own segment resistance, and every other child
is split again.  Every vessel diameter has the form ``d0 * alpha**i * beta**j``
so the recursion is memoized on ``(i, j)`` and costs O(n_alpha * n_beta).

All quantities are CGS.  Segment resistance is ``8 * eta * L / (pi * r**4)``
with ``L = lrr * r``.  ``eta`` and ``lrr`` scale every segment identically,
so they cancel in :func:`conductance_matched_diameter`.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np

__all__ = [
    "structured_tree_dc_resistance",
    "structured_tree_node_count",
    "conductance_matched_diameter",
]

_BISECTION_ITERATIONS = 80
_SCAN_POINTS = 256


def _validate_branching(d_min: float, alpha: float, beta: float) -> None:
    if not np.isfinite(d_min) or d_min <= 0.0:
        raise ValueError(f"d_min must be finite and > 0, got {d_min}")
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")


def structured_tree_dc_resistance(
    initial_d: float,
    *,
    d_min: float,
    alpha: float,
    beta: float,
    lrr: float = 1.0,
    eta: float = 1.0,
) -> float:
    """Return the root DC resistance [dyn s / cm^5] of an unbuilt structured tree.

    Matches ``StructuredTree.build(...)`` followed by
    ``equivalent_resistance()`` (zero terminal resistance), provided the built
    tree is not truncated by ``max_nodes``.
    """

    initial_d = float(initial_d)
    d_min = float(d_min)
    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(initial_d) or initial_d <= 0.0:
        raise ValueError(f"initial_d must be finite and > 0, got {initial_d}")
    _validate_branching(d_min, alpha, beta)

    coeff = 8.0 * float(eta) * float(lrr) / math.pi

    def segment(d: float) -> float:
        r = 0.5 * d
        return coeff / r ** 3

    cache: dict[tuple[int, int], float] = {}

    def expanded(i: int, j: int) -> float:
        key = (i, j)
        cached = cache.get(key)
        if cached is not None:
            return cached
        d = initial_d * alpha ** i * beta ** j
        conductance = 0.0
        for ci, cj in ((i + 1, j), (i, j + 1)):
            dc = initial_d * alpha ** ci * beta ** cj
            child = segment(dc) if dc < d_min else expanded(ci, cj)
            conductance += 1.0 / child
        value = segment(d) + 1.0 / conductance
        cache[key] = value
        return value

    return float(expanded(0, 0))


def structured_tree_node_count(
    initial_d: float,
    *,
    d_min: float,
    alpha: float,
    beta: float,
) -> int:
    """Return the number of vessels ``build_tree_soa`` stores without truncation."""

    initial_d = float(initial_d)
    _validate_branching(float(d_min), float(alpha), float(beta))
    cache: dict[tuple[int, int], int] = {}

    def expanded(i: int, j: int) -> int:
        key = (i, j)
        cached = cache.get(key)
        if cached is not None:
            return cached
        count = 1
        for ci, cj in ((i + 1, j), (i, j + 1)):
            dc = initial_d * alpha ** ci * beta ** cj
            count += 1 if dc < d_min else expanded(ci, cj)
        cache[key] = count
        return count

    return int(expanded(0, 0))


def conductance_matched_diameter(
    diameters: Iterable[float],
    *,
    d_min: float,
    alpha: float,
    beta: float,
    max_nodes: int | None = 100_000,
) -> tuple[float, float]:
    """Return ``(d_ref, relative_residual)`` for a shared tree.

    ``d_ref`` is the root diameter whose structured tree, repeated once per
    outlet, has the same total DC conductance as one tree per outlet built at
    ``diameters``: ``N * G(d_ref) = sum_i G(d_i)``.

    ``G(d)`` is piecewise smooth: it jumps when a generation crosses
    ``d_min``.  The target is bracketed on a log-spaced scan of
    ``[min(d_i), max(d_i)]``; among bracketing intervals the one nearest the
    arithmetic mean diameter is refined by bisection.  If a jump skips over
    the target, the closest evaluated diameter is returned and the relative
    conductance residual ``G(d_ref) / mean(G(d_i)) - 1`` reports the mismatch.

    The match assumes untruncated trees.  If the largest tree would exceed
    ``max_nodes`` (the ``StructuredTree.build`` default), a warning is emitted
    because the built trees will then have lower resistance than modeled here.
    """

    values = np.asarray(list(diameters), dtype=np.float64)
    if values.size == 0:
        raise ValueError("conductance_matched_diameter requires at least one diameter")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("conductance_matched_diameter requires finite diameters > 0")
    _validate_branching(float(d_min), float(alpha), float(beta))

    def conductance(d: float) -> float:
        return 1.0 / structured_tree_dc_resistance(
            d, d_min=d_min, alpha=alpha, beta=beta
        )

    if max_nodes is not None:
        largest = structured_tree_node_count(
            float(values.max()), d_min=d_min, alpha=alpha, beta=beta
        )
        if largest > int(max_nodes):
            warnings.warn(
                f"conductance_matched_diameter: a tree at d={float(values.max()):.4g} "
                f"has {largest} vessels > max_nodes={int(max_nodes)}; built trees "
                "will be truncated, so the DC conductance match is approximate.",
                stacklevel=2,
            )

    target = float(np.mean([conductance(d) for d in values]))
    d_lo = float(values.min())
    d_hi = float(values.max())
    if math.isclose(d_lo, d_hi, rel_tol=1e-12, abs_tol=0.0):
        return d_lo, conductance(d_lo) / target - 1.0

    grid = np.geomspace(d_lo, d_hi, _SCAN_POINTS)
    residual = np.array([conductance(d) - target for d in grid])

    best_index = int(np.argmin(np.abs(residual)))
    best_d, best_residual = float(grid[best_index]), float(residual[best_index])

    brackets = [
        k for k in range(grid.size - 1)
        if residual[k] == 0.0 or residual[k] * residual[k + 1] < 0.0
    ]
    if brackets:
        log_mean = math.log(float(values.mean()))
        k = min(
            brackets,
            key=lambda idx: abs(0.5 * (math.log(grid[idx]) + math.log(grid[idx + 1])) - log_mean),
        )
        lo, hi = math.log(grid[k]), math.log(grid[k + 1])
        f_lo = residual[k]
        for _ in range(_BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            f_mid = conductance(math.exp(mid)) - target
            if abs(f_mid) < abs(best_residual):
                best_d, best_residual = math.exp(mid), f_mid
            if f_mid == 0.0:
                break
            if (f_lo < 0.0) == (f_mid < 0.0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid

    return best_d, best_residual / target
