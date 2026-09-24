"""Closed-form DC (steady Poiseuille) resistance of a structured tree.

The recursion mirrors ``build_tree_soa`` and the ``poiseuille_network`` DC
mode of ``StructuredTree.compute_olufsen_impedance`` without allocating the
tree: the root is always split, a child with ``d < d_min`` is a collapsed
leaf that contributes only its own segment resistance, and every other child
is split again.  Every vessel diameter has the form ``d0 * alpha**i * beta**j``
so the recursion is evaluated bottom-up over the ``(i, j)`` lattice and costs
O(n_alpha * n_beta).  It is iterative because ``alpha`` near 1 gives trees
thousands of generations deep.

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

_SCAN_POINTS = 256
# max_nodes values already reported by the truncation warning.  Tuning calls
# the solver once per optimizer evaluation, and per-evaluation filter resets
# defeat the default once-per-location warning filter.
_TRUNCATION_WARNED: set[int] = set()
_REFINE_POINTS = 64
_REFINE_PASSES = 4


def _validate_branching(d_min: float, alpha: float, beta: float) -> None:
    if not np.isfinite(d_min) or d_min <= 0.0:
        raise ValueError(f"d_min must be finite and > 0, got {d_min}")
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")


def _reduce_tree(initial_d, d_min, alpha, beta, leaf, combine):
    """Fold a function over the unbuilt tree lattice, deepest generation first.

    ``leaf(d)`` is the value of a collapsed child; ``combine(d, left, right)``
    is the value of an expanded vessel of diameter ``d`` given its children.
    """

    def diameter(i: int, j: int) -> float:
        return initial_d * alpha ** i * beta ** j

    # Expanded vessels satisfy d >= d_min (the root is always expanded), so
    # i <= n_alpha and j <= n_beta bound the lattice; +1 absorbs rounding.
    n_alpha = max(0, int(math.log(initial_d / d_min) / -math.log(alpha)) + 1)
    n_beta = max(0, int(math.log(initial_d / d_min) / -math.log(beta)) + 1)

    def expanded(i: int, j: int) -> bool:
        return (i == 0 and j == 0) or diameter(i, j) >= d_min

    values: dict[tuple[int, int], float] = {}
    for i in range(n_alpha, -1, -1):
        for j in range(n_beta, -1, -1):
            if not expanded(i, j):
                continue
            children = []
            for ci, cj in ((i + 1, j), (i, j + 1)):
                if expanded(ci, cj):
                    children.append(values[(ci, cj)])
                else:
                    children.append(leaf(diameter(ci, cj)))
            values[(i, j)] = combine(diameter(i, j), children[0], children[1])
    return values[(0, 0)]


def _dc_resistance_many(initial_d, d_min, alpha, beta, lrr=1.0, eta=1.0):
    """Vectorized root DC resistance for an array of root diameters.

    Every lattice cell ``(i, j)`` holds ``R = R_seg(d)`` plus, when the vessel
    is expanded (``d >= d_min`` or the root), the parallel combination of its
    children.  A collapsed child's value is then exactly its own segment
    resistance, as in ``build_tree_soa``.  Rows are swept from the deepest
    ``i`` upward so only one row of the lattice is held at a time.
    """

    d0 = np.asarray(initial_d, dtype=np.float64)
    coeff = 8.0 * float(eta) * float(lrr) / math.pi
    ratio = float(np.max(d0)) / d_min
    n_alpha = max(0, int(math.log(ratio) / -math.log(alpha)) + 1) if ratio > 1.0 else 0
    n_beta = max(0, int(math.log(ratio) / -math.log(beta)) + 1) if ratio > 1.0 else 0
    beta_pow = beta ** np.arange(n_beta + 2, dtype=np.float64)

    below = None  # row i + 1, shape (n_beta + 2, len(d0))
    for i in range(n_alpha + 1, -1, -1):
        d_row = d0[None, :] * (alpha ** i) * beta_pow[:, None]
        row = coeff / (0.5 * d_row) ** 3
        expanded = d_row >= d_min
        if i == 0:
            expanded[0, :] = True
        if i <= n_alpha:
            for j in range(n_beta, -1, -1):
                mask = expanded[j]
                if not mask.any():
                    continue
                left = below[j, mask]
                right = row[j + 1, mask]
                row[j, mask] += 1.0 / (1.0 / left + 1.0 / right)
        below = row
    return below[0]


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

    return float(
        _dc_resistance_many(np.array([initial_d]), d_min, alpha, beta, lrr=lrr, eta=eta)[0]
    )


def structured_tree_node_count(
    initial_d: float,
    *,
    d_min: float,
    alpha: float,
    beta: float,
) -> int:
    """Return the number of vessels ``build_tree_soa`` stores without truncation."""

    initial_d = float(initial_d)
    if not np.isfinite(initial_d) or initial_d <= 0.0:
        raise ValueError(f"initial_d must be finite and > 0, got {initial_d}")
    _validate_branching(float(d_min), float(alpha), float(beta))
    return int(
        _reduce_tree(
            initial_d,
            float(d_min),
            float(alpha),
            float(beta),
            leaf=lambda _d: 1,
            combine=lambda _d, left, right: 1 + left + right,
        )
    )


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
    arithmetic mean diameter is refined by nested log-spaced scans.  If a jump skips over
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

    def conductance(d):
        return 1.0 / _dc_resistance_many(np.atleast_1d(d), float(d_min), float(alpha), float(beta))

    if max_nodes is not None:
        largest = structured_tree_node_count(
            float(values.max()), d_min=d_min, alpha=alpha, beta=beta
        )
        if largest > int(max_nodes) and int(max_nodes) not in _TRUNCATION_WARNED:
            _TRUNCATION_WARNED.add(int(max_nodes))
            warnings.warn(
                "conductance_matched_diameter: per-outlet trees exceed "
                f"max_nodes={int(max_nodes)} and will be truncated when built, so "
                "the DC conductance match is approximate.",
                stacklevel=2,
            )

    target = float(np.mean(conductance(values)))
    d_lo = float(values.min())
    d_hi = float(values.max())
    if math.isclose(d_lo, d_hi, rel_tol=1e-12, abs_tol=0.0):
        return d_lo, float(conductance(d_lo)[0]) / target - 1.0

    log_mean = math.log(float(values.mean()))
    grid = np.geomspace(d_lo, d_hi, _SCAN_POINTS)
    residual = conductance(grid) - target
    best_index = int(np.argmin(np.abs(residual)))
    best_d, best_residual = float(grid[best_index]), float(residual[best_index])

    # Refine the sign change nearest the arithmetic mean with nested scans;
    # each pass shrinks the bracket by a factor of _REFINE_POINTS.
    for _ in range(_REFINE_PASSES):
        crossings = np.nonzero((residual[:-1] == 0.0) | (residual[:-1] * residual[1:] < 0.0))[0]
        if crossings.size == 0:
            break
        mids = 0.5 * (np.log(grid[crossings]) + np.log(grid[crossings + 1]))
        k = int(crossings[np.argmin(np.abs(mids - log_mean))])
        grid = np.geomspace(grid[k], grid[k + 1], _REFINE_POINTS)
        residual = conductance(grid) - target
        index = int(np.argmin(np.abs(residual)))
        if abs(residual[index]) < abs(best_residual):
            best_d, best_residual = float(grid[index]), float(residual[index])

    return best_d, best_residual / target
