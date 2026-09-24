"""Clinically scaled stopping rules for Nelder-Mead boundary-condition tuning.

SciPy's Nelder-Mead defaults (``xatol=fatol=1e-4`` in raw parameter and loss
units) keep refining far below the precision of the clinical targets.  This
module adds a stopping policy expressed in the units the fit is judged in:

* ``target_tolerance``: stop as soon as every metric in the objective is
  within this relative error of its clinical target.
* ``stall_window`` / ``stall_rel_improvement``: stop when the best loss has
  improved by less than ``stall_rel_improvement`` (relative) over the last
  ``stall_window`` evaluations.
* ``xatol`` / ``fatol``: SciPy's simplex tolerances, with ``xatol`` applied in
  a bounds-normalized [0, 1] parameter space so it means the same thing for
  every parameter.
* ``maxfev``: hard cap on evaluations per Nelder-Mead run.

``restart_min_rel_improvement`` is consumed by the caller's restart loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize


STOP_TARGET_MET = "target_met"
STOP_STALLED = "stalled"
STOP_CONVERGED = "simplex_converged"
STOP_MAXFEV = "maxfev"


@dataclass(frozen=True)
class NelderMeadStopping:
    target_tolerance: float | None = 0.025
    stall_window: int | None = None  # None -> 5 x number of free parameters
    stall_rel_improvement: float | None = 0.01
    xatol: float = 1e-3
    fatol: float = 1e-3
    maxfev: int = 200
    initial_simplex_step: float = 0.1
    restart_min_rel_improvement: float | None = 0.05

    def resolved_stall_window(self, n_free: int) -> int:
        return int(self.stall_window) if self.stall_window is not None else 5 * int(n_free)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FIELDS = set(NelderMeadStopping.__dataclass_fields__)


def resolve_nelder_mead_stopping(
    config: Mapping[str, Any] | NelderMeadStopping | None,
    *,
    label: str = "stopping",
) -> NelderMeadStopping | None:
    """Validate a stopping mapping; ``None`` or ``enabled: false`` means legacy."""

    if config is None or isinstance(config, NelderMeadStopping):
        return config
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} must be a mapping")
    payload = dict(config)
    if not bool(payload.pop("enabled", True)):
        return None
    unknown = sorted(set(payload) - _FIELDS)
    if unknown:
        raise ValueError(f"{label} has unknown keys: {unknown}")

    def _optional_positive(key: str) -> float | None:
        value = payload.get(key, getattr(NelderMeadStopping, key))
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}.{key} must be > 0 or null")
        return value

    def _positive(key: str) -> float:
        value = payload.get(key, getattr(NelderMeadStopping, key))
        if value is None:
            raise ValueError(f"{label}.{key} cannot be null")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}.{key} must be > 0")
        return value

    stall_window = payload.get("stall_window")
    if stall_window is not None:
        stall_window = int(stall_window)
        if stall_window <= 0:
            raise ValueError(f"{label}.stall_window must be > 0 or null")
    maxfev = int(_positive("maxfev"))
    step = _positive("initial_simplex_step")
    if step > 0.5:
        raise ValueError(f"{label}.initial_simplex_step must be <= 0.5")
    return NelderMeadStopping(
        target_tolerance=_optional_positive("target_tolerance"),
        stall_window=stall_window,
        stall_rel_improvement=_optional_positive("stall_rel_improvement"),
        xatol=_positive("xatol"),
        fatol=_positive("fatol"),
        maxfev=maxfev,
        initial_simplex_step=step,
        restart_min_rel_improvement=_optional_positive("restart_min_rel_improvement"),
    )


class BoundsScaler:
    """Affine map between optimizer space and [0, 1] per finite-bounded axis.

    Axes with an infinite bound are left unscaled.
    """

    def __init__(self, bounds):
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        self.scaled = np.isfinite(lb) & np.isfinite(ub) & (ub > lb)
        self.offset = np.where(self.scaled, lb, 0.0)
        self.scale = np.where(self.scaled, ub - lb, 1.0)
        self.unit_bounds = [
            (0.0, 1.0) if s else (float(lo), float(hi))
            for s, lo, hi in zip(self.scaled, lb, ub)
        ]

    def to_unit(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.offset) / self.scale

    def from_unit(self, u) -> np.ndarray:
        return self.offset + np.asarray(u, dtype=float) * self.scale


def initial_simplex(u0, scaler: BoundsScaler, step: float) -> np.ndarray:
    """Simplex of ``u0`` plus one ``step`` along each axis, stepping inward at bounds."""

    u0 = np.asarray(u0, dtype=float)
    n = u0.size
    sim = np.tile(u0, (n + 1, 1))
    for k in range(n):
        if scaler.scaled[k]:
            delta = step if u0[k] + step <= 1.0 else -step
        else:
            delta = step * max(abs(u0[k]), 1.0)
            hi = scaler.unit_bounds[k][1]
            if u0[k] + delta > hi:
                delta = -delta
        sim[k + 1, k] = u0[k] + delta
    return sim


@dataclass
class EvaluationRecord:
    x: np.ndarray
    loss: float
    breakdown: dict


@dataclass
class NelderMeadRunOutcome:
    x: np.ndarray
    loss: float
    breakdown: dict
    reason: str
    n_evaluations: int
    message: str = ""


class _EarlyStop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Tracker:
    def __init__(self, stopping, n_free, target_met):
        self.stopping = stopping
        self.n_free = int(n_free)
        self.window = stopping.resolved_stall_window(n_free)
        self.target_met = target_met
        self.best: EvaluationRecord | None = None
        self.stop_point: EvaluationRecord | None = None
        self.best_history: list[float] = []

    def record(self, x, loss, breakdown):
        if breakdown and np.isfinite(loss):
            rec = EvaluationRecord(np.array(x, dtype=float), float(loss), dict(breakdown))
            if self.best is None or rec.loss < self.best.loss:
                self.best = rec
            tol = self.stopping.target_tolerance
            if tol is not None and self.target_met(rec.breakdown, tol):
                self.stop_point = rec
                self._append()
                raise _EarlyStop(STOP_TARGET_MET)
        self._append()
        if self._stalled():
            raise _EarlyStop(STOP_STALLED)

    def _append(self):
        self.best_history.append(self.best.loss if self.best is not None else np.inf)

    def _stalled(self) -> bool:
        rel = self.stopping.stall_rel_improvement
        k = len(self.best_history)
        # Only compare once the initial simplex has been evaluated.
        if rel is None or k < self.window + self.n_free + 1:
            return False
        old = self.best_history[k - 1 - self.window]
        new = self.best_history[-1]
        if not np.isfinite(old) or old <= 0.0:
            return False
        return (old - new) / old < rel


def run_nelder_mead(
    evaluate: Callable[[np.ndarray], tuple[float, dict | None]],
    x0,
    bounds,
    stopping: NelderMeadStopping,
    target_met: Callable[[dict, float], bool],
) -> NelderMeadRunOutcome:
    """Run one bounded Nelder-Mead pass under ``stopping``.

    ``evaluate(x)`` returns ``(loss, breakdown)`` with ``breakdown=None`` for a
    failed evaluation.  ``target_met(breakdown, tolerance)`` decides the
    target stop.  The outcome is the target-meeting point when the target stop
    fires, otherwise the lowest-loss successful evaluation.
    """

    scaler = BoundsScaler(bounds)
    x0 = np.clip(np.asarray(x0, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])
    u0 = scaler.to_unit(x0)
    tracker = _Tracker(stopping, u0.size, target_met)

    def _fun(u):
        x = scaler.from_unit(u)
        loss, breakdown = evaluate(x)
        tracker.record(x, loss, breakdown)
        return loss

    message = ""
    try:
        result = minimize(
            _fun,
            u0,
            method="Nelder-Mead",
            bounds=scaler.unit_bounds,
            options={
                "maxfev": int(stopping.maxfev),
                "xatol": float(stopping.xatol),
                "fatol": float(stopping.fatol),
                "initial_simplex": initial_simplex(u0, scaler, stopping.initial_simplex_step),
            },
        )
        reason = STOP_CONVERGED if result.success else STOP_MAXFEV
        message = str(result.message)
    except _EarlyStop as stop:
        reason = stop.reason

    chosen = tracker.stop_point if reason == STOP_TARGET_MET else tracker.best
    n_evals = len(tracker.best_history)
    if chosen is None:
        return NelderMeadRunOutcome(x0, np.inf, {}, reason, n_evals, message)
    return NelderMeadRunOutcome(chosen.x, chosen.loss, chosen.breakdown, reason, n_evals, message)
