"""Run a pulmonary 0D model until its MPA pressure is periodic.

Seeds such as learnedZeroD carry ``number_of_cardiac_cycles: 2`` with
``steady_initial: False``.  Cycle 2 of that cold start is still filling the
compliance, so its diastolic pressure sits well below the periodic value
(TST-STAN-3 iter-2: 7.6 mmHg at cycle 2 vs 15.4 mmHg from cycle 5 on).
Scoring that cycle biases every tuned parameter.

``simulate_until_periodic`` overrides the cycle count for evaluation only:
it runs at least ``min_cycles`` and doubles the count, up to ``max_cycles``,
until the last two cycles' mean MPA pressure differ by less than
``mean_pressure_tol_mmhg``.  The model files themselves are never changed.

Control (``periodic_evaluation`` in the impedance config):
``{min_cycles: 6, max_cycles: 24, mean_pressure_tol_mmhg: 0.1}`` (defaults,
also used when omitted) or ``{enabled: false}`` to score the seed's cycle
count as before.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

MMHG_TO_BARYE = 1333.2

DEFAULT_MIN_CYCLES = 6
DEFAULT_MAX_CYCLES = 24
DEFAULT_MEAN_PRESSURE_TOL_MMHG = 0.1

_KEYS = {"enabled", "min_cycles", "max_cycles", "mean_pressure_tol_mmhg"}


@dataclass(frozen=True)
class PeriodicEvaluation:
    min_cycles: int = DEFAULT_MIN_CYCLES
    max_cycles: int = DEFAULT_MAX_CYCLES
    mean_pressure_tol_mmhg: float = DEFAULT_MEAN_PRESSURE_TOL_MMHG

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_cycles": self.min_cycles,
            "max_cycles": self.max_cycles,
            "mean_pressure_tol_mmhg": self.mean_pressure_tol_mmhg,
        }


def resolve_periodic_evaluation(
    payload: Any, *, label: str = "periodic_evaluation"
) -> PeriodicEvaluation | None:
    """Defaults when omitted; None (seed cycle count) for ``enabled: false``."""
    if isinstance(payload, PeriodicEvaluation):
        return payload
    if payload is None:
        return PeriodicEvaluation()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    unknown = set(payload) - _KEYS
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    if not bool(payload.get("enabled", True)):
        return None
    min_cycles = int(payload.get("min_cycles") or DEFAULT_MIN_CYCLES)
    max_cycles = int(payload.get("max_cycles") or max(DEFAULT_MAX_CYCLES, min_cycles))
    tol = payload.get("mean_pressure_tol_mmhg")
    tol = DEFAULT_MEAN_PRESSURE_TOL_MMHG if tol is None else float(tol)
    if min_cycles < 2:
        raise ValueError(f"{label}.min_cycles must be >= 2 (drift needs two cycles)")
    if max_cycles < min_cycles:
        raise ValueError(f"{label}.max_cycles must be >= min_cycles")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"{label}.mean_pressure_tol_mmhg must be > 0")
    return PeriodicEvaluation(min_cycles, max_cycles, tol)


def periodic_evaluation_payload(payload: Any, *, label: str = "periodic_evaluation") -> dict | None:
    """Resolved mapping for a config, or None when omitted (defaults apply)."""
    if payload is None:
        return None
    resolved = resolve_periodic_evaluation(payload, label=label)
    return {"enabled": False} if resolved is None else resolved.to_dict()


def cycle_metrics(time, pressure_mmhg, cycle_duration: float) -> list[dict[str, float]]:
    """Sys/dia/mean pressure of each complete cycle, oldest first.

    Cycle k (counted back from the end) is ``tmax-(k+1)T < t <= tmax-kT``,
    the same window the last-cycle metrics use.
    """
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure_mmhg, dtype=float)
    if time.size == 0 or not cycle_duration or cycle_duration <= 0.0:
        return []
    t_max, t_min = float(time.max()), float(time.min())
    n_cycles = int(math.floor((t_max - t_min) / cycle_duration + 1e-6))
    cycles = []
    for k in range(max(n_cycles, 1)):
        hi = t_max - k * cycle_duration
        mask = (time > hi - cycle_duration) & (time <= hi + 1e-12)
        if not np.any(mask):
            break
        window = pressure[mask]
        cycles.append({"sys": float(window.max()), "dia": float(window.min()), "mean": float(window.mean())})
    return cycles[::-1]


def simulate_until_periodic(
    simulate: Callable[[int], Any],
    pressure_series: Callable[[Any], tuple[Any, Any]],
    *,
    cycle_duration: float | None,
    seed_cycles: int | None,
    settings: PeriodicEvaluation | None,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(result, periodicity)`` for the last simulation run.

    ``simulate(n)`` runs the model for ``n`` cycles with all cycles in the
    output; ``n`` is None when ``settings`` is None (seed cycle count kept).
    ``pressure_series(result)`` returns the MPA ``(time, pressure_mmhg)``.
    """
    if settings is None:
        result = simulate(None)
        info = _describe(result, pressure_series, cycle_duration, seed_cycles, None)
        info.update({"enabled": False, "attempts": [info["cycles_used"]]})
        return result, info

    cycles = min(max(settings.min_cycles, int(seed_cycles or 0)), settings.max_cycles)
    attempts: list[int] = []
    while True:
        result = simulate(cycles)
        attempts.append(cycles)
        info = _describe(result, pressure_series, cycle_duration, seed_cycles, settings)
        drift = info["final_drift_mmhg"]
        # Stop when periodic, at the cap, or when the solver did not produce
        # the requested cycles (more would not help).
        if (
            info["converged"]
            or cycles >= settings.max_cycles
            or drift is None
            or info["cycles_used"] < cycles
        ):
            break
        cycles = min(2 * cycles, settings.max_cycles)
    info.update({"enabled": True, "attempts": attempts, "settings": settings.to_dict()})
    return result, info


def _describe(result, pressure_series, cycle_duration, seed_cycles, settings) -> dict[str, Any]:
    time, pressure = pressure_series(result)
    cycles = cycle_metrics(time, pressure, cycle_duration) if cycle_duration else []
    drift = None
    if len(cycles) >= 2:
        drift = {key: float(cycles[-1][key] - cycles[-2][key]) for key in ("sys", "dia", "mean")}
    final_drift = None if drift is None else abs(drift["mean"])
    # Geometric-tail estimate of how far the last cycle's mean still is from
    # the periodic value: |d_n| r / (1 - r) with r = d_n / d_(n-1).  A drift
    # below tolerance can hide a larger offset when the decay is slow.
    residual = None
    if len(cycles) >= 3 and final_drift is not None:
        previous = cycles[-2]["mean"] - cycles[-3]["mean"]
        ratio = drift["mean"] / previous if previous != 0.0 else None
        if ratio is not None and 0.0 < ratio < 1.0:
            residual = float(final_drift * ratio / (1.0 - ratio))
    tol = settings.mean_pressure_tol_mmhg if settings is not None else DEFAULT_MEAN_PRESSURE_TOL_MMHG
    return {
        "seed_number_of_cardiac_cycles": None if seed_cycles is None else int(seed_cycles),
        "cycles_used": len(cycles),
        # |mean(last) - mean(previous)| of MPA pressure [mmHg].
        "final_drift_mmhg": final_drift,
        # Signed last-minus-previous cycle sys/dia/mean MPA pressure [mmHg].
        "final_cycle_drift_mmhg": drift,
        "estimated_residual_mmhg": residual,
        "converged": bool(final_drift is not None and final_drift < tol),
        "last_cycles_mpa_mmhg": cycles[-3:],
    }


@contextmanager
def override_simparams(simparams, number_of_cardiac_cycles: int | None):
    """Temporarily set the cycle count (and all-cycle output) on a SimParams."""
    if simparams is None or number_of_cardiac_cycles is None:
        yield
        return
    keys = ("number_of_cardiac_cycles", "output_all_cycles")
    missing = object()
    saved = {key: getattr(simparams, key, missing) for key in keys}
    simparams.number_of_cardiac_cycles = int(number_of_cardiac_cycles)
    simparams.output_all_cycles = True
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(simparams, key)
            else:
                setattr(simparams, key, value)


def payload_with_cycles(payload: Mapping[str, Any], number_of_cardiac_cycles: int | None) -> dict:
    """Deep copy of a solver payload with the cycle count overridden."""
    copied = copy.deepcopy(dict(payload))
    if number_of_cardiac_cycles is not None:
        simparams = copied.setdefault("simulation_parameters", {})
        simparams["number_of_cardiac_cycles"] = int(number_of_cardiac_cycles)
        simparams["output_all_cycles"] = True
    return copied


def vessel_pressure_series(vessel_rows: Callable[[Any], Any]) -> Callable[[Any], tuple[Any, Any]]:
    """``pressure_series`` for a result frame given a vessel row selector."""

    def _series(result):
        rows = vessel_rows(result)
        return (
            np.asarray(rows.time, dtype=float),
            np.asarray(rows.pressure_in, dtype=float) / MMHG_TO_BARYE,
        )

    return _series
