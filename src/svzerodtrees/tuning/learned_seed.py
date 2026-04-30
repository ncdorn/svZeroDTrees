"""Helpers for preparing reduced seeds from learned full 0D references."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json
import math
from numbers import Real

import numpy as np
from scipy.optimize import Bounds, minimize

from ..io.blocks import BoundaryCondition
from ..io.config_handler import ConfigHandler
from ..tune_bcs.clinical_targets import ClinicalTargets
from ..tune_bcs.pa_config import PAConfig

MMHG_TO_BARYE = 1333.2
DEFAULT_SIDE_BC_RESISTANCE = 1000.0
DEFAULT_SIDE_BC_PD = 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(entry) for entry in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    return value


def _normalize_resistance_bc(
    bc: float | dict[str, Any] | BoundaryCondition | None,
    *,
    name: str,
) -> BoundaryCondition:
    if bc is None:
        config = {
            "bc_name": name,
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": DEFAULT_SIDE_BC_RESISTANCE,
                "Pd": DEFAULT_SIDE_BC_PD,
            },
        }
    elif isinstance(bc, BoundaryCondition):
        config = {
            "bc_name": name,
            "bc_type": bc.type,
            "bc_values": deepcopy(bc.values),
        }
    elif isinstance(bc, Real):
        config = {
            "bc_name": name,
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": float(bc),
                "Pd": DEFAULT_SIDE_BC_PD,
            },
        }
    elif isinstance(bc, dict):
        bc_copy = deepcopy(bc)
        if "bc_values" in bc_copy:
            config = {
                "bc_name": name,
                "bc_type": bc_copy.get("bc_type", "RESISTANCE"),
                "bc_values": bc_copy["bc_values"],
            }
        else:
            values = dict(bc_copy)
            values.setdefault("Pd", DEFAULT_SIDE_BC_PD)
            config = {
                "bc_name": name,
                "bc_type": "RESISTANCE",
                "bc_values": values,
            }
    else:
        raise TypeError(f"{name} must be a resistance value, dict, BoundaryCondition, or None")

    normalized = BoundaryCondition.from_config(config)
    if normalized.type != "RESISTANCE":
        raise ValueError(f"{name} must be a RESISTANCE boundary condition")
    return normalized


def _cycle_duration_from_handler(config_handler: ConfigHandler) -> float | None:
    period = getattr(config_handler.simparams, "cardiac_period", None)
    if period is not None:
        try:
            period_float = float(period)
        except (TypeError, ValueError):
            period_float = None
        if period_float is not None and math.isfinite(period_float) and period_float > 0.0:
            return period_float

    inflow = config_handler.bcs.get("INFLOW")
    times = getattr(inflow, "t", None)
    if times is None:
        return None
    times_array = np.asarray(times, dtype=float)
    if times_array.size < 2:
        return None
    duration = float(times_array.max() - times_array.min())
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _slice_last_cycle(result, cycle_duration: float | None):
    if cycle_duration is None or "time" not in result:
        return result
    time = np.asarray(result["time"], dtype=float)
    if time.size < 2:
        return result
    cutoff = float(time.max()) - float(cycle_duration)
    sliced = result[time >= cutoff]
    return sliced if not sliced.empty else result


def _vessel_result(result, vessel):
    by_name = result[result.name == vessel.name]
    if not by_name.empty:
        return by_name
    branch_name = f"branch{vessel.branch}_seg0"
    by_branch = result[result.name == branch_name]
    if not by_branch.empty:
        return by_branch
    raise ValueError(f"simulation result missing vessel '{vessel.name}'")


def _series(frame, preferred: str, fallback: str) -> np.ndarray:
    column = preferred if preferred in frame else fallback
    if column not in frame:
        raise ValueError(f"simulation result missing '{preferred}'/'{fallback}' columns")
    return np.asarray(frame[column], dtype=float)


def _integral_or_mean(values: np.ndarray, time: np.ndarray | None) -> float:
    if time is not None and time.size == values.size and values.size >= 2:
        return float(np.trapz(values, time))
    return float(np.mean(values))


def _extract_pa_metrics(
    result,
    *,
    mpa_vessel,
    rpa_vessel,
    cycle_duration: float | None,
    mpa_flow_column: str,
    rpa_flow_column: str,
) -> dict[str, Any]:
    mpa_result = _slice_last_cycle(_vessel_result(result, mpa_vessel), cycle_duration)
    rpa_result = _slice_last_cycle(_vessel_result(result, rpa_vessel), cycle_duration)
    if mpa_result.empty or rpa_result.empty:
        raise ValueError("simulation result has no usable PA data")

    pressure = np.asarray(mpa_result["pressure_in"], dtype=float) / MMHG_TO_BARYE
    mpa_flow = _series(mpa_result, mpa_flow_column, "flow_out")
    rpa_flow = _series(rpa_result, rpa_flow_column, "flow_out")

    mpa_time = np.asarray(mpa_result["time"], dtype=float) if "time" in mpa_result else None
    rpa_time = np.asarray(rpa_result["time"], dtype=float) if "time" in rpa_result else None
    total_flow = _integral_or_mean(mpa_flow, mpa_time)
    rpa_total = _integral_or_mean(rpa_flow, rpa_time)
    if not math.isfinite(total_flow) or total_flow == 0.0:
        raise ValueError("MPA flow is zero or non-finite; cannot compute RPA split")

    return {
        "P_mpa": [
            float(np.max(pressure)),
            float(np.min(pressure)),
            float(np.mean(pressure)),
        ],
        "rpa_split": float(rpa_total / total_flow),
        "mean_mpa_flow": float(np.mean(mpa_flow)),
        "mean_rpa_flow": float(np.mean(rpa_flow)),
    }


def _reduced_metrics(pa_config: PAConfig) -> dict[str, Any]:
    return _extract_pa_metrics(
        pa_config.result,
        mpa_vessel=pa_config.mpa,
        rpa_vessel=pa_config.rpa_prox,
        cycle_duration=_cycle_duration_from_inflow(pa_config.inflow),
        mpa_flow_column="flow_out",
        rpa_flow_column="flow_in",
    )


def _cycle_duration_from_inflow(inflow: BoundaryCondition) -> float | None:
    times = getattr(inflow, "t", None)
    if times is None:
        return None
    times_array = np.asarray(times, dtype=float)
    if times_array.size < 2:
        return None
    duration = float(times_array.max() - times_array.min())
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _is_steady_inflow(inflow: BoundaryCondition) -> bool:
    flow = np.asarray(getattr(inflow, "Q", []), dtype=float)
    return bool(flow.size > 0 and np.allclose(flow, flow[0]))


def _loss(
    pa_config: PAConfig,
    reference_metrics: dict[str, Any],
    resistances: np.ndarray,
) -> float:
    if not np.all(np.isfinite(resistances)) or np.any(resistances < 0.0):
        return 1.0e12

    pa_config.lpa_prox.R = float(resistances[0])
    pa_config.rpa_prox.R = float(resistances[1])

    try:
        pa_config.simulate()
        metrics = _reduced_metrics(pa_config)
    except Exception:
        return 1.0e12

    p_ref = np.asarray(reference_metrics["P_mpa"], dtype=float)
    p_fit = np.asarray(metrics["P_mpa"], dtype=float)
    p_scale = np.maximum(np.abs(p_ref), 1.0)
    pressure_components = ((p_fit - p_ref) / p_scale) ** 2
    pressure_weights = np.asarray([1.5, 1.0, 1.2], dtype=float)
    pressure_loss = float(np.dot(pressure_weights, pressure_components) * 100.0)

    split_ref = float(reference_metrics["rpa_split"])
    split_scale = max(abs(split_ref), 1.0e-6)
    split_loss = float(((float(metrics["rpa_split"]) - split_ref) / split_scale) ** 2 * 100.0)
    return pressure_loss + split_loss


def _initial_resistances(pa_config: PAConfig) -> np.ndarray:
    initial = np.asarray([pa_config.lpa_prox.R, pa_config.rpa_prox.R], dtype=float)
    if not np.all(np.isfinite(initial)) or np.any(initial <= 0.0):
        initial = np.asarray([1.0, 1.0], dtype=float)
    return initial


def prepare_reduced_rri_seed_from_learned(
    *,
    learned_config: str | Path,
    output_config: str | Path | None = None,
    reduced_template: str | Path | None = None,
    metrics_path: str | Path | None = None,
    lpa_bc: float | dict[str, Any] | BoundaryCondition | None = None,
    rpa_bc: float | dict[str, Any] | BoundaryCondition | None = None,
    maxiter: int = 200,
    solver: str = "Nelder-Mead",
) -> dict[str, Any]:
    """Fit a reduced RRI PAConfig seed to a learned full 0D reference.

    The learned config is simulated first and used as the target source for MPA
    pressure and RPA flow split. The reduced RRI model uses fixed LPA/RPA
    resistance boundary conditions: default ``R=1000.0, Pd=0.0`` per side, or
    caller-provided resistance BCs.
    """

    learned_path = Path(learned_config).expanduser()
    if not learned_path.exists():
        raise FileNotFoundError(f"learned 0D config not found: {learned_path}")

    learned_handler = ConfigHandler.from_json(str(learned_path), is_pulmonary=True)
    learned_result = learned_handler.simulate()
    reference_metrics = _extract_pa_metrics(
        learned_result,
        mpa_vessel=learned_handler.mpa,
        rpa_vessel=learned_handler.rpa,
        cycle_duration=_cycle_duration_from_handler(learned_handler),
        mpa_flow_column="flow_in",
        rpa_flow_column="flow_in",
    )

    clinical_targets = ClinicalTargets(
        mpa_p=reference_metrics["P_mpa"],
        q=reference_metrics["mean_mpa_flow"],
        rpa_split=reference_metrics["rpa_split"],
        wedge_p=0.0,
        steady=_is_steady_inflow(learned_handler.bcs["INFLOW"]),
    )
    pa_config = PAConfig.from_config_handler(
        learned_handler,
        clinical_targets,
        steady=clinical_targets.steady,
    )
    pa_config.bcs = {
        "INFLOW": pa_config.inflow,
        "RPA_BC": _normalize_resistance_bc(rpa_bc, name="RPA_BC"),
        "LPA_BC": _normalize_resistance_bc(lpa_bc, name="LPA_BC"),
    }

    x0 = _initial_resistances(pa_config)
    result = minimize(
        fun=lambda x: _loss(pa_config, reference_metrics, np.asarray(x, dtype=float)),
        x0=x0,
        method=solver,
        bounds=Bounds([0.0, 0.0], [np.inf, np.inf]),
        options={"maxiter": int(maxiter)},
    )

    final_x = np.asarray(result.x if result.x is not None else x0, dtype=float)
    pa_config.lpa_prox.R = float(final_x[0])
    pa_config.rpa_prox.R = float(final_x[1])
    pa_config.simulate()
    optimized_metrics = _reduced_metrics(pa_config)

    optimized_config = pa_config.config
    output_path = Path(output_config).expanduser() if output_config is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pa_config.to_json(output_path)

    fit_residuals = {
        "P_mpa": (
            np.asarray(optimized_metrics["P_mpa"], dtype=float)
            - np.asarray(reference_metrics["P_mpa"], dtype=float)
        ).tolist(),
        "rpa_split": float(optimized_metrics["rpa_split"] - reference_metrics["rpa_split"]),
    }
    metrics: dict[str, Any] = {
        "method": "rri_from_learned_reference",
        "learned_config": str(learned_path),
        "source_template": str(Path(reduced_template).expanduser()) if reduced_template else None,
        "output_config": str(output_path) if output_path is not None else None,
        "reference_metrics": reference_metrics,
        "optimized_metrics": optimized_metrics,
        "fit_residuals": fit_residuals,
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "fun": float(result.fun) if result.fun is not None else None,
            "nit": int(result.nit) if hasattr(result, "nit") else None,
            "nfev": int(result.nfev) if hasattr(result, "nfev") else None,
            "x": final_x.tolist(),
            "solver": solver,
            "maxiter": int(maxiter),
        },
        "boundary_conditions": {
            "LPA_BC": pa_config.bcs["LPA_BC"].to_dict(),
            "RPA_BC": pa_config.bcs["RPA_BC"].to_dict(),
        },
        "optimized_config": optimized_config,
    }
    metrics = _json_safe(metrics)

    if metrics_path is not None:
        metrics_out = Path(metrics_path).expanduser()
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return metrics
