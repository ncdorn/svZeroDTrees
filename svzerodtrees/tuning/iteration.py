"""Utilities for iterative 3D-0D tuning decisions and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from svzerodtrees.simulation import SimulationDirectory
from svzerodtrees.tune_bcs import ClinicalTargets


DEFAULT_ITERATION_THRESHOLDS: dict[str, float] = {
    "mpa_sys": 5.0,
    "mpa_dia": 3.0,
    "mpa_mean": 3.0,
    "rpa_split": 0.05,
}


def _to_float_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("input sequence has no finite values")
    return arr


def _load_centerline_pressure_series(csv_path: str | Path) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"centerline pressure CSV not found: {path}")

    df = pd.read_csv(path)
    candidate_cols = [
        "mpa_pressure_mmhg",
        "pressure_mmhg",
        "pressure",
        "mpa_pressure",
    ]
    for col in candidate_cols:
        if col in df.columns:
            return _to_float_array(df[col].to_numpy())

    raise ValueError(
        "centerline pressure CSV must contain one of: "
        + ", ".join(candidate_cols)
    )


def compute_centerline_mpa_metrics(
    *,
    pressure_csv: str | Path | None = None,
    pressure_values: Sequence[float] | None = None,
    last_n: int | None = None,
) -> dict[str, float]:
    """Compute systolic/diastolic/mean MPA pressure (mmHg) for gate decisions."""

    if pressure_values is None and pressure_csv is None:
        raise ValueError("one of pressure_values or pressure_csv is required")

    if pressure_values is not None:
        arr = _to_float_array(pressure_values)
    else:
        arr = _load_centerline_pressure_series(pressure_csv)  # type: ignore[arg-type]

    if last_n is not None:
        if last_n <= 0:
            raise ValueError("last_n must be positive")
        arr = arr[-last_n:]

    return {
        "mpa_sys": float(np.max(arr)),
        "mpa_dia": float(np.min(arr)),
        "mpa_mean": float(np.mean(arr)),
    }


def _flow_totals_from_simulation_dir(simulation_dir: str | Path) -> tuple[float, float]:
    sim = SimulationDirectory.from_directory(str(simulation_dir))
    lpa_flow, rpa_flow = sim.flow_split(get_mean=True, verbose=False)
    lpa_total = float(sum(float(v) for v in lpa_flow.values()))
    rpa_total = float(sum(float(v) for v in rpa_flow.values()))
    return lpa_total, rpa_total


def compute_flow_split_metrics(
    *,
    simulation_dir: str | Path | None = None,
    lpa_total_flow: float | None = None,
    rpa_total_flow: float | None = None,
) -> dict[str, float]:
    """Compute LPA/RPA flow totals and RPA split for gate decisions."""

    if simulation_dir is not None:
        lpa_val, rpa_val = _flow_totals_from_simulation_dir(simulation_dir)
    elif lpa_total_flow is not None and rpa_total_flow is not None:
        lpa_val = float(lpa_total_flow)
        rpa_val = float(rpa_total_flow)
    else:
        raise ValueError(
            "provide simulation_dir or both lpa_total_flow and rpa_total_flow"
        )

    total = lpa_val + rpa_val
    if total <= 0.0:
        raise ValueError("total flow must be positive to compute split")

    return {
        "lpa_flow": lpa_val,
        "rpa_flow": rpa_val,
        "rpa_split": rpa_val / total,
    }


def _clinical_targets_from_input(
    clinical_targets: str | Path | Mapping[str, Any],
) -> dict[str, float]:
    if isinstance(clinical_targets, Mapping):
        if "mpa_p" not in clinical_targets or "rpa_split" not in clinical_targets:
            raise ValueError("clinical_targets mapping must define mpa_p and rpa_split")
        mpa_p = clinical_targets["mpa_p"]
        if not isinstance(mpa_p, Sequence) or len(mpa_p) < 3:
            raise ValueError("clinical_targets.mpa_p must be a length-3 sequence")
        return {
            "mpa_sys": float(mpa_p[0]),
            "mpa_dia": float(mpa_p[1]),
            "mpa_mean": float(mpa_p[2]),
            "rpa_split": float(clinical_targets["rpa_split"]),
        }

    ct = ClinicalTargets.from_csv(str(clinical_targets))
    return {
        "mpa_sys": float(ct.mpa_p[0]),
        "mpa_dia": float(ct.mpa_p[1]),
        "mpa_mean": float(ct.mpa_p[2]),
        "rpa_split": float(ct.rpa_split),
    }


def evaluate_iteration_gate(
    *,
    metrics: Mapping[str, float],
    clinical_targets: str | Path | Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate convergence gate and return machine-readable decision payload."""

    limits = dict(DEFAULT_ITERATION_THRESHOLDS)
    if thresholds is not None:
        limits.update({k: float(v) for k, v in thresholds.items()})

    targets = _clinical_targets_from_input(clinical_targets)
    required = ["mpa_sys", "mpa_dia", "mpa_mean", "rpa_split"]
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"metrics missing required keys: {missing}")

    deltas = {key: abs(float(metrics[key]) - float(targets[key])) for key in required}
    close_to_targets = all(deltas[key] <= limits[key] for key in required)

    return {
        "decision": "converged" if close_to_targets else "not_close",
        "close_to_targets": close_to_targets,
        "thresholds": limits,
        "clinical_targets": targets,
        "metrics": {key: float(metrics[key]) for key in required},
        "deltas": deltas,
        "postop_submission_requested": bool(close_to_targets),
        "regenerated_config_path": None,
    }


def generate_reduced_pa_from_iteration(
    *,
    iteration_dir: str | Path,
    tuned_pa_config: str | Path,
    optimizer: str = "Nelder-Mead",
    nm_iter: int = 5,
    output_name: str = "simplified_zerod_tuned_RRI.json",
    tuning_iter: int = 1,
) -> dict[str, Any]:
    """Run reduced PA regeneration for a completed iteration."""

    sim = SimulationDirectory.from_directory(str(iteration_dir))
    result = sim.optimize_RRI(
        str(tuned_pa_config),
        optimizer=optimizer,
        nm_iter=nm_iter,
        output_name=output_name,
        tuning_iter=tuning_iter,
    )

    output_config = (
        result.get("output_config") if isinstance(result, Mapping) else None
    )
    if output_config is None:
        output_config = str(Path(iteration_dir) / output_name)

    return {
        "regenerated_config_path": str(output_config),
        "regeneration": result,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def write_iteration_metrics(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write iteration_metrics.json payload."""
    return _write_json(path, payload)


def write_iteration_decision(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write iteration_decision.json payload."""
    return _write_json(path, payload)
