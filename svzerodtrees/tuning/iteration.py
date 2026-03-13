"""Utilities for iterative 3D-0D tuning decisions and artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import os

import numpy as np
import pandas as pd

from svzerodtrees.io import ConfigHandler
from svzerodtrees.io.blocks.boundary_condition import (
    validate_boundary_condition_configs,
    validate_impedance_timing_config,
)
from svzerodtrees.microvasculature import TreeParameters
from svzerodtrees.simulation import SimulationDirectory
from svzerodtrees.tune_bcs import (
    ClinicalTargets,
    FixedParam,
    FreeParam,
    ImpedanceTuner,
    TiedParam,
    TuneSpace,
    construct_impedance_trees,
    positive,
    unit_interval,
)


DEFAULT_ITERATION_THRESHOLDS: dict[str, float] = {
    "mpa_sys": 5.0,
    "mpa_dia": 3.0,
    "mpa_mean": 3.0,
    "rpa_split": 0.05,
}

DEFAULT_IMPEDANCE_TUNING_CONFIG: dict[str, Any] = {
    "solver": "Nelder-Mead",
    "nm_iter": 5,
    "n_procs": 24,
    "grid_search_init": True,
    "d_min": 0.01,
    "use_mean": True,
    "specify_diameter": True,
    "rescale_inflow": True,
    "convert_to_cm": False,
    "compliance_model": "olufsen",
}

OPTIMIZED_PARAMS_FILENAME = "optimized_params.csv"
OPTIMIZATION_LOG_FILENAME = "stree_impedance_optimization.log"
PA_CONFIG_SNAPSHOT_FILENAME = "pa_config_tuning_snapshot.json"
TUNED_ZEROD_CONFIG_FILENAME = "svzerod_3d_coupling_tuned.json"


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_impedance_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config is None or config.get("tune_space") is None:
        raise ValueError(
            "impedance tuning config must include explicit tune_space with keys "
            "free, fixed, and tied"
        )

    merged = dict(DEFAULT_IMPEDANCE_TUNING_CONFIG)
    for key, value in config.items():
        if value is None:
            continue
        if key == "tune_space":
            merged["tune_space"] = value
        else:
            merged[key] = value

    solver = str(merged.get("solver", "")).strip()
    if not solver:
        raise ValueError("impedance tuning solver cannot be empty")
    merged["solver"] = solver

    merged["nm_iter"] = int(merged["nm_iter"])
    merged["n_procs"] = int(merged["n_procs"])
    merged["d_min"] = float(merged["d_min"])
    merged["grid_search_init"] = bool(merged["grid_search_init"])
    merged["use_mean"] = bool(merged["use_mean"])
    merged["specify_diameter"] = bool(merged["specify_diameter"])
    merged["rescale_inflow"] = bool(merged["rescale_inflow"])
    merged["convert_to_cm"] = bool(merged["convert_to_cm"])
    merged["compliance_model"] = str(merged["compliance_model"]).strip().lower()
    merged["tune_space"] = _normalize_tune_space_config(merged.get("tune_space"))

    if merged["nm_iter"] <= 0:
        raise ValueError("impedance tuning nm_iter must be > 0")
    if merged["n_procs"] <= 0:
        raise ValueError("impedance tuning n_procs must be > 0")
    if merged["d_min"] <= 0.0:
        raise ValueError("impedance tuning d_min must be > 0")
    if merged["compliance_model"] not in {"constant", "olufsen"}:
        raise ValueError("impedance tuning compliance_model must be constant or olufsen")

    return merged


def _logit(value: float) -> float:
    numeric = float(value)
    if not (0.0 < numeric < 1.0):
        raise ValueError("logit transform requires 0 < x < 1")
    return float(np.log(numeric / (1.0 - numeric)))


def _parse_bound(value: float | int | str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    token = str(value).strip().lower()
    if token in {"inf", "+inf"}:
        return float("inf")
    if token == "-inf":
        return float("-inf")
    raise ValueError("bound strings must be one of inf, +inf, -inf")


def _name_list(items: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return [str(item.get(key, "")).strip() for item in items]


def _validate_unique_names(
    items: Sequence[Mapping[str, Any]],
    *,
    key: str,
    label: str,
) -> None:
    names = _name_list(items, key)
    duplicates = sorted({name for name in names if names.count(name) > 1 and name})
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {', '.join(duplicates)}")


def _normalize_tune_space_config(tune_space: Any) -> dict[str, list[dict[str, Any]]]:
    if tune_space is None:
        raise ValueError(
            "impedance tune_space is required and must be a mapping with keys "
            "free, fixed, and tied"
        )
    if not isinstance(tune_space, Mapping):
        raise ValueError("impedance tune_space must be a mapping")
    missing_keys = [key for key in ("free", "fixed", "tied") if key not in tune_space]
    if missing_keys:
        raise ValueError(
            "impedance tune_space must define keys: free, fixed, tied"
        )

    free_raw = tune_space.get("free", [])
    fixed_raw = tune_space.get("fixed", [])
    tied_raw = tune_space.get("tied", [])
    if not isinstance(free_raw, list) or not isinstance(fixed_raw, list) or not isinstance(tied_raw, list):
        raise ValueError("tune_space.free, tune_space.fixed, and tune_space.tied must be lists")
    if not free_raw:
        raise ValueError("tune_space.free cannot be empty")

    normalized_free: list[dict[str, Any]] = []
    normalized_fixed: list[dict[str, Any]] = []
    normalized_tied: list[dict[str, Any]] = []

    for entry in free_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("tune_space.free entries must be mappings")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("tune_space.free entries require non-empty name")
        to_native = str(entry.get("to_native", "identity")).strip().lower()
        from_native = str(entry.get("from_native", "identity")).strip().lower()
        if to_native not in {"identity", "positive", "unit_interval"}:
            raise ValueError(
                f"unsupported to_native transform '{to_native}' for free param '{name}'"
            )
        if from_native not in {"identity", "log", "logit"}:
            raise ValueError(
                f"unsupported from_native transform '{from_native}' for free param '{name}'"
            )
        lb = _parse_bound(entry.get("lb"))
        ub = _parse_bound(entry.get("ub"))
        if lb >= ub:
            raise ValueError(f"free param '{name}' must satisfy lb < ub")
        normalized_free.append(
            {
                "name": name,
                "init": float(entry.get("init")),
                "lb": lb,
                "ub": ub,
                "to_native": to_native,
                "from_native": from_native,
            }
        )

    for entry in fixed_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("tune_space.fixed entries must be mappings")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("tune_space.fixed entries require non-empty name")
        normalized_fixed.append(
            {
                "name": name,
                "value": float(entry.get("value")),
            }
        )

    for entry in tied_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("tune_space.tied entries must be mappings")
        name = str(entry.get("name", "")).strip()
        other = str(entry.get("other", "")).strip()
        if not name or not other:
            raise ValueError("tune_space.tied entries require non-empty name and other")
        fn = str(entry.get("fn", "identity")).strip().lower()
        if fn != "identity":
            raise ValueError(
                f"unsupported tied fn '{fn}' for tied param '{name}'; only identity is supported"
            )
        normalized_tied.append({"name": name, "other": other, "fn": fn})

    _validate_unique_names(normalized_free, key="name", label="tune_space.free")
    _validate_unique_names(normalized_fixed, key="name", label="tune_space.fixed")
    _validate_unique_names(normalized_tied, key="name", label="tune_space.tied")

    return {
        "free": normalized_free,
        "fixed": normalized_fixed,
        "tied": normalized_tied,
    }


def _resolve_to_native(name: str) -> Callable[[float], float]:
    mapping: dict[str, Callable[[float], float]] = {
        "identity": lambda value: float(value),
        "positive": positive,
        "unit_interval": unit_interval,
    }
    if name not in mapping:
        raise ValueError(f"unsupported to_native transform '{name}'")
    return mapping[name]


def _resolve_from_native(name: str) -> Callable[[float], float]:
    mapping: dict[str, Callable[[float], float]] = {
        "identity": lambda value: float(value),
        "log": np.log,
        "logit": _logit,
    }
    if name not in mapping:
        raise ValueError(f"unsupported from_native transform '{name}'")
    return mapping[name]


def _build_tune_space_from_config(tune_space_cfg: Mapping[str, Any]) -> TuneSpace:
    free_entries = tune_space_cfg.get("free", [])
    fixed_entries = tune_space_cfg.get("fixed", [])
    tied_entries = tune_space_cfg.get("tied", [])

    free = [
        FreeParam(
            str(entry["name"]),
            init=float(entry["init"]),
            lb=float(entry["lb"]),
            ub=float(entry["ub"]),
            to_native=_resolve_to_native(str(entry.get("to_native", "identity"))),
            from_native=_resolve_from_native(str(entry.get("from_native", "identity"))),
        )
        for entry in free_entries
    ]
    fixed = [
        FixedParam(str(entry["name"]), float(entry["value"]))
        for entry in fixed_entries
    ]
    tied = [
        TiedParam(
            str(entry["name"]),
            str(entry["other"]),
            fn=(lambda value: float(value)),
        )
        for entry in tied_entries
    ]
    return TuneSpace(free=free, fixed=fixed, tied=tied)


def _required_xi_pa_labels(tune_space_cfg: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for section in ("free", "fixed", "tied"):
        for entry in tune_space_cfg.get(section, []):
            names.add(str(entry.get("name", "")).strip().lower())
    required: set[str] = set()
    if "lpa.xi" in names:
        required.add("lpa")
    if "rpa.xi" in names:
        required.add("rpa")
    return required


def _validate_required_xi_in_optimized_csv(
    *,
    optimized_csv: Path,
    required_pa: set[str],
) -> None:
    if not required_pa:
        return
    opt_params = pd.read_csv(optimized_csv)
    if "pa" not in opt_params.columns:
        raise ValueError("optimized_params.csv must include a 'pa' column")
    if "xi" not in opt_params.columns:
        required = ", ".join(f"{pa}.xi" for pa in sorted(required_pa))
        raise ValueError(
            "optimized_params.csv must include an 'xi' column because tune_space "
            f"includes {required}"
        )
    for pa in sorted(required_pa):
        rows = opt_params[opt_params["pa"].astype(str).str.lower() == pa]
        if rows.empty:
            raise ValueError(
                f"optimized_params.csv must include a row for pa={pa} "
                f"because tune_space includes {pa}.xi"
            )
        xi_values = pd.to_numeric(rows["xi"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(xi_values)
        if not finite_mask.any():
            raise ValueError(
                f"optimized_params.csv must include a finite xi value for pa={pa} "
                f"because tune_space includes {pa}.xi"
            )


def _load_tree_params(opt_csv: Path) -> tuple[TreeParameters, TreeParameters]:
    opt_params = pd.read_csv(opt_csv)
    if "pa" not in opt_params.columns:
        raise ValueError("optimized_params.csv must include a 'pa' column")
    lpa_rows = opt_params[opt_params["pa"].astype(str).str.lower() == "lpa"]
    rpa_rows = opt_params[opt_params["pa"].astype(str).str.lower() == "rpa"]
    if lpa_rows.empty or rpa_rows.empty:
        raise ValueError(
            "optimized_params.csv must include one row each for pa=lpa and pa=rpa"
        )
    return TreeParameters.from_row(lpa_rows), TreeParameters.from_row(rpa_rows)


def _validate_impedance_artifact(path: Path) -> None:
    with path.open(encoding="utf-8") as ff:
        payload = json.load(ff)
    validate_boundary_condition_configs(payload.get("boundary_conditions", []))
    validate_impedance_timing_config(payload)


def run_impedance_tuning_for_iteration(
    *,
    iteration_dir: str | Path,
    seed_config: str | Path,
    mesh_surfaces: str | Path,
    clinical_targets: str | Path,
    impedance_config: Mapping[str, Any] | None = None,
    results_dir: str | Path | None = None,
    tuned_config_name: str = TUNED_ZEROD_CONFIG_FILENAME,
) -> dict[str, Any]:
    """Run impedance tuning + BC tree assignment for one tuning iteration."""

    iteration_path = Path(iteration_dir)
    output_dir = Path(results_dir) if results_dir is not None else iteration_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_config_path = Path(seed_config)
    mesh_surfaces_path = Path(mesh_surfaces)
    targets_path = Path(clinical_targets)

    if not seed_config_path.exists():
        raise FileNotFoundError(f"seed 0D config not found: {seed_config_path}")
    if not mesh_surfaces_path.exists():
        raise FileNotFoundError(f"mesh-surfaces path not found: {mesh_surfaces_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"clinical targets not found: {targets_path}")

    tuning = _resolve_impedance_config(impedance_config)
    required_xi_pa = _required_xi_pa_labels(tuning["tune_space"])
    targets = ClinicalTargets.from_csv(str(targets_path))
    tune_space = _build_tune_space_from_config(tuning["tune_space"])

    opt_log = output_dir / OPTIMIZATION_LOG_FILENAME
    with _pushd(output_dir):
        reduced_config = ConfigHandler.from_json(str(seed_config_path), is_pulmonary=True)
        tuner = ImpedanceTuner(
            reduced_config,
            str(mesh_surfaces_path),
            targets,
            tune_space,
            compliance_model=str(tuning["compliance_model"]),
            grid_search_init=bool(tuning["grid_search_init"]),
            rescale_inflow=bool(tuning["rescale_inflow"]),
            n_procs=int(tuning["n_procs"]),
            convert_to_cm=bool(tuning["convert_to_cm"]),
            log_file=str(opt_log),
            solver=str(tuning["solver"]),
        )
        tuner.tune(nm_iter=int(tuning["nm_iter"]))

    optimized_csv = output_dir / OPTIMIZED_PARAMS_FILENAME
    pa_snapshot = output_dir / PA_CONFIG_SNAPSHOT_FILENAME
    if not optimized_csv.exists():
        raise FileNotFoundError(
            f"impedance tuning did not produce {OPTIMIZED_PARAMS_FILENAME}"
        )
    if not pa_snapshot.exists():
        raise FileNotFoundError(
            f"impedance tuning did not produce {PA_CONFIG_SNAPSHOT_FILENAME}"
        )
    _validate_impedance_artifact(pa_snapshot)

    _validate_required_xi_in_optimized_csv(
        optimized_csv=optimized_csv,
        required_pa=required_xi_pa,
    )
    lpa_params, rpa_params = _load_tree_params(optimized_csv)
    tuned_config = ConfigHandler.from_json(str(seed_config_path), is_pulmonary=True)
    construct_impedance_trees(
        tuned_config,
        str(mesh_surfaces_path),
        targets.wedge_p,
        lpa_params,
        rpa_params,
        d_min=float(tuning["d_min"]),
        convert_to_cm=bool(tuning["convert_to_cm"]),
        n_procs=int(tuning["n_procs"]),
        use_mean=bool(tuning["use_mean"]),
        specify_diameter=bool(tuning["specify_diameter"]),
    )

    tuned_zerod_config = output_dir / tuned_config_name
    tuned_config.to_json(str(tuned_zerod_config))
    _validate_impedance_artifact(tuned_zerod_config)

    return {
        "optimized_params_csv": str(optimized_csv),
        "stree_optimization_log": str(opt_log),
        "pa_config_snapshot": str(pa_snapshot),
        "tuned_zerod_config": str(tuned_zerod_config),
        "impedance_config": tuning,
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
