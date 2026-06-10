"""Local reduced-PA adaptation benchmark harness."""

from __future__ import annotations

import copy
import csv
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from ..config import AdaptBenchmarkConfig, AdaptBenchmarkScenarioConfig
from .integrator import run_adaptation
from .models.cwss import CWSSAdaptation
from .models.cwss_ims import CWSSIMSAdaptation
from .setup import initialize_from_paths
from .utils import pack_state, rel_change
from .workflow import _apply_territory_homeostatic_update


def _require_existing_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"{label} must be a file: {candidate}")
    return candidate


def _merged_parameters(
    global_overrides: Dict[str, Dict[str, Any]] | None,
    scenario_overrides: Dict[str, Dict[str, Any]] | None,
    model: str,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if global_overrides is not None:
        merged.update(global_overrides.get(model, {}))
    if scenario_overrides is not None:
        merged.update(scenario_overrides.get(model, {}))
    return merged


def _branch_total_resistance(pa_config, branch: str) -> float:
    branch_lower = str(branch).lower()
    if branch_lower == "lpa":
        return float(pa_config.lpa_prox.R) + float(pa_config.bcs["LPA_BC"].R)
    if branch_lower == "rpa":
        return float(pa_config.rpa_prox.R) + float(pa_config.bcs["RPA_BC"].R)
    raise ValueError(f"unsupported branch '{branch}'")


def _ensure_simulated(pa_config) -> None:
    if getattr(pa_config, "result", None) is None:
        pa_config.simulate()


def _reduced_pa_stage_metrics(pa_config) -> Dict[str, float]:
    _ensure_simulated(pa_config)
    lpa_result = pa_config.result[pa_config.result.name == "branch2_seg0"]
    rpa_result = pa_config.result[pa_config.result.name == "branch4_seg0"]
    lpa_flow = float(np.mean(lpa_result["flow_out"]))
    rpa_flow = float(np.mean(rpa_result["flow_out"]))
    total_flow = lpa_flow + rpa_flow
    return {
        "lpa_flow": lpa_flow,
        "rpa_flow": rpa_flow,
        "lpa_split": float(lpa_flow / total_flow) if abs(total_flow) > 1e-12 else 0.0,
        "rpa_split": float(rpa_flow / total_flow) if abs(total_flow) > 1e-12 else 0.0,
        "lpa_resistance": _branch_total_resistance(pa_config, "lpa"),
        "rpa_resistance": _branch_total_resistance(pa_config, "rpa"),
        "mpa_sys": float(pa_config.P_mpa[0]),
        "mpa_dia": float(pa_config.P_mpa[1]),
        "mpa_mean": float(pa_config.P_mpa[2]),
    }


def _target_metrics_from_pa(pa_config) -> Dict[str, float | None]:
    targets = pa_config.clinical_targets
    mpa_p = getattr(targets, "mpa_p", None)
    return {
        "rpa_split": float(getattr(targets, "rpa_split", np.nan)),
        "mpa_sys": float(mpa_p[0]) if mpa_p is not None else None,
        "mpa_dia": float(mpa_p[1]) if mpa_p is not None else None,
        "mpa_mean": float(mpa_p[2]) if mpa_p is not None else None,
        "wedge_p": float(getattr(targets, "wedge_p", np.nan)),
    }


def _build_postop_baseline(preop_pa, postop_pa):
    baseline = copy.deepcopy(postop_pa)
    baseline.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    baseline.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)
    baseline.update_bcs()
    baseline.simulate()
    return baseline


def _augment_dynamic_solver_metrics(
    result: Dict[str, Any],
    *,
    model: str,
    parameter_set: Dict[str, Any],
    preop_config_path: str,
    postop_config_path: str,
    clinical_targets_csv: str,
    tree_params_csv: str,
    reduced_order_pa: str,
) -> Dict[str, Any]:
    enriched = dict(result)
    enriched["solver_t_end"] = float(parameter_set.get("t_end", 3600.0))
    enriched["solver_rtol"] = float(parameter_set.get("rtol", 1e-6))
    enriched["solver_atol"] = float(parameter_set.get("atol", 1e-7))
    enriched["solver_max_step"] = float(parameter_set.get("max_step", 60.0))
    enriched["solver_method"] = str(parameter_set.get("solver_method", "RK23"))
    enriched["tree_max_nodes"] = int(parameter_set.get("max_nodes", 100_000))
    enriched["requested_iterations_input"] = int(parameter_set.get("iterations", 1))
    enriched["config_paths"] = {
        "preop_rri_config": str(preop_config_path),
        "postop_rri_config": str(postop_config_path),
        "clinical_targets_csv": str(clinical_targets_csv),
        "tree_params_csv": str(tree_params_csv),
        "reduced_order_pa": str(reduced_order_pa),
    }
    if "terminal_resistance" in parameter_set:
        enriched["terminal_resistance"] = float(parameter_set.get("terminal_resistance") or 0.0)
    if "terminal_load_policy" in parameter_set:
        enriched["terminal_load_policy"] = str(parameter_set.get("terminal_load_policy") or "")
    if "benchmark_case" in parameter_set:
        enriched["benchmark_case"] = str(parameter_set.get("benchmark_case") or "")
    if model == "M1":
        enriched["wss_gain"] = float(parameter_set.get("wss_gain", 0.01))
    if model == "M3":
        enriched["k_arr"] = [
            float(value)
            for value in parameter_set.get("k_arr", [1.0, 1.0, 1.0, 1.0])
        ]
    return enriched


def _tree_size_metrics(preop_pa, *, max_nodes: int) -> Dict[str, Any]:
    lpa_nodes = int(preop_pa.lpa_tree.store.n_nodes())
    rpa_nodes = int(preop_pa.rpa_tree.store.n_nodes())
    return {
        "tree_max_nodes": int(max_nodes),
        "lpa_tree_nodes": lpa_nodes,
        "rpa_tree_nodes": rpa_nodes,
        "lpa_tree_max_nodes_reached": int(lpa_nodes >= int(max_nodes)),
        "rpa_tree_max_nodes_reached": int(rpa_nodes >= int(max_nodes)),
    }


def _run_m2_algebraic_update(preop_pa, postop_baseline_pa, parameter_set: Dict[str, Any]):
    adapted = copy.deepcopy(postop_baseline_pa)
    y_initial = pack_state(adapted.lpa_tree, adapted.rpa_tree)

    preop_metrics = _reduced_pa_stage_metrics(preop_pa)
    postop_metrics = _reduced_pa_stage_metrics(postop_baseline_pa)
    territory_metrics = {
        "lpa": {
            "preop_flow": preop_metrics["lpa_flow"],
            "postop_flow": postop_metrics["lpa_flow"],
            "preop_resistance": preop_metrics["lpa_resistance"],
            "postop_resistance": postop_metrics["lpa_resistance"],
        },
        "rpa": {
            "preop_flow": preop_metrics["rpa_flow"],
            "postop_flow": postop_metrics["rpa_flow"],
            "preop_resistance": preop_metrics["rpa_resistance"],
            "postop_resistance": postop_metrics["rpa_resistance"],
        },
    }

    territory_metrics["lpa"].update(
        _apply_territory_homeostatic_update(
            adapted.lpa_tree,
            preop_flow=preop_metrics["lpa_flow"],
            postop_flow=postop_metrics["lpa_flow"],
            preop_resistance=preop_metrics["lpa_resistance"],
            postop_resistance=postop_metrics["lpa_resistance"],
            iterations=int(parameter_set.get("iterations", 1)),
            wss_gain=float(parameter_set.get("wss_gain", 1.0)),
            ims_gain=float(parameter_set.get("ims_gain", 1.0)),
            compliance_gain=float(parameter_set.get("compliance_gain", 1.0)),
        )
    )
    territory_metrics["rpa"].update(
        _apply_territory_homeostatic_update(
            adapted.rpa_tree,
            preop_flow=preop_metrics["rpa_flow"],
            postop_flow=postop_metrics["rpa_flow"],
            preop_resistance=preop_metrics["rpa_resistance"],
            postop_resistance=postop_metrics["rpa_resistance"],
            iterations=int(parameter_set.get("iterations", 1)),
            wss_gain=float(parameter_set.get("wss_gain", 1.0)),
            ims_gain=float(parameter_set.get("ims_gain", 1.0)),
            compliance_gain=float(parameter_set.get("compliance_gain", 1.0)),
        )
    )
    adapted.update_bcs()
    adapted.simulate()
    y_final = pack_state(adapted.lpa_tree, adapted.rpa_tree)
    solver_metrics = {
        "stable": 1,
        "geom_err": float(rel_change(y_final, y_initial)),
        "t95": None,
        "n_rhs": None,
        "preop_rpa_split": float(preop_metrics["rpa_split"]),
        "postop_rpa_split": float(postop_metrics["rpa_split"]),
        "final_rpa_split": float(adapted.rpa_split),
        "tree_max_nodes": int(parameter_set.get("max_nodes", 100_000)),
        "requested_iterations_input": int(parameter_set.get("iterations", 1)),
        "update_mode": "algebraic",
        "solver_diagnostics": {
            "termination_reason": None,
            "event_time": None,
            "flow_split_history": [],
            "accepted_step_flow_split_history": [],
            "accepted_step_convergence_history": [],
        },
        "territory_metrics": territory_metrics,
        "parameter_set": json.loads(json.dumps(parameter_set, sort_keys=True)),
    }
    return adapted, solver_metrics


def _plot_flow_split_history(
    history: Iterable[Dict[str, Any]],
    *,
    output_png: Path,
    title: str,
) -> None:
    rows = list(history)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [float(item["t"]) for item in rows],
        [float(item["rpa_split"]) for item in rows],
        color="tab:blue",
        linewidth=1.8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPA split")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_flow_split_history_csv(
    history: Iterable[Dict[str, Any]],
    *,
    output_csv: Path,
) -> None:
    rows = list(history)
    if not rows:
        return
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "rpa_split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time_s": float(row["t"]),
                    "rpa_split": float(row["rpa_split"]),
                }
            )


def _normalized_row(
    *,
    scenario: str,
    model: str,
    summary_status: str,
    update_mode: str,
    solver_metrics: Dict[str, Any],
    parameter_set: Dict[str, Any] | None = None,
    patient_id: str | None = None,
    scenario_group: str | None = None,
    perturbation_severity: str | None = None,
) -> Dict[str, Any]:
    diagnostics = solver_metrics.get("solver_diagnostics") or {}
    parameter_set = parameter_set or {}
    k_arr = solver_metrics.get("k_arr") or parameter_set.get("k_arr") or [None, None, None, None]
    k_arr = list(k_arr) + [None, None, None, None]
    radius_change = diagnostics.get("radius_change") or {}

    def _nested_float(section: str, key: str):
        values = radius_change.get(section) or {}
        value = values.get(key)
        return None if value is None else float(value)

    row = {
        "case_name": str(parameter_set.get("benchmark_case") or scenario),
        "patient_id": patient_id,
        "scenario_group": scenario_group,
        "perturbation_severity": perturbation_severity,
        "scenario": scenario,
        "model": model,
        "status": summary_status,
        "update_mode": update_mode,
        "terminal_load_policy": parameter_set.get("terminal_load_policy"),
        "terminal_resistance": parameter_set.get("terminal_resistance"),
        "wss_gain": parameter_set.get("wss_gain"),
        "k_tau_r": k_arr[0],
        "k_sig_r": k_arr[1],
        "k_tau_h": k_arr[2],
        "k_sig_h": k_arr[3],
        "stable": solver_metrics.get("stable"),
        "final_rpa_split": solver_metrics.get("final_rpa_split"),
        "final_lpa_split": (
            None
            if solver_metrics.get("final_rpa_split") is None
            else 1.0 - float(solver_metrics["final_rpa_split"])
        ),
        "preop_rpa_split": solver_metrics.get("preop_rpa_split"),
        "postop_rpa_split": solver_metrics.get("postop_rpa_split"),
        "geom_err": solver_metrics.get("geom_err"),
        "t95": solver_metrics.get("t95"),
        "n_rhs": solver_metrics.get("n_rhs"),
        "tree_max_nodes": solver_metrics.get("tree_max_nodes"),
        "lpa_tree_nodes": solver_metrics.get("lpa_tree_nodes"),
        "rpa_tree_nodes": solver_metrics.get("rpa_tree_nodes"),
        "lpa_tree_max_nodes_reached": solver_metrics.get("lpa_tree_max_nodes_reached"),
        "rpa_tree_max_nodes_reached": solver_metrics.get("rpa_tree_max_nodes_reached"),
        "termination_reason": diagnostics.get("termination_reason"),
        "event_time": diagnostics.get("event_time"),
        "lpa_radius_mean_relative_change": _nested_float("lpa_radius", "mean_relative_change"),
        "lpa_radius_max_abs_relative_change": _nested_float("lpa_radius", "max_abs_relative_change"),
        "rpa_radius_mean_relative_change": _nested_float("rpa_radius", "mean_relative_change"),
        "rpa_radius_max_abs_relative_change": _nested_float("rpa_radius", "max_abs_relative_change"),
        "lpa_thickness_mean_relative_change": _nested_float("lpa_thickness", "mean_relative_change"),
        "lpa_thickness_max_abs_relative_change": _nested_float("lpa_thickness", "max_abs_relative_change"),
        "rpa_thickness_mean_relative_change": _nested_float("rpa_thickness", "mean_relative_change"),
        "rpa_thickness_max_abs_relative_change": _nested_float("rpa_thickness", "max_abs_relative_change"),
    }
    row.update(_screen_benchmark_row(row, parameter_set=parameter_set))
    return row


def _as_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _screen_benchmark_row(
    row: Dict[str, Any],
    *,
    parameter_set: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parameter_set = parameter_set or {}
    split_floor = float(parameter_set.get("collapse_split_floor", 0.01))
    split_ceiling = float(parameter_set.get("collapse_split_ceiling", 0.99))
    radius_limit = float(parameter_set.get("radius_max_abs_relative_change_limit", 10.0))
    thickness_limit = float(parameter_set.get("thickness_max_abs_relative_change_limit", 10.0))

    final_split = _as_float_or_none(row.get("final_rpa_split"))
    one_branch_collapse = (
        final_split is not None
        and (final_split <= split_floor or final_split >= split_ceiling)
    )

    radius_values = [
        _as_float_or_none(row.get("lpa_radius_max_abs_relative_change")),
        _as_float_or_none(row.get("rpa_radius_max_abs_relative_change")),
    ]
    thickness_values = [
        _as_float_or_none(row.get("lpa_thickness_max_abs_relative_change")),
        _as_float_or_none(row.get("rpa_thickness_max_abs_relative_change")),
    ]
    radius_bounds_violation = any(
        value is not None and value > radius_limit for value in radius_values
    )
    thickness_bounds_violation = any(
        value is not None and value > thickness_limit for value in thickness_values
    )

    numeric_keys = [
        "terminal_resistance",
        "wss_gain",
        "k_tau_r",
        "k_sig_r",
        "k_tau_h",
        "k_sig_h",
        "final_rpa_split",
        "final_lpa_split",
        "preop_rpa_split",
        "postop_rpa_split",
        "geom_err",
        "t95",
        "event_time",
        "lpa_radius_mean_relative_change",
        "lpa_radius_max_abs_relative_change",
        "rpa_radius_mean_relative_change",
        "rpa_radius_max_abs_relative_change",
        "lpa_thickness_mean_relative_change",
        "lpa_thickness_max_abs_relative_change",
        "rpa_thickness_mean_relative_change",
        "rpa_thickness_max_abs_relative_change",
    ]
    nonfinite_state_detected = False
    for key in numeric_keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            if not np.isfinite(float(value)):
                nonfinite_state_detected = True
                break
        except (TypeError, ValueError):
            continue

    terminal_policy = str(row.get("terminal_load_policy") or "none")
    terminal_resistance = _as_float_or_none(row.get("terminal_resistance"))
    nonphysical_terminal_load = (
        terminal_policy == "explicit_terminal_resistance"
        and (terminal_resistance is None or terminal_resistance < 0.0)
    )

    stability_screen_failed = (
        row.get("status") != "ok"
        or one_branch_collapse
        or radius_bounds_violation
        or thickness_bounds_violation
        or nonfinite_state_detected
        or nonphysical_terminal_load
    )
    return {
        "collapse_split_floor": split_floor,
        "collapse_split_ceiling": split_ceiling,
        "radius_max_abs_relative_change_limit": radius_limit,
        "thickness_max_abs_relative_change_limit": thickness_limit,
        "one_branch_collapse": int(one_branch_collapse),
        "radius_bounds_violation": int(radius_bounds_violation),
        "thickness_bounds_violation": int(thickness_bounds_violation),
        "nonfinite_state_detected": int(nonfinite_state_detected),
        "nonphysical_terminal_load": int(nonphysical_terminal_load),
        "stability_screen_failed": int(stability_screen_failed),
    }


def _run_single_model(
    *,
    scenario_name: str,
    model: str,
    parameter_set: Dict[str, Any],
    preop_config_path: str,
    postop_config_path: str,
    tree_params_csv: str,
    clinical_targets_csv: str,
    output_dir: Path,
    patient_id: str | None = None,
    scenario_group: str | None = None,
    perturbation_severity: str | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_nodes = int(parameter_set.get("max_nodes", 100_000))
    preop_pa, postop_pa = initialize_from_paths(
        preop_config_path,
        postop_config_path,
        tree_params_csv,
        clinical_targets_csv,
        max_nodes=max_nodes,
    )
    preop_metrics = _reduced_pa_stage_metrics(preop_pa)
    postop_baseline = _build_postop_baseline(preop_pa, postop_pa)
    postop_metrics = _reduced_pa_stage_metrics(postop_baseline)
    target_metrics = _target_metrics_from_pa(preop_pa)
    tree_size_metrics = _tree_size_metrics(preop_pa, max_nodes=max_nodes)

    artifacts: Dict[str, str] = {}
    update_mode = "dynamic"
    history = []
    if model == "M1":
        terminal_resistance = float(parameter_set.get("terminal_resistance", 0.0))
        preop_pa.lpa_tree.terminal_resistance = terminal_resistance
        preop_pa.rpa_tree.terminal_resistance = terminal_resistance
        result, flow_log, sol, adapted_pa, hists = run_adaptation(
            preop_pa,
            postop_pa,
            CWSSAdaptation,
            [float(parameter_set.get("wss_gain", 0.01)), 0.0, 0.0, 0.0],
            t_end=float(parameter_set.get("t_end", 3600.0)),
            rtol=float(parameter_set.get("rtol", 1e-6)),
            atol=float(parameter_set.get("atol", 1e-7)),
            max_step=float(parameter_set.get("max_step", 60.0)),
            method=str(parameter_set.get("solver_method", "RK23")),
            collapse_split_floor=float(parameter_set.get("collapse_split_floor", 0.01)),
            collapse_split_ceiling=float(parameter_set.get("collapse_split_ceiling", 0.99)),
            radius_max_abs_relative_change_limit=float(
                parameter_set.get("radius_max_abs_relative_change_limit", 10.0)
            ),
            thickness_max_abs_relative_change_limit=float(
                parameter_set.get("thickness_max_abs_relative_change_limit", 10.0)
            ),
        )
        solver_metrics = _augment_dynamic_solver_metrics(
            result,
            model=model,
            parameter_set=parameter_set,
            preop_config_path=preop_config_path,
            postop_config_path=postop_config_path,
            clinical_targets_csv=clinical_targets_csv,
            tree_params_csv=tree_params_csv,
            reduced_order_pa=preop_config_path,
        )
        solver_metrics.update(tree_size_metrics)
        history = list(
            (solver_metrics.get("solver_diagnostics") or {}).get(
                "accepted_step_flow_split_history", flow_log
            )
        )
        if hists:
            histories_png = output_dir / "adaptation_histories.png"
            hists[0].savefig(histories_png, dpi=200, bbox_inches="tight")
            plt.close(hists[0])
            artifacts["adaptation_histories_png"] = str(histories_png)
    elif model == "M3":
        if "terminal_resistance" in parameter_set:
            terminal_resistance = float(parameter_set.get("terminal_resistance") or 0.0)
            preop_pa.lpa_tree.terminal_resistance = terminal_resistance
            preop_pa.rpa_tree.terminal_resistance = terminal_resistance
        result, flow_log, sol, adapted_pa, hists = run_adaptation(
            preop_pa,
            postop_pa,
            CWSSIMSAdaptation,
            [float(value) for value in parameter_set.get("k_arr", [1.0, 1.0, 1.0, 1.0])],
            t_end=float(parameter_set.get("t_end", 3600.0)),
            rtol=float(parameter_set.get("rtol", 1e-6)),
            atol=float(parameter_set.get("atol", 1e-7)),
            max_step=float(parameter_set.get("max_step", 60.0)),
            method=str(parameter_set.get("solver_method", "RK23")),
            collapse_split_floor=float(parameter_set.get("collapse_split_floor", 0.01)),
            collapse_split_ceiling=float(parameter_set.get("collapse_split_ceiling", 0.99)),
            radius_max_abs_relative_change_limit=float(
                parameter_set.get("radius_max_abs_relative_change_limit", 10.0)
            ),
            thickness_max_abs_relative_change_limit=float(
                parameter_set.get("thickness_max_abs_relative_change_limit", 10.0)
            ),
        )
        solver_metrics = _augment_dynamic_solver_metrics(
            result,
            model=model,
            parameter_set=parameter_set,
            preop_config_path=preop_config_path,
            postop_config_path=postop_config_path,
            clinical_targets_csv=clinical_targets_csv,
            tree_params_csv=tree_params_csv,
            reduced_order_pa=preop_config_path,
        )
        solver_metrics.update(tree_size_metrics)
        history = list(
            (solver_metrics.get("solver_diagnostics") or {}).get(
                "accepted_step_flow_split_history", flow_log
            )
        )
        if hists:
            histories_png = output_dir / "adaptation_histories.png"
            hists[0].savefig(histories_png, dpi=200, bbox_inches="tight")
            plt.close(hists[0])
            artifacts["adaptation_histories_png"] = str(histories_png)
    elif model == "M2":
        update_mode = "algebraic"
        adapted_pa, solver_metrics = _run_m2_algebraic_update(
            preop_pa,
            postop_baseline,
            parameter_set,
        )
        solver_metrics.update(tree_size_metrics)
    else:
        raise ValueError(f"unsupported benchmark model '{model}'")

    adapted_metrics = _reduced_pa_stage_metrics(adapted_pa)
    if history:
        history_png = output_dir / "flow_split_history.png"
        history_csv = output_dir / "flow_split_history.csv"
        _plot_flow_split_history(
            history,
            output_png=history_png,
            title=f"{scenario_name} {model} flow-split history",
        )
        _write_flow_split_history_csv(history, output_csv=history_csv)
        artifacts["flow_split_history_png"] = str(history_png)
        artifacts["flow_split_history_csv"] = str(history_csv)

    normalized = _normalized_row(
        scenario=scenario_name,
        model=model,
        summary_status="ok",
        update_mode=update_mode,
        solver_metrics=solver_metrics,
        parameter_set=parameter_set,
        patient_id=patient_id,
        scenario_group=scenario_group,
        perturbation_severity=perturbation_severity,
    )
    summary = {
        "status": "ok",
        "scenario": scenario_name,
        "model": model,
        "update_mode": update_mode,
        "parameter_provenance": json.loads(json.dumps(parameter_set, sort_keys=True)),
        "artifacts": dict(artifacts),
        "normalized_metrics": normalized,
        "hemodynamics": {
            "reduced_zerod": {
                "preop": preop_metrics,
                "postop_initial": postop_metrics,
                "adapted_final": adapted_metrics,
                "target": target_metrics,
            }
        },
        "solver_metrics": solver_metrics,
    }
    metrics = {
        "scenario": scenario_name,
        "model": model,
        "update_mode": update_mode,
        "normalized_metrics": normalized,
        "hemodynamics": summary["hemodynamics"],
        "solver_metrics": solver_metrics,
    }
    summary_path = output_dir / "adaptation_summary.json"
    metrics_path = output_dir / "adaptation_metrics.json"
    summary["artifacts"]["adaptation_summary_json"] = str(summary_path)
    summary["artifacts"]["adaptation_metrics_json"] = str(metrics_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_reduced_pa_adaptation_case(
    *,
    scenario_name: str,
    model: str,
    parameter_set: Dict[str, Any] | None,
    preop_rri_config: str,
    postop_rri_config: str,
    tree_params_csv: str,
    clinical_targets_csv: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    return _run_single_model(
        scenario_name=scenario_name,
        model=str(model).upper(),
        parameter_set=dict(parameter_set or {}),
        preop_config_path=str(_require_existing_file(preop_rri_config, label="preop_rri_config")),
        postop_config_path=str(_require_existing_file(postop_rri_config, label="postop_rri_config")),
        tree_params_csv=str(_require_existing_file(tree_params_csv, label="tree_params_csv")),
        clinical_targets_csv=str(_require_existing_file(clinical_targets_csv, label="clinical_targets_csv")),
        output_dir=Path(output_dir).expanduser().resolve(),
    )


def _plot_stability_convergence(rows: List[Dict[str, Any]], output_png: Path) -> None:
    if not rows:
        return
    labels = [f"{row['scenario']}:{row['model']}" for row in rows]
    stable = [float(row["stable"] or 0.0) for row in rows]
    event_times = [
        np.nan if row.get("event_time") in (None, "") else float(row["event_time"])
        for row in rows
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.bar(labels, stable, color="tab:green")
    ax1.set_ylabel("Stable")
    ax1.set_ylim(0.0, 1.1)
    ax1.set_title("Stability")
    ax1.grid(axis="y", alpha=0.25)
    ax2.bar(labels, event_times, color="tab:orange")
    ax2.set_ylabel("Event time (s)")
    ax2.set_title("Convergence Event Time")
    ax2.grid(axis="y", alpha=0.25)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_rpa_split(rows: List[Dict[str, Any]], output_png: Path) -> None:
    if not rows:
        return
    labels = [f"{row['scenario']}:{row['model']}" for row in rows]
    values = [
        np.nan if row.get("final_rpa_split") in (None, "") else float(row["final_rpa_split"])
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("Final RPA split")
    ax.set_title("Final RPA Split by Scenario and Model")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_split_history_overlay(
    summaries: List[Dict[str, Any]],
    *,
    output_png: Path,
    split: str,
) -> None:
    if split not in {"rpa", "lpa"}:
        raise ValueError("split overlay must be 'rpa' or 'lpa'")

    plotted = False
    fig, ax = plt.subplots(figsize=(10, 5))
    for summary in summaries:
        if summary.get("status") != "ok":
            continue
        artifacts = summary.get("artifacts") or {}
        history_csv = artifacts.get("flow_split_history_csv")
        if not history_csv:
            continue
        history_path = Path(history_csv)
        if not history_path.exists():
            continue
        times: List[float] = []
        values: List[float] = []
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                rpa_split = float(row["rpa_split"])
                times.append(float(row["time_s"]))
                values.append(rpa_split if split == "rpa" else 1.0 - rpa_split)
        if not times:
            continue
        label = f"{summary['scenario']}:{summary['model']}"
        ax.plot(times, values, linewidth=1.5, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    label = "RPA" if split == "rpa" else "LPA"
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{label} split")
    ax.set_title(f"{label} Split History Overlay")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate_final_rpa_split(rows: List[Dict[str, Any]], output_png: Path) -> None:
    ok_rows = [
        row for row in rows
        if row.get("status") == "ok" and row.get("final_rpa_split") not in (None, "")
    ]
    if not ok_rows:
        return

    case_names = sorted({str(row.get("case_name") or row.get("scenario")) for row in ok_rows})
    x_by_case = {case: idx for idx, case in enumerate(case_names)}
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(case_names)), 5))
    for row in ok_rows:
        case = str(row.get("case_name") or row.get("scenario"))
        label = str(row.get("patient_id") or row.get("scenario_group") or "unlabeled")
        marker = "x" if int(row.get("stability_screen_failed") or 0) else "o"
        ax.scatter(
            x_by_case[case],
            float(row["final_rpa_split"]),
            marker=marker,
            label=label,
            alpha=0.8,
        )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=7, ncol=2)
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha="right")
    ax.set_ylabel("Final RPA split")
    ax.set_title("Final RPA Split Across Benchmark Cases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_convergence_table(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_name") or ""), str(row.get("model") or ""))
        item = grouped.setdefault(
            key,
            {
                "case_name": key[0],
                "model": key[1],
                "n_total": 0,
                "n_ok": 0,
                "n_stable": 0,
                "n_converged": 0,
                "n_screen_failed": 0,
                "n_one_branch_collapse": 0,
                "n_radius_bounds_violation": 0,
                "n_thickness_bounds_violation": 0,
                "n_nonfinite_state_detected": 0,
                "n_nonphysical_terminal_load": 0,
            },
        )
        item["n_total"] += 1
        if row.get("status") == "ok":
            item["n_ok"] += 1
        if int(row.get("stable") or 0):
            item["n_stable"] += 1
        if row.get("termination_reason") not in (None, "", "t_end_reached"):
            item["n_converged"] += 1
        for src, dst in [
            ("stability_screen_failed", "n_screen_failed"),
            ("one_branch_collapse", "n_one_branch_collapse"),
            ("radius_bounds_violation", "n_radius_bounds_violation"),
            ("thickness_bounds_violation", "n_thickness_bounds_violation"),
            ("nonfinite_state_detected", "n_nonfinite_state_detected"),
            ("nonphysical_terminal_load", "n_nonphysical_terminal_load"),
        ]:
            if int(row.get(src) or 0):
                item[dst] += 1

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "case_name",
            "model",
            "n_total",
            "n_ok",
            "n_stable",
            "n_converged",
            "n_screen_failed",
            "n_one_branch_collapse",
            "n_radius_bounds_violation",
            "n_thickness_bounds_violation",
            "n_nonfinite_state_detected",
            "n_nonphysical_terminal_load",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped.values())


def _write_failure_table(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    failure_rows = [
        row for row in rows
        if row.get("status") != "ok" or int(row.get("stability_screen_failed") or 0)
    ]
    if not failure_rows:
        output_csv.write_text("", encoding="utf-8")
        return
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "case_name",
            "patient_id",
            "scenario_group",
            "perturbation_severity",
            "scenario",
            "model",
            "status",
            "stable",
            "termination_reason",
            "final_rpa_split",
            "one_branch_collapse",
            "radius_bounds_violation",
            "thickness_bounds_violation",
            "nonfinite_state_detected",
            "nonphysical_terminal_load",
            "stability_screen_failed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_summary_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_name",
                "patient_id",
                "scenario_group",
                "perturbation_severity",
                "scenario",
                "model",
                "status",
                "update_mode",
                "terminal_load_policy",
                "terminal_resistance",
                "wss_gain",
                "k_tau_r",
                "k_sig_r",
                "k_tau_h",
                "k_sig_h",
                "stable",
                "final_rpa_split",
                "final_lpa_split",
                "preop_rpa_split",
                "postop_rpa_split",
                "geom_err",
                "t95",
                "n_rhs",
                "tree_max_nodes",
                "lpa_tree_nodes",
                "rpa_tree_nodes",
                "lpa_tree_max_nodes_reached",
                "rpa_tree_max_nodes_reached",
                "termination_reason",
                "event_time",
                "lpa_radius_mean_relative_change",
                "lpa_radius_max_abs_relative_change",
                "rpa_radius_mean_relative_change",
                "rpa_radius_max_abs_relative_change",
                "lpa_thickness_mean_relative_change",
                "lpa_thickness_max_abs_relative_change",
                "rpa_thickness_mean_relative_change",
                "rpa_thickness_max_abs_relative_change",
                "collapse_split_floor",
                "collapse_split_ceiling",
                "radius_max_abs_relative_change_limit",
                "thickness_max_abs_relative_change_limit",
                "one_branch_collapse",
                "radius_bounds_violation",
                "thickness_bounds_violation",
                "nonfinite_state_detected",
                "nonphysical_terminal_load",
                "stability_screen_failed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _failure_summary(
    *,
    scenario_name: str,
    model: str,
    update_mode: str,
    parameter_set: Dict[str, Any],
    output_dir: Path,
    exc: Exception,
    patient_id: str | None = None,
    scenario_group: str | None = None,
    perturbation_severity: str | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    error_type = type(exc).__name__
    error_message = str(exc)
    k_arr = list(parameter_set.get("k_arr") or [None, None, None, None]) + [None, None, None, None]
    normalized = {
        "case_name": str(parameter_set.get("benchmark_case") or scenario_name),
        "patient_id": patient_id,
        "scenario_group": scenario_group,
        "perturbation_severity": perturbation_severity,
        "scenario": scenario_name,
        "model": model,
        "status": "error",
        "update_mode": update_mode,
        "terminal_load_policy": parameter_set.get("terminal_load_policy"),
        "terminal_resistance": parameter_set.get("terminal_resistance"),
        "wss_gain": parameter_set.get("wss_gain"),
        "k_tau_r": k_arr[0],
        "k_sig_r": k_arr[1],
        "k_tau_h": k_arr[2],
        "k_sig_h": k_arr[3],
        "stable": 0,
        "final_rpa_split": None,
        "final_lpa_split": None,
        "preop_rpa_split": None,
        "postop_rpa_split": None,
        "geom_err": None,
        "t95": None,
        "n_rhs": None,
        "termination_reason": f"{error_type}: {error_message}",
        "event_time": None,
        "lpa_radius_mean_relative_change": None,
        "lpa_radius_max_abs_relative_change": None,
        "rpa_radius_mean_relative_change": None,
        "rpa_radius_max_abs_relative_change": None,
        "lpa_thickness_mean_relative_change": None,
        "lpa_thickness_max_abs_relative_change": None,
        "rpa_thickness_mean_relative_change": None,
        "rpa_thickness_max_abs_relative_change": None,
    }
    normalized.update(_screen_benchmark_row(normalized, parameter_set=parameter_set))
    summary_path = output_dir / "adaptation_summary.json"
    metrics_path = output_dir / "adaptation_metrics.json"
    payload = {
        "status": "error",
        "scenario": scenario_name,
        "model": model,
        "update_mode": update_mode,
        "parameter_provenance": json.loads(json.dumps(parameter_set, sort_keys=True)),
        "normalized_metrics": normalized,
        "error": {
            "type": error_type,
            "message": error_message,
            "traceback": traceback.format_exc(),
        },
        "artifacts": {
            "adaptation_summary_json": str(summary_path),
            "adaptation_metrics_json": str(metrics_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _run_scenario_model_job(job: Dict[str, Any]) -> Dict[str, Any]:
    update_mode = "algebraic" if job["model"] == "M2" else "dynamic"
    try:
        summary = _run_single_model(
            scenario_name=job["scenario_name"],
            model=job["model"],
            parameter_set=job["parameter_set"],
            preop_config_path=job["preop_config_path"],
            postop_config_path=job["postop_config_path"],
            tree_params_csv=job["tree_params_csv"],
            clinical_targets_csv=job["clinical_targets_csv"],
            output_dir=Path(job["output_dir"]),
            patient_id=job.get("patient_id"),
            scenario_group=job.get("scenario_group"),
            perturbation_severity=job.get("perturbation_severity"),
        )
    except Exception as exc:
        summary = _failure_summary(
            scenario_name=job["scenario_name"],
            model=job["model"],
            update_mode=update_mode,
            parameter_set=job["parameter_set"],
            output_dir=Path(job["output_dir"]),
            exc=exc,
            patient_id=job.get("patient_id"),
            scenario_group=job.get("scenario_group"),
            perturbation_severity=job.get("perturbation_severity"),
        )
    return {
        "scenario_index": job["scenario_index"],
        "model_index": job["model_index"],
        "scenario_name": job["scenario_name"],
        "patient_id": job.get("patient_id"),
        "scenario_group": job.get("scenario_group"),
        "perturbation_severity": job.get("perturbation_severity"),
        "model": job["model"],
        "summary": summary,
    }


def run_adaptation_benchmark_study(spec: AdaptBenchmarkConfig) -> Dict[str, Any]:
    output_root = Path(spec.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    scenario_summaries: List[Dict[str, Any]] = []
    flat_model_summaries: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    for scenario_index, scenario in enumerate(spec.scenarios):
        tree_params_csv = scenario.tree_params_csv or spec.tree_params_csv
        clinical_targets_csv = scenario.clinical_targets_csv or spec.clinical_targets_csv
        if tree_params_csv is None:
            raise ValueError(
                f"scenario '{scenario.name}' requires tree_params_csv either locally or in adapt_benchmark defaults"
            )
        if clinical_targets_csv is None:
            raise ValueError(
                f"scenario '{scenario.name}' requires clinical_targets_csv either locally or in adapt_benchmark defaults"
            )
        preop_path = _require_existing_file(
            scenario.preop_rri_config,
            label=f"scenario '{scenario.name}' preop_rri_config",
        )
        postop_path = _require_existing_file(
            scenario.postop_rri_config,
            label=f"scenario '{scenario.name}' postop_rri_config",
        )
        tree_params_path = _require_existing_file(
            tree_params_csv,
            label=f"scenario '{scenario.name}' tree_params_csv",
        )
        clinical_targets_path = _require_existing_file(
            clinical_targets_csv,
            label=f"scenario '{scenario.name}' clinical_targets_csv",
        )
        scenario_dir = output_root / scenario.name
        for model_index, model in enumerate(spec.models):
            model_dir = scenario_dir / model.lower()
            parameter_set = _merged_parameters(
                spec.parameter_overrides,
                scenario.parameter_overrides,
                model,
            )
            jobs.append(
                {
                    "scenario_index": scenario_index,
                    "model_index": model_index,
                    "scenario_name": scenario.name,
                    "model": model,
                    "parameter_set": parameter_set,
                    "preop_config_path": str(preop_path),
                    "postop_config_path": str(postop_path),
                    "tree_params_csv": str(tree_params_path),
                    "clinical_targets_csv": str(clinical_targets_path),
                    "output_dir": str(model_dir),
                    "patient_id": scenario.patient_id,
                    "scenario_group": scenario.scenario_group,
                    "perturbation_severity": scenario.perturbation_severity,
                }
            )

    workers = max(1, int(getattr(spec, "workers", 1) or 1))
    if workers == 1 or len(jobs) <= 1:
        job_results = [_run_scenario_model_job(job) for job in jobs]
    else:
        job_results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = [executor.submit(_run_scenario_model_job, job) for job in jobs]
            for future in as_completed(futures):
                job_results.append(future.result())

    job_results.sort(key=lambda item: (item["scenario_index"], item["model_index"]))
    scenario_models: Dict[int, List[Dict[str, str]]] = {}
    scenario_meta: Dict[int, Dict[str, Any]] = {}
    for result in job_results:
        summary = result["summary"]
        rows.append(dict(summary["normalized_metrics"]))
        flat_model_summaries.append(summary)
        scenario_models.setdefault(result["scenario_index"], []).append(
            {
                "model": result["model"],
                "summary_path": summary["artifacts"]["adaptation_summary_json"],
                "metrics_path": summary["artifacts"]["adaptation_metrics_json"],
            }
        )
        scenario_meta[result["scenario_index"]] = {
            "name": result["scenario_name"],
            "patient_id": result["patient_id"],
            "scenario_group": result["scenario_group"],
            "perturbation_severity": result["perturbation_severity"],
        }

    for scenario_index in sorted(scenario_models):
        scenario_summaries.append(
            {
                **scenario_meta[scenario_index],
                "models": scenario_models[scenario_index],
            }
        )

    summary_json = output_root / "benchmark_summary.json"
    summary_csv = output_root / "benchmark_summary.csv"
    stability_png = output_root / "benchmark_stability_convergence.png"
    final_split_png = output_root / "benchmark_final_rpa_split.png"
    rpa_overlay_png = output_root / "benchmark_rpa_split_overlay.png"
    lpa_overlay_png = output_root / "benchmark_lpa_split_overlay.png"
    aggregate_final_split_png = output_root / "benchmark_aggregate_final_rpa_split.png"
    convergence_table_csv = output_root / "benchmark_convergence_table.csv"
    failure_table_csv = output_root / "benchmark_failure_table.csv"
    _write_summary_csv(rows, summary_csv)
    _plot_stability_convergence(rows, stability_png)
    _plot_final_rpa_split(rows, final_split_png)
    _plot_split_history_overlay(flat_model_summaries, output_png=rpa_overlay_png, split="rpa")
    _plot_split_history_overlay(flat_model_summaries, output_png=lpa_overlay_png, split="lpa")
    _plot_aggregate_final_rpa_split(rows, aggregate_final_split_png)
    _write_convergence_table(rows, convergence_table_csv)
    _write_failure_table(rows, failure_table_csv)

    patient_overlay_artifacts: Dict[str, str] = {}
    patients = sorted({row.get("patient_id") for row in rows if row.get("patient_id")})
    for patient_id in patients:
        patient_summaries = [
            summary for summary in flat_model_summaries
            if (summary.get("normalized_metrics") or {}).get("patient_id") == patient_id
        ]
        safe_patient = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(patient_id)
        )
        patient_rpa_png = output_root / f"benchmark_{safe_patient}_rpa_split_overlay.png"
        patient_lpa_png = output_root / f"benchmark_{safe_patient}_lpa_split_overlay.png"
        _plot_split_history_overlay(patient_summaries, output_png=patient_rpa_png, split="rpa")
        _plot_split_history_overlay(patient_summaries, output_png=patient_lpa_png, split="lpa")
        patient_overlay_artifacts[f"benchmark_{safe_patient}_rpa_split_overlay_png"] = str(patient_rpa_png)
        patient_overlay_artifacts[f"benchmark_{safe_patient}_lpa_split_overlay_png"] = str(patient_lpa_png)

    payload = {
        "study_id": spec.study_id,
        "models": list(spec.models),
        "output_dir": str(output_root),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "artifacts": {
            "benchmark_stability_convergence_png": str(stability_png),
            "benchmark_final_rpa_split_png": str(final_split_png),
            "benchmark_rpa_split_overlay_png": str(rpa_overlay_png),
            "benchmark_lpa_split_overlay_png": str(lpa_overlay_png),
            "benchmark_aggregate_final_rpa_split_png": str(aggregate_final_split_png),
            "benchmark_convergence_table_csv": str(convergence_table_csv),
            "benchmark_failure_table_csv": str(failure_table_csv),
            **patient_overlay_artifacts,
        },
        "rows": rows,
        "scenarios": scenario_summaries,
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
