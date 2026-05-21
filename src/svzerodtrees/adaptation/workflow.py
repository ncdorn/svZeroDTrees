"""Stable structured-tree adaptation entrypoints."""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from ..io import ConfigHandler
from ..microvasculature.compliance.constant import ConstantCompliance
from ..microvasculature.compliance.olufsen import OlufsenCompliance
from ..simulation.simulation_directory import SimulationDirectory
from ..tune_bcs.clinical_targets import ClinicalTargets
from .microvascular_adaptor import MicrovascularAdaptor

_FLOW_EPS = 1e-8
_RESISTANCE_EPS = 1e-8


def _sum_flows(simdir: SimulationDirectory) -> tuple[float, float]:
    lpa_flow, rpa_flow = simdir.flow_split(get_mean=True, verbose=False)
    return (
        float(sum(float(value) for value in lpa_flow.values())),
        float(sum(float(value) for value in rpa_flow.values())),
    )


def _mean_resistances(simdir: SimulationDirectory) -> tuple[float, float]:
    try:
        _, _, _, lpa_resistance, rpa_resistance = simdir._compute_pressure_drops(get_mean=True)
        return float(lpa_resistance), float(rpa_resistance)
    except Exception:
        lpa_resistance, rpa_resistance = simdir.compute_pressure_drop(steady=True)
        return float(lpa_resistance), float(rpa_resistance)


def _scale_compliance(tree, *, diameter_scale: float, compliance_gain: float) -> None:
    model = getattr(tree, "compliance_model", None)
    if model is None:
        model = getattr(getattr(tree, "root", None), "compliance_model", None)
    if model is None:
        return

    scale = max(float(diameter_scale), 1e-6) ** max(float(compliance_gain), 0.0)
    if isinstance(model, ConstantCompliance):
        model.value = float(model.value) / scale
        model.params["Eh/r"] = model.value
    elif isinstance(model, OlufsenCompliance):
        model.k1 = float(model.k1) / scale
        model.k3 = float(model.k3) / scale
        model.params["k1"] = model.k1
        model.params["k2"] = model.k2
        model.params["k3"] = model.k3


def _apply_territory_homeostatic_update(
    tree,
    *,
    preop_flow: float,
    postop_flow: float,
    preop_resistance: float,
    postop_resistance: float,
    iterations: int,
    wss_gain: float,
    ims_gain: float,
    compliance_gain: float,
) -> dict[str, float]:
    flow_ratio = max(abs(float(postop_flow)), _FLOW_EPS) / max(abs(float(preop_flow)), _FLOW_EPS)
    resistance_ratio = max(float(postop_resistance), _RESISTANCE_EPS) / max(
        float(preop_resistance),
        _RESISTANCE_EPS,
    )
    flow_scale = flow_ratio ** (float(wss_gain) / 3.0)
    ims_scale = resistance_ratio ** (float(ims_gain) / 4.0)
    total_scale = (flow_scale * ims_scale) ** max(int(iterations), 1)
    tree.store.d = np.asarray(tree.store.d, dtype=float) * total_scale
    _scale_compliance(tree, diameter_scale=total_scale, compliance_gain=compliance_gain)
    return {
        "flow_ratio": float(flow_ratio),
        "resistance_ratio": float(resistance_ratio),
        "flow_scale": float(flow_scale),
        "ims_scale": float(ims_scale),
        "total_scale": float(total_scale),
    }


def _model_parameters_for_summary(parameter_set: dict | None) -> dict:
    return json.loads(json.dumps(parameter_set or {}, sort_keys=True))


def run_structured_tree_adaptation(
    *,
    preop_dir: str,
    postop_dir: str,
    adapted_dir: str,
    clinical_targets: str,
    reduced_order_pa: str,
    tree_params: str,
    model: str,
    territory_scheme: str = "lpa_rpa",
    parameter_set: dict | None = None,
    mode: str = "predict",
    convert_to_cm: bool = False,
    output_root: str | None = None,
) -> dict:
    resolved_model = str(model).upper()
    if resolved_model not in {"M1", "M2", "M3"}:
        raise ValueError(f"unsupported adaptation model '{model}'")
    if territory_scheme != "lpa_rpa":
        raise ValueError("territory_scheme currently supports only 'lpa_rpa'")
    if mode not in {"predict", "retrospective_fit"}:
        raise ValueError("mode must be one of predict|retrospective_fit")

    preop = SimulationDirectory.from_directory(preop_dir, convert_to_cm=convert_to_cm)
    postop = SimulationDirectory.from_directory(postop_dir, convert_to_cm=convert_to_cm)
    adapted = SimulationDirectory.from_directory(adapted_dir, convert_to_cm=convert_to_cm)
    targets = ClinicalTargets.from_csv(clinical_targets)
    adaptor = MicrovascularAdaptor(
        preop,
        postop,
        adapted,
        targets,
        reduced_order_pa=reduced_order_pa,
        tree_params=tree_params,
        method="cwss",
        location="uniform",
        n_iter=int((parameter_set or {}).get("iterations", 1)),
        bc_type="impedance",
        convert_to_cm=convert_to_cm,
    )

    preop_lpa_flow, preop_rpa_flow = _sum_flows(preop)
    postop_lpa_flow, postop_rpa_flow = _sum_flows(postop)
    preop_lpa_resistance, preop_rpa_resistance = _mean_resistances(preop)
    postop_lpa_resistance, postop_rpa_resistance = _mean_resistances(postop)

    territory_metrics: dict[str, dict[str, float]] = {
        "lpa": {
            "preop_flow": preop_lpa_flow,
            "postop_flow": postop_lpa_flow,
            "preop_resistance": preop_lpa_resistance,
            "postop_resistance": postop_lpa_resistance,
        },
        "rpa": {
            "preop_flow": preop_rpa_flow,
            "postop_flow": postop_rpa_flow,
            "preop_resistance": preop_rpa_resistance,
            "postop_resistance": postop_rpa_resistance,
        },
    }

    params = parameter_set or {}
    solver_metrics: dict[str, float | int] | None = None
    if resolved_model == "M1":
        solver_metrics = adaptor.adapt_cwss(
            n_iter=int(params.get("iterations", 1)),
            wss_gain=float(params.get("wss_gain", 0.01)),
            t_end=params.get("t_end"),
            rtol=float(params.get("rtol", 1e-6)),
            atol=float(params.get("atol", 1e-7)),
            max_step=float(params.get("max_step", 60.0)),
        )
    elif resolved_model == "M2":
        adaptor.lpa_tree, adaptor.rpa_tree = adaptor.construct_impedance_trees()
        territory_metrics["lpa"].update(
            _apply_territory_homeostatic_update(
                adaptor.lpa_tree,
                preop_flow=preop_lpa_flow,
                postop_flow=postop_lpa_flow,
                preop_resistance=preop_lpa_resistance,
                postop_resistance=postop_lpa_resistance,
                iterations=int(params.get("iterations", 1)),
                wss_gain=float(params.get("wss_gain", 1.0)),
                ims_gain=float(params.get("ims_gain", 1.0)),
                compliance_gain=float(params.get("compliance_gain", 1.0)),
            )
        )
        territory_metrics["rpa"].update(
            _apply_territory_homeostatic_update(
                adaptor.rpa_tree,
                preop_flow=preop_rpa_flow,
                postop_flow=postop_rpa_flow,
                preop_resistance=preop_rpa_resistance,
                postop_resistance=postop_rpa_resistance,
                iterations=int(params.get("iterations", 1)),
                wss_gain=float(params.get("wss_gain", 1.0)),
                ims_gain=float(params.get("ims_gain", 1.0)),
                compliance_gain=float(params.get("compliance_gain", 1.0)),
            )
        )
        adaptor._finalize_coupling_with_adapted_trees()
    else:
        adaptor.adapt_cwss_ims(params.get("k_arr", [1.0, 1.0, 1.0, 1.0]))

    output_dir = Path(output_root or adapted_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_coupler = Path(adapted_dir) / "svzerod_3Dcoupling.json"
    exported_coupler = output_dir / "adapted_svzerod_3Dcoupling.json"
    if adapted_coupler.exists():
        shutil.copy2(adapted_coupler, exported_coupler)
    elif adaptor.adapted_simdir.svzerod_3Dcoupling is not None:
        adaptor.adapted_simdir.svzerod_3Dcoupling.to_json(str(exported_coupler))
    else:
        raise RuntimeError("adaptation did not produce an adapted svzerod_3Dcoupling.json")

    summary = {
        "status": "ok",
        "model": resolved_model,
        "mode": mode,
        "territory_scheme": territory_scheme,
        "parameter_provenance": _model_parameters_for_summary(params),
        "artifacts": {
            "adapted_coupler_json": str(exported_coupler),
        },
        "territory_deltas": territory_metrics,
    }
    metrics = {
        "model": resolved_model,
        "territory_metrics": territory_metrics,
    }
    if solver_metrics is not None:
        summary["solver_metrics"] = solver_metrics
        metrics["solver_metrics"] = solver_metrics
    summary_path = output_dir / "adaptation_summary.json"
    metrics_path = output_dir / "adaptation_metrics.json"
    summary["artifacts"]["adaptation_summary_json"] = str(summary_path)
    summary["artifacts"]["adaptation_metrics_json"] = str(metrics_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return summary
