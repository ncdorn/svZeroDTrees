"""Stable structured-tree adaptation entrypoints."""

from __future__ import annotations

from svzerodtrees.pa_naming import pa_side
from pathlib import Path
import copy
import json
import os
import shutil

import numpy as np
import pandas as pd

from ..io import ConfigHandler
from ..io.blocks.boundary_condition import resolve_coupled_impedance_kernel_steps
from ..microvasculature.compliance.constant import ConstantCompliance
from ..microvasculature.compliance.olufsen import OlufsenCompliance
from ..microvasculature.structured_tree.structuredtree import DEFAULT_MAX_NODES
from ..simulation.simulation_directory import SimulationDirectory
from ..tune_bcs.clinical_targets import ClinicalTargets
from .artifacts import write_reduced_pa_flow_split_convergence_artifacts
from .microvascular_adaptor import (
    MicrovascularAdaptor,
    _impedance_kernel_steps_from_config,
    _resolve_inflow_time_array,
)
from .tuned_trees import (
    TunedTreeModel,
    adaptation_clinical_targets,
    load_tuned_tree_model,
    write_adapted_tuned_trees,
)

_FLOW_EPS = 1e-8
_RESISTANCE_EPS = 1e-8


def _sum_flows(simdir: SimulationDirectory) -> tuple[float, float]:
    lpa_flow, rpa_flow = simdir.flow_split(get_mean=True, verbose=False)
    return (
        float(sum(float(value) for value in lpa_flow.values())),
        float(sum(float(value) for value in rpa_flow.values())),
    )


def _stage_metrics(
    *,
    lpa_flow: float,
    rpa_flow: float,
    lpa_resistance: float,
    rpa_resistance: float,
) -> dict[str, float]:
    total_flow = float(lpa_flow) + float(rpa_flow)
    lpa_split = float(lpa_flow) / total_flow if abs(total_flow) > _FLOW_EPS else 0.0
    rpa_split = float(rpa_flow) / total_flow if abs(total_flow) > _FLOW_EPS else 0.0
    return {
        "lpa_flow": float(lpa_flow),
        "rpa_flow": float(rpa_flow),
        "lpa_split": lpa_split,
        "rpa_split": rpa_split,
        "lpa_resistance": float(lpa_resistance),
        "rpa_resistance": float(rpa_resistance),
    }


def _mean_resistances(simdir: SimulationDirectory) -> tuple[float, float]:
    try:
        _, _, _, lpa_resistance, rpa_resistance = simdir._compute_pressure_drops(get_mean=True)
        return float(lpa_resistance), float(rpa_resistance)
    except Exception:
        try:
            return _mean_resistances_from_postprocessed_mpa(simdir)
        except Exception:
            pass
        lpa_resistance, rpa_resistance = simdir.compute_pressure_drop(steady=True)
        return float(lpa_resistance), float(rpa_resistance)


def _resolve_mpa_pressure_csv(simdir: SimulationDirectory) -> Path:
    sim_path = Path(simdir.path)
    candidates = [
        sim_path / "mpa_pressure_vs_time.csv",
        sim_path.parent / "results" / "mpa_pressure_vs_time.csv",
        sim_path.parent / "results" / "postprocess" / "mpa_pressure_vs_time.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"mpa_pressure_vs_time.csv not found for simulation directory {sim_path}"
    )


def _mean_mpa_pressure_from_csv(simdir: SimulationDirectory) -> float:
    pressure_df = pd.read_csv(_resolve_mpa_pressure_csv(simdir))
    pressure_col = next(
        (
            col
            for col in ("mpa_pressure_mmhg", "pressure_mmhg", "pressure", "mpa_pressure")
            if col in pressure_df.columns
        ),
        None,
    )
    if pressure_col is None:
        raise ValueError(
            "mpa pressure csv must contain one of: "
            "mpa_pressure_mmhg, pressure_mmhg, pressure, mpa_pressure"
        )
    pressure_mmhg = pd.to_numeric(pressure_df[pressure_col], errors="coerce").to_numpy(dtype=float)
    pressure_mmhg = pressure_mmhg[np.isfinite(pressure_mmhg)]
    if pressure_mmhg.size == 0:
        raise ValueError("mpa pressure csv contains no finite pressure samples")
    return float(np.mean(pressure_mmhg)) * 1333.2


def _mean_resistances_from_postprocessed_mpa(simdir: SimulationDirectory) -> tuple[float, float]:
    if simdir.svzerod_data is None:
        raise ValueError("svZeroD_data not found")
    if simdir.svzerod_3Dcoupling is None:
        raise ValueError("svzerod_3Dcoupling not found")

    mean_mpa_pressure = _mean_mpa_pressure_from_csv(simdir)
    lpa_flow, rpa_flow = simdir.flow_split(get_mean=True, verbose=False)
    q_lpa = float(sum(float(value) for value in lpa_flow.values()))
    q_rpa = float(sum(float(value) for value in rpa_flow.values()))
    if not np.isfinite(q_lpa) or not np.isfinite(q_rpa) or q_lpa == 0.0 or q_rpa == 0.0:
        raise ValueError("mean flow split is required to compute fallback resistances")

    lpa_outlet_pressures: list[float] = []
    rpa_outlet_pressures: list[float] = []
    for block in simdir.svzerod_3Dcoupling.coupling_blocks.values():
        _, _, pressure = simdir.svzerod_data.get_result(block)
        pressure_arr = np.asarray(pressure, dtype=float)
        pressure_arr = pressure_arr[np.isfinite(pressure_arr)]
        if pressure_arr.size == 0:
            continue
        mean_pressure = float(np.mean(pressure_arr[-100:]))
        surface_name = str(getattr(block, "surface", "")).lower()
        if pa_side(surface_name) == "lpa":
            lpa_outlet_pressures.append(mean_pressure)
        elif pa_side(surface_name) == "rpa":
            rpa_outlet_pressures.append(mean_pressure)

    if not lpa_outlet_pressures or not rpa_outlet_pressures:
        raise ValueError("outlet pressures are required to compute fallback resistances")

    lpa_resistance = (mean_mpa_pressure - float(np.mean(lpa_outlet_pressures))) / q_lpa
    rpa_resistance = (mean_mpa_pressure - float(np.mean(rpa_outlet_pressures))) / q_rpa
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


def _territory_homeostatic_scale(
    *,
    preop_flow: float,
    postop_flow: float,
    preop_resistance: float,
    postop_resistance: float,
    iterations: int,
    wss_gain: float,
    ims_gain: float,
) -> dict[str, float]:
    flow_ratio = max(abs(float(postop_flow)), _FLOW_EPS) / max(abs(float(preop_flow)), _FLOW_EPS)
    resistance_ratio = max(float(postop_resistance), _RESISTANCE_EPS) / max(
        float(preop_resistance),
        _RESISTANCE_EPS,
    )
    flow_scale = flow_ratio ** (float(wss_gain) / 3.0)
    ims_scale = resistance_ratio ** (float(ims_gain) / 4.0)
    total_scale = (flow_scale * ims_scale) ** max(int(iterations), 1)
    return {
        "flow_ratio": float(flow_ratio),
        "resistance_ratio": float(resistance_ratio),
        "flow_scale": float(flow_scale),
        "ims_scale": float(ims_scale),
        "total_scale": float(total_scale),
    }


def _apply_diameter_scale(tree, *, total_scale: float, compliance_gain: float) -> None:
    if hasattr(tree, "apply_diameter_scale"):
        # Recorded in the tree metadata so rebuilt trees keep the adaptation.
        tree.apply_diameter_scale(total_scale)
    else:
        tree.store.d = np.asarray(tree.store.d, dtype=float) * total_scale
    _scale_compliance(tree, diameter_scale=total_scale, compliance_gain=compliance_gain)


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
    update = _territory_homeostatic_scale(
        preop_flow=preop_flow,
        postop_flow=postop_flow,
        preop_resistance=preop_resistance,
        postop_resistance=postop_resistance,
        iterations=iterations,
        wss_gain=wss_gain,
        ims_gain=ims_gain,
    )
    _apply_diameter_scale(tree, total_scale=update["total_scale"], compliance_gain=compliance_gain)
    return update


def _model_parameters_for_summary(parameter_set: dict | None) -> dict:
    return json.loads(json.dumps(parameter_set or {}, sort_keys=True))


def _iterations_or_default(parameter_set: dict, default: int = 1) -> int:
    value = parameter_set.get("iterations")
    return int(value) if value is not None else int(default)


def _optional_max_nodes(parameter_set: dict) -> int | None:
    value = parameter_set.get("max_nodes")
    return int(value) if value is not None else None


def _reduced_pa_tree_max_nodes(parameter_set: dict, tuned_model: TunedTreeModel) -> int:
    """Node budget of the M1/M3 reduced-PA trees: explicit, else the tuned one."""
    explicit = _optional_max_nodes(parameter_set)
    if explicit is not None:
        return explicit
    budgets = [tree.max_nodes for tree in tuned_model.trees if tree.max_nodes is not None]
    return max(budgets) if budgets else DEFAULT_MAX_NODES


def _coupled_kernel_grid(coupler) -> tuple[list[float] | None, int | None]:
    """Kernel time grid of ``regenerate_impedance_bcs_for_coupled_timing``.

    Building the exported kernels on the same grid keeps the exported adapted
    coupler equal to the one the 3D run regenerates.  Returns (None, None)
    when the coupler has no coupled cardiac period and step size.
    """
    resolve_period = getattr(coupler, "_resolve_coupled_cardiac_period", None)
    step_size = getattr(getattr(coupler, "simparams", None), "external_step_size", None)
    period = resolve_period() if callable(resolve_period) else None
    if period is None or step_size is None:
        return None, None
    kernel_steps = resolve_coupled_impedance_kernel_steps(
        cardiac_period=period,
        external_step_size=step_size,
    )
    return np.linspace(0.0, float(period), kernel_steps + 1).tolist(), kernel_steps


def _write_m2_adapted_coupler(
    *,
    preop,
    postop,
    adapted,
    adapted_dir: str,
    tuned_model: TunedTreeModel,
    adapt_tree,
    max_nodes_override: int | None,
) -> dict[str, dict]:
    """Write the postop coupler with every tuned tree adapted, by BC name."""
    postop_coupler = getattr(postop, "svzerod_3Dcoupling", None)
    if postop_coupler is None:
        raise ValueError("M2 adaptation requires the postop svzerod_3Dcoupling.json")
    time, kernel_steps = _coupled_kernel_grid(postop_coupler)
    if time is None:
        time = list(
            _resolve_inflow_time_array(
                getattr(preop, "svzerod_3Dcoupling", None),
                postop_coupler,
                getattr(preop, "zerod_config", None),
            )
        )
        kernel_steps = _impedance_kernel_steps_from_config(postop_coupler)
    # Work on a copy so the postop coupling config is not mutated.
    coupler = copy.deepcopy(postop_coupler)
    coupler.path = os.path.join(adapted_dir, "svzerod_3Dcoupling.json")
    coupler.bcs.pop("INFLOW", None)
    coupler.inflows.pop("INFLOW", None)
    tree_metrics = write_adapted_tuned_trees(
        coupler,
        tuned_model,
        adapt_tree,
        time=time,
        kernel_steps=kernel_steps,
        max_nodes_override=max_nodes_override,
    )
    adapted.svzerod_3Dcoupling = coupler
    print("saving adapted config to " + coupler.path)
    coupler.to_json(coupler.path)
    return tree_metrics


def _require_side_trees(tuned_model: TunedTreeModel, model: str) -> None:
    """M1/M3 adapt one tree per side; refuse per-cap tuned models."""
    if tuned_model.per_outlet:
        counts = {side: len(tuned_model.trees_for_side(side)) for side in ("lpa", "rpa")}
        raise ValueError(
            f"{model} integrates adaptation on one LPA and one RPA tree inside the "
            "reduced-order PA (RRI) model, but the tuned config has per-cap trees "
            f"({counts['lpa']} LPA, {counts['rpa']} RPA). Running it would replace the "
            "tuned per-cap trees with two trees at the optimized_params.csv diameter. "
            "Use M2, which adapts every tuned per-cap tree; extending M1/M3 to per-cap "
            "trees is an open modeling decision (svZeroDTrees "
            "docs/pulmonary_tuning_model.md, section 7)."
        )


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
    tuned_config: str | None = None,
    outlet_cap_mapping: str | None = None,
    wedge_pressure_policy: str | None = None,
    precapillary_fraction: float | None = None,
    diastolic_offset_mmhg: float | None = None,
) -> dict:
    """Adapt the tuned preop structured trees to the postop hemodynamics.

    The starting trees, their outlet BCs/caps and the outlet pressure ``Pd``
    come from ``tuned_config`` (default: the preop 3D coupler), checked
    against ``outlet_cap_mapping`` when given.  ``wedge_pressure_policy`` (and
    its parameters) optionally cross-check ``Pd`` against the tuning policy.
    ``tree_params`` (optimized_params.csv) and ``reduced_order_pa`` are used by
    the reduced-PA models M1/M3 only.
    """
    resolved_model = str(model).upper()
    if resolved_model not in {"M1", "M2", "M3"}:
        raise ValueError(f"unsupported adaptation model '{model}'")
    if territory_scheme != "lpa_rpa":
        raise ValueError("territory_scheme currently supports only 'lpa_rpa'")
    if mode not in {"predict", "retrospective_fit"}:
        raise ValueError("mode must be one of predict|retrospective_fit")
    if outlet_cap_mapping is not None and not os.path.isfile(outlet_cap_mapping):
        raise FileNotFoundError(f"outlet cap mapping not found: {outlet_cap_mapping}")
    if tuned_config is not None and not os.path.isfile(tuned_config):
        raise FileNotFoundError(f"tuned config not found: {tuned_config}")

    preop = SimulationDirectory.from_directory(preop_dir, convert_to_cm=convert_to_cm)
    postop = SimulationDirectory.from_directory(postop_dir, convert_to_cm=convert_to_cm)
    adapted = SimulationDirectory.from_directory(adapted_dir, convert_to_cm=convert_to_cm)
    tuned_source = tuned_config if tuned_config is not None else getattr(preop, "svzerod_3Dcoupling", None)
    if tuned_source is None:
        raise ValueError(
            "adaptation needs the tuned config: pass tuned_config or provide a preop "
            "directory with svzerod_3Dcoupling.json"
        )
    tuned_model = load_tuned_tree_model(tuned_source, outlet_cap_mapping=outlet_cap_mapping)
    targets, outlet_pressure = adaptation_clinical_targets(
        clinical_targets,
        tuned_outlet_pressure_dyn=tuned_model.outlet_pressure_dyn,
        wedge_pressure_policy=wedge_pressure_policy,
        precapillary_fraction=precapillary_fraction,
        diastolic_offset_mmhg=diastolic_offset_mmhg,
    )
    params = parameter_set or {}
    adaptor = None
    if resolved_model in {"M1", "M3"}:
        _require_side_trees(tuned_model, resolved_model)
        adaptor = MicrovascularAdaptor(
            preop,
            postop,
            adapted,
            targets,
            reduced_order_pa=reduced_order_pa,
            tree_params=tree_params,
            method="cwss",
            location="uniform",
            n_iter=int(params.get("iterations", 1)),
            bc_type="impedance",
            convert_to_cm=convert_to_cm,
            tuned_model=tuned_model,
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
    threed_hemodynamics = {
        "preop": _stage_metrics(
            lpa_flow=preop_lpa_flow,
            rpa_flow=preop_rpa_flow,
            lpa_resistance=preop_lpa_resistance,
            rpa_resistance=preop_rpa_resistance,
        ),
        "postop": _stage_metrics(
            lpa_flow=postop_lpa_flow,
            rpa_flow=postop_rpa_flow,
            lpa_resistance=postop_lpa_resistance,
            rpa_resistance=postop_rpa_resistance,
        ),
    }

    solver_metrics: dict[str, float | int] | None = None
    tree_metrics: dict[str, dict] | None = None
    if resolved_model == "M1":
        solver_metrics = adaptor.adapt_cwss(
            n_iter=_iterations_or_default(params, 1),
            wss_gain=float(params.get("wss_gain", 0.01)),
            terminal_resistance=float(params.get("terminal_resistance") or 0.0),
            t_end=params.get("t_end"),
            rtol=float(params.get("rtol", 1e-6)),
            atol=float(params.get("atol", 1e-7)),
            max_step=float(params.get("max_step", 60.0)),
            method=str(params.get("solver_method", "RK23")),
            max_nodes=_reduced_pa_tree_max_nodes(params, tuned_model),
        )
    elif resolved_model == "M2":
        compliance_gain = float(params.get("compliance_gain", 1.0))
        side_updates = {
            side: _territory_homeostatic_scale(
                preop_flow=territory_metrics[side]["preop_flow"],
                postop_flow=territory_metrics[side]["postop_flow"],
                preop_resistance=territory_metrics[side]["preop_resistance"],
                postop_resistance=territory_metrics[side]["postop_resistance"],
                iterations=_iterations_or_default(params, 1),
                wss_gain=float(params.get("wss_gain", 1.0)),
                ims_gain=float(params.get("ims_gain", 1.0)),
            )
            for side in ("lpa", "rpa")
        }
        for side, update in side_updates.items():
            territory_metrics[side].update(update)
            territory_metrics[side]["n_trees"] = len(tuned_model.trees_for_side(side))

        def _adapt_tree(tuned_tree, tree):
            # Territory-level update: every tree on a side gets that side's scale.
            total_scale = side_updates[tuned_tree.side]["total_scale"]
            _apply_diameter_scale(tree, total_scale=total_scale, compliance_gain=compliance_gain)
            return {"model": "M2", "territory": tuned_tree.side, "total_scale": total_scale}

        tree_metrics = _write_m2_adapted_coupler(
            preop=preop,
            postop=postop,
            adapted=adapted,
            adapted_dir=adapted_dir,
            tuned_model=tuned_model,
            adapt_tree=_adapt_tree,
            max_nodes_override=_optional_max_nodes(params),
        )
    else:
        solver_metrics = adaptor.adapt_cwss_ims(
            params.get("k_arr", [1.0, 1.0, 1.0, 1.0]),
            t_end=float(params.get("t_end", 3600.0)),
            rtol=float(params.get("rtol", 1e-6)),
            atol=float(params.get("atol", 1e-7)),
            max_step=float(params.get("max_step", 60.0)),
            method=str(params.get("solver_method", "RK23")),
            max_nodes=_reduced_pa_tree_max_nodes(params, tuned_model),
        )

    output_dir = Path(output_root or adapted_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_coupler = Path(adapted_dir) / "svzerod_3Dcoupling.json"
    exported_coupler = output_dir / "adapted_svzerod_3Dcoupling.json"
    if adapted_coupler.exists():
        shutil.copy2(adapted_coupler, exported_coupler)
    elif adapted.svzerod_3Dcoupling is not None:
        adapted.svzerod_3Dcoupling.to_json(str(exported_coupler))
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
        "hemodynamics": {
            "threed": threed_hemodynamics,
        },
        "tuned_model": tuned_model.provenance(),
        "outlet_pressure": outlet_pressure,
    }
    metrics = {
        "model": resolved_model,
        "territory_metrics": territory_metrics,
        "hemodynamics": {
            "threed": threed_hemodynamics,
        },
        "outlet_pressure": outlet_pressure,
    }
    if tree_metrics is not None:
        summary["tree_metrics"] = tree_metrics
        metrics["tree_metrics"] = tree_metrics
    if solver_metrics is not None:
        summary["solver_metrics"] = solver_metrics
        metrics["solver_metrics"] = solver_metrics
        internal_zerod = {
            "preop": {"rpa_split": float(solver_metrics["preop_rpa_split"])},
            "postop_initial": {"rpa_split": float(solver_metrics["postop_rpa_split"])},
            "adapted_final": {"rpa_split": float(solver_metrics["final_rpa_split"])},
            "target": {"rpa_split": float(targets.rpa_split)},
        }
        summary["hemodynamics"]["internal_zerod"] = internal_zerod
        metrics["hemodynamics"]["internal_zerod"] = internal_zerod
        if resolved_model == "M1":
            solver_diagnostics = solver_metrics.get("solver_diagnostics") or {}
            accepted_history = solver_diagnostics.get("accepted_step_flow_split_history") or []
            if accepted_history:
                convergence_artifacts = write_reduced_pa_flow_split_convergence_artifacts(
                    output_dir=output_dir,
                    flow_split_history=accepted_history,
                    preop_rpa_split=float(solver_metrics["preop_rpa_split"]),
                    postop_rpa_split=float(solver_metrics["postop_rpa_split"]),
                    target_rpa_split=float(targets.rpa_split),
                    final_rpa_split=float(solver_metrics["final_rpa_split"]),
                )
                summary["artifacts"].update(convergence_artifacts)
                metrics["artifacts"] = dict(convergence_artifacts)
    summary_path = output_dir / "adaptation_summary.json"
    metrics_path = output_dir / "adaptation_metrics.json"
    summary["artifacts"]["adaptation_summary_json"] = str(summary_path)
    summary["artifacts"]["adaptation_metrics_json"] = str(metrics_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return summary
