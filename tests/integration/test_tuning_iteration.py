from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tuning.iteration import (
    OPTIMIZATION_LOG_FILENAME,
    OPTIMIZED_PARAMS_FILENAME,
    PA_CONFIG_SNAPSHOT_FILENAME,
    _build_tune_space_from_config,
    _resolve_impedance_config,
    compute_centerline_mpa_metrics,
    compute_flow_split_metrics,
    evaluate_iteration_gate,
    generate_reduced_pa_from_iteration,
    run_impedance_tuning_for_iteration,
    write_iteration_decision,
    write_iteration_metrics,
)
from svzerodtrees.tuning.learned_seed import prepare_reduced_rri_seed_from_learned
from svzerodtrees.tune_bcs.assign_bcs import resolve_cap_to_bc_mapping


def _tune_space_with_xi() -> dict[str, list[dict[str, object]]]:
    return {
        "free": [
            {"name": "lpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0},
            {"name": "lpa.eta_sym", "init": 0.6, "lb": 0.3, "ub": 0.9},
            {"name": "rpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0},
            {"name": "rpa.eta_sym", "init": 0.7, "lb": 0.3, "ub": 0.9},
            {"name": "lpa.inductance", "init": 1.0, "lb": 0.0, "ub": "inf"},
            {"name": "rpa.inductance", "init": 1.0, "lb": 0.0, "ub": "inf"},
            {"name": "comp.lpa.k2", "init": -75.0, "lb": -100.0, "ub": -1.0},
        ],
        "fixed": [
            {"name": "lrr", "value": 10.0},
            {"name": "d_min", "value": 0.01},
        ],
        "tied": [
            {"name": "comp.rpa.k2", "other": "comp.lpa.k2", "fn": "identity"},
        ],
    }


def _constant_tune_space() -> dict[str, list[dict[str, object]]]:
    return {
        "free": [
            {"name": "comp.lpa.C", "init": 66000.0, "lb": 1.0, "ub": 100000.0},
            {"name": "comp.rpa.C", "init": 66000.0, "lb": 1.0, "ub": 100000.0},
        ],
        "fixed": [
            {"name": "lrr", "value": 10.0},
            {"name": "d_min", "value": 0.01},
        ],
        "tied": [],
    }


def _impedance_artifact_payload(
    *,
    bc_values: dict[str, object],
    coupled: bool = False,
    number_of_time_pts: int = 2,
    number_of_time_pts_per_cardiac_cycle: int = 2,
    inflow_q: list[float] | None = None,
    inflow_t: list[float] | None = None,
) -> dict[str, object]:
    simparams: dict[str, object] = {
        "cardiac_period": 1.0,
        "output_all_cycles": False,
        "steady_initial": False,
    }
    if coupled:
        simparams["coupled_simulation"] = True
        simparams["number_of_time_pts"] = number_of_time_pts
        simparams["external_step_size"] = 1.0
        simparams["density"] = 1.06
        simparams["viscosity"] = 0.04
    else:
        simparams["number_of_time_pts_per_cardiac_cycle"] = number_of_time_pts_per_cardiac_cycle
        simparams["number_of_cardiac_cycles"] = 1

    if coupled and inflow_t is None:
        sample_count = len(list(bc_values.get("z", []))) + 1
        resolved_inflow_t = [idx / max(sample_count - 1, 1) for idx in range(sample_count)]
    else:
        resolved_inflow_t = inflow_t if inflow_t is not None else [0.0, 1.0]

    resolved_inflow_q = inflow_q if inflow_q is not None else [1.0] * len(resolved_inflow_t)
    return {
        "simulation_parameters": simparams,
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": resolved_inflow_q, "t": resolved_inflow_t},
            },
            {
                "bc_name": "OUT",
                "bc_type": "IMPEDANCE",
                "bc_values": bc_values,
            }
        ],
    }


def _seed_config_payload(
    inflow_q: list[float] | None = None,
    inflow_t: list[float] | None = None,
) -> dict[str, object]:
    resolved_inflow_q = inflow_q if inflow_q is not None else [6.0, 6.0]
    resolved_inflow_t = inflow_t if inflow_t is not None else [0.0, 1.0]
    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": resolved_inflow_q, "t": resolved_inflow_t},
            }
        ]
    }


def _learned_bifurcation_payload() -> dict[str, object]:
    def _vessel(vessel_id: int, name: str, resistance: float, bc: dict[str, str] | None = None):
        vessel = {
            "vessel_id": vessel_id,
            "vessel_name": name,
            "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": resistance,
                "C": 1.0,
                "L": 0.0,
                "stenosis_coefficient": 0.0,
            },
        }
        if bc is not None:
            vessel["boundary_conditions"] = bc
        return vessel

    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": [10.0, 10.0], "t": [0.0, 1.0]},
            },
            {
                "bc_name": "LPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 500.0, "Pd": 0.0},
            },
            {
                "bc_name": "RPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 500.0, "Pd": 0.0},
            },
        ],
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 2,
            "number_of_cardiac_cycles": 1,
        },
        "vessels": [
            _vessel(0, "branch0_seg0", 10.0, {"inlet": "INFLOW"}),
            _vessel(1, "branch1_seg0", 20.0, {"outlet": "LPA_BC"}),
            _vessel(2, "branch2_seg0", 30.0, {"outlet": "RPA_BC"}),
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "areas": [1.0, 1.0],
            }
        ],
    }


def _full_pa_multi_outlet_payload() -> dict[str, object]:
    def _vessel(vessel_id: int, name: str, bc: dict[str, str] | None = None):
        vessel = {
            "vessel_id": vessel_id,
            "vessel_name": name,
            "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": 10.0,
                "C": 1.0,
                "L": 0.0,
                "stenosis_coefficient": 0.0,
            },
        }
        if bc is not None:
            vessel["boundary_conditions"] = bc
        return vessel

    outlet_bcs = ["LPA_1", "LPA_2", "RPA_1", "RPA_2"]
    return {
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 2,
            "number_of_cardiac_cycles": 1,
            "cardiac_period": 1.0,
        },
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": [6.0, 6.0], "t": [0.0, 1.0]},
            },
            *[
                {
                    "bc_name": name,
                    "bc_type": "RESISTANCE",
                    "bc_values": {"R": 500.0, "Pd": 0.0},
                }
                for name in outlet_bcs
            ],
        ],
        "vessels": [
            _vessel(0, "branch0_seg0", {"inlet": "INFLOW"}),
            _vessel(1, "branch1_seg0"),
            _vessel(2, "branch2_seg0"),
            _vessel(3, "branch3_seg0", {"outlet": "LPA_1"}),
            _vessel(4, "branch4_seg0", {"outlet": "LPA_2"}),
            _vessel(5, "branch5_seg0", {"outlet": "RPA_1"}),
            _vessel(6, "branch6_seg0", {"outlet": "RPA_2"}),
        ],
        "junctions": [],
    }


def _full_pa_impedance_snapshot_payload() -> dict[str, object]:
    payload = _full_pa_multi_outlet_payload()
    for bc in payload["boundary_conditions"]:
        if bc["bc_name"] == "INFLOW":
            continue
        bc["bc_type"] = "IMPEDANCE"
        bc["bc_values"] = {"z": [1.0, 0.5], "Pd": 12.0}
    return payload


def _reduced_rri_impedance_snapshot_payload() -> dict[str, object]:
    payload = _impedance_artifact_payload(
        bc_values={"z": [1.0, 0.5], "Pd": 12.0},
        coupled=False,
        number_of_time_pts_per_cardiac_cycle=3,
        inflow_q=[6.0, 6.0],
    )
    payload["boundary_conditions"][1]["bc_name"] = "LPA_BC"
    payload["boundary_conditions"].append(
        {
            "bc_name": "RPA_BC",
            "bc_type": "IMPEDANCE",
            "bc_values": {"z": [1.0, 0.5], "Pd": 12.0},
        }
    )
    payload["vessels"] = [{"vessel_id": idx} for idx in range(5)]
    return payload


def _fake_learned_seed_result(_config):
    rows = []
    for time, pressure, flow in [(0.0, 20.0, 10.0), (1.0, 30.0, 10.0)]:
        rows.append(
            {
                "name": "branch0_seg0",
                "time": time,
                "pressure_in": pressure * 1333.2,
                "pressure_out": pressure * 1333.2,
                "flow_in": flow,
                "flow_out": flow,
            }
        )
        for name, branch_flow in {
            "branch1_seg0": 6.0,
            "branch2_seg0": 4.0,
            "branch3_seg0": 4.0,
            "branch4_seg0": 4.0,
        }.items():
            rows.append(
                {
                    "name": name,
                    "time": time,
                    "pressure_in": pressure * 1333.2,
                    "pressure_out": pressure * 1333.2,
                    "flow_in": branch_flow,
                    "flow_out": branch_flow,
                }
            )
    return pd.DataFrame(rows)


def _write_constant_inflow_csv(tmp_path: Path, mean_flow: float) -> Path:
    inflow_path = tmp_path / "inflow.csv"
    inflow_path.write_text(
        f"t,q\n0.0,{float(mean_flow)}\n1.0,{float(mean_flow)}\n",
        encoding="utf-8",
    )
    return inflow_path


def test_compute_centerline_metrics_from_values():
    metrics = compute_centerline_mpa_metrics(pressure_values=[10.0, 20.0, 15.0, 13.0])
    assert metrics["mpa_sys"] == pytest.approx(20.0)
    assert metrics["mpa_dia"] == pytest.approx(10.0)
    assert metrics["mpa_mean"] == pytest.approx(14.5)


def test_compute_centerline_metrics_from_csv(tmp_path: Path):
    csv_path = tmp_path / "mpa_pressure_vs_time.csv"
    csv_path.write_text(
        "timestep_id,time_s,mpa_pressure_mmhg\n0,0.0,12.0\n1,0.1,18.0\n2,0.2,14.0\n",
        encoding="utf-8",
    )

    metrics = compute_centerline_mpa_metrics(pressure_csv=csv_path)
    assert metrics["mpa_sys"] == pytest.approx(18.0)
    assert metrics["mpa_dia"] == pytest.approx(12.0)
    assert metrics["mpa_mean"] == pytest.approx(14.666666666666666)


def test_evaluate_iteration_gate_passes_on_boundary():
    metrics = {
        "mpa_sys": 45.0,
        "mpa_dia": 20.0,
        "mpa_mean": 30.0,
        "rpa_split": 0.45,
    }
    targets = {"mpa_p": [50.0, 23.0, 33.0], "rpa_split": 0.50}

    result = evaluate_iteration_gate(metrics=metrics, clinical_targets=targets)
    assert result["close_to_targets"] is True
    assert result["decision"] == "converged"


def test_evaluate_iteration_gate_fails_when_over_threshold():
    metrics = {
        "mpa_sys": 44.9,
        "mpa_dia": 19.9,
        "mpa_mean": 29.9,
        "rpa_split": 0.449,
    }
    targets = {"mpa_p": [50.0, 23.0, 33.0], "rpa_split": 0.50}

    result = evaluate_iteration_gate(metrics=metrics, clinical_targets=targets)
    assert result["close_to_targets"] is False
    assert result["decision"] == "not_close"


def test_compute_flow_split_metrics_from_totals():
    result = compute_flow_split_metrics(lpa_total_flow=30.0, rpa_total_flow=20.0)
    assert result["rpa_split"] == pytest.approx(0.4)


def test_many_outlet_mapping_uses_matching_bc_names():
    class BC:
        def __init__(self, name: str):
            self.name = name

    class Config:
        bcs = {
            "INFLOW": BC("INFLOW"),
            "lpa_cap_1": BC("lpa_cap_1"),
            "lpa_cap_2": BC("lpa_cap_2"),
            "rpa_cap_1": BC("rpa_cap_1"),
            "rpa_cap_2": BC("rpa_cap_2"),
        }
        tree_params = {}

    cap_info = {
        "/mesh/lpa_cap_1.vtp": 1.0,
        "/mesh/lpa_cap_2.vtp": 1.0,
        "/mesh/rpa_cap_1.vtp": 1.0,
        "/mesh/rpa_cap_2.vtp": 1.0,
    }

    mapping = resolve_cap_to_bc_mapping(Config(), cap_info, bc_prefix="IMPEDANCE")
    assert mapping["/mesh/lpa_cap_1.vtp"] == "lpa_cap_1"
    assert mapping["/mesh/rpa_cap_2.vtp"] == "rpa_cap_2"


def test_many_outlet_mapping_rejects_ambiguous_order_only_mapping():
    class BC:
        def __init__(self, name: str):
            self.name = name

    class Config:
        bcs = {
            "INFLOW": BC("INFLOW"),
            "OUTLET_0": BC("OUTLET_0"),
            "OUTLET_1": BC("OUTLET_1"),
        }
        tree_params = {}

    cap_info = {"/mesh/lpa_cap_1.vtp": 1.0, "/mesh/rpa_cap_1.vtp": 1.0}
    with pytest.raises(ValueError, match="could not deterministically map"):
        resolve_cap_to_bc_mapping(Config(), cap_info, bc_prefix="IMPEDANCE")

    mapping = resolve_cap_to_bc_mapping(
        Config(),
        cap_info,
        bc_prefix="IMPEDANCE",
        allow_ordered_outlet_mapping=True,
    )
    assert mapping["/mesh/lpa_cap_1.vtp"] == "OUTLET_0"


def test_generate_reduced_pa_wrapper(monkeypatch, tmp_path: Path):
    calls = {}

    class DummySim:
        @classmethod
        def from_directory(cls, path):
            calls["path"] = path
            return cls()

        def optimize_RRI(self, tuned_pa_config, **kwargs):
            calls["tuned_pa_config"] = tuned_pa_config
            calls["kwargs"] = kwargs
            return {
                "LPA": [1.0, 2.0, 3.0],
                "RPA": [4.0, 5.0, 6.0],
                "output_config": str(tmp_path / "out.json"),
            }

    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.SimulationDirectory",
        DummySim,
    )

    result = generate_reduced_pa_from_iteration(
        iteration_dir=tmp_path,
        tuned_pa_config=tmp_path / "pa_config_tuning_snapshot.json",
        nm_iter=7,
        tuning_iter=2,
    )

    assert calls["path"] == str(tmp_path)
    assert calls["tuned_pa_config"].endswith("pa_config_tuning_snapshot.json")
    assert calls["kwargs"]["nm_iter"] == 7
    assert calls["kwargs"]["tuning_iter"] == 2
    assert result["regenerated_config_path"].endswith("out.json")


def test_prepare_reduced_rri_seed_from_learned_fits_and_writes_outputs(monkeypatch, tmp_path: Path):
    learned = tmp_path / "baseline_0d_learned.json"
    template = tmp_path / "template_reduced.json"
    output = tmp_path / "prepared" / "simplified_zerod_tuned_RRI.json"
    metrics = tmp_path / "prepared" / "rri_seed_metrics.json"
    learned.write_text(json.dumps(_learned_bifurcation_payload()), encoding="utf-8")
    template.write_text(json.dumps({"reduced": True}), encoding="utf-8")
    monkeypatch.setattr(
        "svzerodtrees.io.config_handler.simulate_pysvzerod",
        _fake_learned_seed_result,
    )
    monkeypatch.setattr(
        "svzerodtrees.tune_bcs.pa_config.simulate_pysvzerod",
        _fake_learned_seed_result,
    )

    result = prepare_reduced_rri_seed_from_learned(
        learned_config=learned,
        reduced_template=template,
        output_config=output,
        metrics_path=metrics,
    )

    assert result["method"] == "rri_from_learned_reference"
    assert result["reference_metrics"]["P_mpa"] == pytest.approx([30.0, 20.0, 25.0])
    assert result["reference_metrics"]["rpa_split"] == pytest.approx(0.4)
    assert result["optimized_metrics"]["rpa_split"] == pytest.approx(0.4)
    assert result["boundary_conditions"]["LPA_BC"]["bc_values"]["R"] == pytest.approx(1000.0)
    assert result["boundary_conditions"]["RPA_BC"]["bc_values"]["R"] == pytest.approx(1000.0)
    assert result["boundary_conditions"]["LPA_BC"]["bc_values"]["Pd"] == pytest.approx(0.0)
    assert json.loads(output.read_text(encoding="utf-8")) == result["optimized_config"]
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["learned_config"] == str(learned)
    assert payload["source_template"] == str(template)
    assert payload["output_config"] == str(output)


def test_prepare_reduced_rri_seed_from_learned_uses_provided_resistance_bcs(monkeypatch, tmp_path: Path):
    learned = tmp_path / "baseline_0d_learned.json"
    output = tmp_path / "simplified_zerod_tuned_RRI.json"
    learned.write_text(json.dumps(_learned_bifurcation_payload()), encoding="utf-8")
    monkeypatch.setattr(
        "svzerodtrees.io.config_handler.simulate_pysvzerod",
        _fake_learned_seed_result,
    )
    monkeypatch.setattr(
        "svzerodtrees.tune_bcs.pa_config.simulate_pysvzerod",
        _fake_learned_seed_result,
    )

    result = prepare_reduced_rri_seed_from_learned(
        learned_config=learned,
        output_config=output,
        lpa_bc={"R": 1500.0, "Pd": 12.0},
        rpa_bc={
            "bc_name": "custom",
            "bc_type": "RESISTANCE",
            "bc_values": {"R": 2500.0, "Pd": 13.0},
        },
        maxiter=1,
    )

    assert result["boundary_conditions"]["LPA_BC"]["bc_values"] == pytest.approx(
        {"R": 1500.0, "Pd": 12.0}
    )
    assert result["boundary_conditions"]["RPA_BC"]["bc_name"] == "RPA_BC"
    assert result["boundary_conditions"]["RPA_BC"]["bc_values"] == pytest.approx(
        {"R": 2500.0, "Pd": 13.0}
    )


def test_prepare_reduced_rri_seed_from_learned_rejects_non_resistance_bcs(monkeypatch, tmp_path: Path):
    learned = tmp_path / "baseline_0d_learned.json"
    learned.write_text(json.dumps(_learned_bifurcation_payload()), encoding="utf-8")
    monkeypatch.setattr(
        "svzerodtrees.io.config_handler.simulate_pysvzerod",
        _fake_learned_seed_result,
    )

    with pytest.raises(ValueError, match="RESISTANCE"):
        prepare_reduced_rri_seed_from_learned(
            learned_config=learned,
            lpa_bc={
                "bc_name": "LPA_BC",
                "bc_type": "RCR",
                "bc_values": {"Rp": 1.0, "Rd": 2.0, "C": 1e-4, "Pd": 0.0},
            },
        )


def test_write_iteration_json_contract(tmp_path: Path):
    metrics_path = tmp_path / "metrics" / "iteration_metrics.json"
    decision_path = tmp_path / "metrics" / "iteration_decision.json"

    metrics_payload = {"mpa_sys": 50.0, "mpa_dia": 20.0, "mpa_mean": 30.0, "rpa_split": 0.4}
    decision_payload = {
        "decision": "not_close",
        "close_to_targets": False,
        "thresholds": {"mpa_sys": 5.0, "mpa_dia": 3.0, "mpa_mean": 3.0, "rpa_split": 0.05},
        "regenerated_config_path": "simplified_zerod_tuned_RRI.json",
        "postop_submission_requested": False,
    }

    write_iteration_metrics(metrics_path, metrics_payload)
    write_iteration_decision(decision_path, decision_payload)

    metrics_loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    decision_loaded = json.loads(decision_path.read_text(encoding="utf-8"))

    assert metrics_loaded["mpa_sys"] == 50.0
    assert decision_loaded["decision"] == "not_close"


def test_run_impedance_tuning_for_iteration_contract(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    calls: dict[str, object] = {}

    class DummyConfigHandler:
        def __init__(self, path: str):
            self.path = path

        @classmethod
        def from_json(cls, path: str, is_pulmonary: bool = False):
            calls.setdefault("from_json", []).append((path, is_pulmonary))
            return cls(path)

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 12.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=3,
                )),
                encoding="utf-8",
            )
            calls["to_json"] = path

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, path: str):
            calls["targets_path"] = path
            return cls()

    class DummyTuner:
        def __init__(self, *args, **kwargs):
            calls["tune_space"] = args[3]
            calls["tuner_kwargs"] = kwargs

        def tune(self, nm_iter: int = 1):
            calls["nm_iter"] = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0, 0.5], "Pd": 12.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=3,
                    inflow_q=[3.0, 3.0],
                )),
                encoding="utf-8",
            )
            Path(str(calls["tuner_kwargs"]["log_file"])).write_text("log", encoding="utf-8")

    def _fake_construct(
        config_handler,
        mesh_path,
        wedge_p,
        lpa_params,
        rpa_params,
        d_min,
        **kwargs,
    ):
        calls["construct"] = {
            "mesh_path": mesh_path,
            "wedge_p": wedge_p,
            "d_min": d_min,
            "kwargs": kwargs,
            "lpa_params": lpa_params,
            "rpa_params": rpa_params,
        }
        assert config_handler is not None

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.construct_impedance_trees", _fake_construct)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        inflow_path=inflow_path,
        impedance_config={
            "nm_iter": 7,
            "n_procs": 12,
            "use_mean": False,
            "diameter_scale": 0.1,
            "tune_space": _tune_space_with_xi(),
        },
    )

    assert Path(result["optimized_params_csv"]).name == OPTIMIZED_PARAMS_FILENAME
    assert Path(result["stree_optimization_log"]).name == OPTIMIZATION_LOG_FILENAME
    assert Path(result["pa_config_snapshot"]).name == PA_CONFIG_SNAPSHOT_FILENAME
    assert Path(result["tuned_zerod_config"]).exists()
    assert calls["nm_iter"] == 7
    assert calls["construct"]["mesh_path"] == str(mesh_surfaces)
    assert calls["construct"]["wedge_p"] == 12.0
    assert calls["construct"]["kwargs"]["n_procs"] == 12
    assert calls["construct"]["kwargs"]["use_mean"] is False
    assert calls["construct"]["kwargs"]["diameter_scale"] == pytest.approx(0.1)
    assert calls["construct"]["kwargs"]["diameter_std_cap"] is None
    assert calls["tuner_kwargs"]["tuning_model"] == "rri"
    assert calls["tuner_kwargs"]["convert_to_cm"] is False
    free_names = [param.name for param in calls["tune_space"].free]
    assert free_names == [entry["name"] for entry in _tune_space_with_xi()["free"]]
    assert "lpa.alpha" not in free_names
    assert "rpa.alpha" not in free_names
    assert "lpa.beta" not in free_names
    assert "rpa.beta" not in free_names


def test_run_impedance_tuning_for_iteration_full_pa_contract(monkeypatch, tmp_path: Path):
    seed = tmp_path / "full_pa_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    calls: dict[str, object] = {}

    class BC:
        def __init__(self, name: str):
            self.name = name

    class DummyConfigHandler:
        bcs = {
            "INFLOW": BC("INFLOW"),
            "lpa_cap_1": BC("lpa_cap_1"),
            "rpa_cap_1": BC("rpa_cap_1"),
        }

        @classmethod
        def from_json(cls, path: str, is_pulmonary: bool = False):
            calls.setdefault("from_json", []).append((path, is_pulmonary))
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 12.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, path: str):
            calls["targets_path"] = path
            return cls()

    class DummyTuner:
        def __init__(self, *args, **kwargs):
            calls["tuner_kwargs"] = kwargs

        def tune(self, nm_iter: int = 1):
            calls["nm_iter"] = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0, 0.5], "Pd": 12.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=3,
                    inflow_q=[6.0, 6.0],
                )),
                encoding="utf-8",
            )
            Path(str(calls["tuner_kwargs"]["log_file"])).write_text("log", encoding="utf-8")

    def _fake_validate(config_handler, mesh_path, **kwargs):
        calls["validate"] = {
            "config_handler": config_handler,
            "mesh_path": mesh_path,
            "kwargs": kwargs,
        }
        return {"/mesh/lpa_cap_1.vtp": "lpa_cap_1", "/mesh/rpa_cap_1.vtp": "rpa_cap_1"}

    def _fake_construct(config_handler, mesh_path, wedge_p, lpa_params, rpa_params, d_min, **kwargs):
        calls["construct"] = {
            "mesh_path": mesh_path,
            "wedge_p": wedge_p,
            "d_min": d_min,
            "kwargs": kwargs,
        }

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.validate_cap_to_bc_mapping", _fake_validate)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.construct_impedance_trees", _fake_construct)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        inflow_path=inflow_path,
        impedance_config={
            "tuning_model": "full_pa",
            "nm_iter": 3,
            "n_procs": 8,
            "use_mean": False,
            "diameter_scale": 0.25,
            "diameter_std_cap": 1.5,
            "allow_ordered_outlet_mapping": True,
            "tune_space": _tune_space_with_xi(),
        },
    )

    assert result["tuning_model"] == "full_pa"
    assert result["impedance_config"]["diameter_std_cap"] == pytest.approx(1.5)
    assert calls["nm_iter"] == 3
    assert calls["validate"]["mesh_path"] == str(mesh_surfaces)
    assert calls["validate"]["kwargs"]["allow_ordered_outlet_mapping"] is True
    assert calls["tuner_kwargs"]["tuning_model"] == "full_pa"
    assert calls["tuner_kwargs"]["diameter_scale"] == pytest.approx(0.25)
    assert calls["tuner_kwargs"]["diameter_std_cap"] == pytest.approx(1.5)
    assert calls["construct"]["kwargs"]["diameter_std_cap"] == pytest.approx(1.5)


def test_run_impedance_tuning_for_iteration_full_pa_rejects_reduced_snapshot(
    monkeypatch, tmp_path: Path
):
    seed = tmp_path / "full_pa_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_full_pa_multi_outlet_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        bcs = {
            "INFLOW": SimpleNamespace(name="INFLOW"),
            "LPA_1": SimpleNamespace(name="LPA_1"),
            "LPA_2": SimpleNamespace(name="LPA_2"),
            "RPA_1": SimpleNamespace(name="RPA_1"),
            "RPA_2": SimpleNamespace(name="RPA_2"),
        }

        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_full_pa_impedance_snapshot_payload()),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class ReducedSnapshotTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_reduced_rri_impedance_snapshot_payload()),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", ReducedSnapshotTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.validate_cap_to_bc_mapping", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("svzerodtrees.tuning.iteration.construct_impedance_trees", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )

    with pytest.raises(ValueError, match="reduced PA/RRI|lost vessel topology|collapsed"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={
                "tuning_model": "full_pa",
                "rescale_inflow": True,
                "tune_space": _tune_space_with_xi(),
            },
        )


def test_run_impedance_tuning_for_iteration_clears_stale_snapshot(
    monkeypatch, tmp_path: Path
):
    seed = tmp_path / "full_pa_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    results_dir = iteration_dir / "results"
    results_dir.mkdir(parents=True)
    seed.write_text(json.dumps(_full_pa_multi_outlet_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")
    (results_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
        json.dumps(_reduced_rri_impedance_snapshot_payload()),
        encoding="utf-8",
    )

    class DummyConfigHandler:
        bcs = {
            "INFLOW": SimpleNamespace(name="INFLOW"),
            "LPA_1": SimpleNamespace(name="LPA_1"),
            "LPA_2": SimpleNamespace(name="LPA_2"),
            "RPA_1": SimpleNamespace(name="RPA_1"),
            "RPA_2": SimpleNamespace(name="RPA_2"),
        }

        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class NoSnapshotTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            (Path.cwd() / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", NoSnapshotTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.validate_cap_to_bc_mapping", lambda *_args, **_kwargs: {})

    with pytest.raises(FileNotFoundError, match=PA_CONFIG_SNAPSHOT_FILENAME):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={
                "tuning_model": "full_pa",
                "rescale_inflow": True,
                "tune_space": _tune_space_with_xi(),
            },
        )
    assert not (results_dir / PA_CONFIG_SNAPSHOT_FILENAME).exists()


def test_run_impedance_tuning_for_iteration_rri_expands_reduced_bcs_for_caps(
    monkeypatch, tmp_path: Path
):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_learned_bifurcation_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    calls: dict[str, object] = {"construct_bc_names": []}

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **kwargs):
            calls["tuner_kwargs"] = kwargs
            self.log_file = kwargs["log_file"]

        def tune(self, nm_iter: int = 1):
            calls["nm_iter"] = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            snapshot = _reduced_rri_impedance_snapshot_payload()
            snapshot["boundary_conditions"][0]["bc_values"]["Q"] = [3.0, 3.0]
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(snapshot),
                encoding="utf-8",
            )
            Path(str(self.log_file)).write_text("log", encoding="utf-8")

    def _fake_construct(config_handler, *_args, **_kwargs):
        outlet_names = [
            name
            for name, bc in config_handler.bcs.items()
            if "inflow" not in str(getattr(bc, "name", name)).lower()
        ]
        calls["construct_bc_names"].append(outlet_names)
        if len(outlet_names) != 4:
            raise ValueError(
                "number of outlet boundary conditions does not match number of cap "
                f"surfaces: bcs={len(outlet_names)}, caps=4"
            )
        for name in outlet_names:
            config_handler.bcs[name] = SimpleNamespace(
                name=name,
                type="IMPEDANCE",
                values={"z": [1.0], "Pd": 12.0},
                Z=[1.0],
                Pd=12.0,
                to_dict=lambda name=name: {
                    "bc_name": name,
                    "bc_type": "IMPEDANCE",
                    "bc_values": {"z": [1.0], "Pd": 12.0},
                },
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.construct_impedance_trees", _fake_construct)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.vtp_info",
        lambda *_args, **_kwargs: (
            {"rpa_cap_1.vtp": 2.0, "rpa_cap_2.vtp": 8.0},
            {"lpa_cap_1.vtp": 1.0, "lpa_cap_2.vtp": 4.0},
            {},
        ),
    )
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        inflow_path=inflow_path,
        impedance_config={
            "tuning_model": "rri",
            "rescale_inflow": True,
            "tune_space": _tune_space_with_xi(),
        },
    )

    assert Path(result["tuned_zerod_config"]).exists()
    assert calls["construct_bc_names"][0] == ["LPA_BC", "RPA_BC"]
    assert calls["construct_bc_names"][1] == [
        "lpa_cap_1",
        "lpa_cap_2",
        "rpa_cap_1",
        "rpa_cap_2",
    ]


def test_full_pa_tuner_loss_applies_trial_bcs_and_writes_csv(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)

    class BC:
        def __init__(self, name: str, bc_type: str = "RESISTANCE"):
            self.name = name
            self.type = bc_type
            self.Q = [6.0, 6.0]
            self.t = [0.0, 1.0]
            self.Z = [1.0]

    class DummyModel:
        def __init__(self):
            self.bcs = {
                "INFLOW": BC("INFLOW", "FLOW"),
                "lpa_cap_1": BC("lpa_cap_1"),
                "rpa_cap_1": BC("rpa_cap_1"),
            }
            self.simparams = SimpleNamespace(cardiac_period=1.0)
            self.mpa = SimpleNamespace(name="branch0_seg0", branch=0)
            self.rpa = SimpleNamespace(name="branch2_seg0", branch=2)

        def to_json(self, path: str):
            Path(path).write_text(json.dumps({"boundary_conditions": []}), encoding="utf-8")

        def simulate(self):
            return pd.DataFrame(
                {
                    "name": ["branch0_seg0", "branch0_seg0", "branch2_seg0", "branch2_seg0"],
                    "time": [0.0, 1.0, 0.0, 1.0],
                    "pressure_in": [40.0 * 1333.2, 20.0 * 1333.2, 10.0, 10.0],
                    "flow_in": [10.0, 10.0, 4.0, 4.0],
                }
            )

    def _fake_construct(model, mesh_path, wedge_p, lpa_params, rpa_params, d_min, **kwargs):
        calls["construct"] = {
            "mesh_path": mesh_path,
            "wedge_p": wedge_p,
            "d_min": d_min,
            "kwargs": kwargs,
        }
        model.bcs["lpa_cap_1"] = BC("lpa_cap_1", "IMPEDANCE")
        model.bcs["rpa_cap_1"] = BC("rpa_cap_1", "IMPEDANCE")

    monkeypatch.setattr("svzerodtrees.tune_bcs.impedance_tuner.construct_impedance_trees", _fake_construct)
    monkeypatch.setattr("svzerodtrees.tune_bcs.impedance_tuner.validate_boundary_condition_configs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("svzerodtrees.tune_bcs.impedance_tuner.validate_impedance_timing_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("svzerodtrees.tune_bcs.impedance_tuner.validate_flow_cardiac_output_config", lambda *_args, **_kwargs: None)

    tune_space = _build_tune_space_from_config(
        _resolve_impedance_config(
            {
                "compliance_model": "constant",
                "tune_space": _constant_tune_space(),
            }
        )["tune_space"]
    )
    tuner = ImpedanceTuner(
        DummyModel(),
        str(tmp_path / "mesh-surfaces"),
        SimpleNamespace(mpa_p=[40.0, 20.0, 30.0], rpa_split=0.4, wedge_p=12.0),
        tune_space,
        compliance_model="constant",
        rescale_inflow=False,
        tuning_model="full_pa",
        use_mean=False,
        diameter_scale=0.5,
        diameter_std_cap=2.0,
        allow_ordered_outlet_mapping=True,
    )
    tuner._geom_defaults = {
        "lpa.default_diameter": 0.3,
        "rpa.default_diameter": 0.4,
        "n_outlets_scale": 1.0,
    }
    tuner._full_pa_base_config = DummyModel()
    tuner._expected_snapshot_cardiac_output = 6.0
    tuner._opt_csv_path = str(tmp_path / OPTIMIZED_PARAMS_FILENAME)

    x0, _ = tune_space.pack_init_and_bounds()
    loss = tuner.loss_fn(x0, tuner._full_pa_base_config, finalize=True)

    assert loss > 0.0
    assert calls["construct"]["kwargs"]["use_mean"] is False
    assert calls["construct"]["kwargs"]["diameter_scale"] == pytest.approx(0.5)
    assert calls["construct"]["kwargs"]["diameter_std_cap"] == pytest.approx(2.0)
    assert (tmp_path / OPTIMIZED_PARAMS_FILENAME).exists()
    assert (tmp_path / PA_CONFIG_SNAPSHOT_FILENAME).exists()


def test_run_impedance_tuning_for_iteration_missing_snapshot_raises(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text("{}", encoding="utf-8")

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            (Path.cwd() / OPTIMIZED_PARAMS_FILENAME).write_text("pa\nlpa\nrpa\n", encoding="utf-8")

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)

    with pytest.raises(FileNotFoundError, match=PA_CONFIG_SNAPSHOT_FILENAME):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_missing_required_xi_raises(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text("{}", encoding="utf-8")

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[3.0, 3.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: (_ for _ in ()).throw(AssertionError("_load_tree_params should not be called")),
    )

    with pytest.raises(ValueError, match="must include an 'xi' column"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_rejects_legacy_snapshot(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(
                    {
                        "simulation_parameters": {
                            "number_of_cardiac_cycles": 1,
                            "number_of_time_pts_per_cardiac_cycle": 2,
                        },
                        "boundary_conditions": [],
                    }
                ),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"Z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)

    with pytest.raises(ValueError, match="unsupported keys: Z"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_rejects_legacy_tuned_config(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"tree": 0, "z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[3.0, 3.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.construct_impedance_trees",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="unsupported keys: tree"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_rejects_snapshot_timestep_mismatch(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=3,
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)

    with pytest.raises(ValueError, match="number_of_time_pts_per_cardiac_cycle = len\\(z\\) \\+ 1"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_rejects_coupled_tuned_config_wrong_number_of_time_pts(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=3,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[3.0, 3.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.construct_impedance_trees",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="number_of_time_pts = 2; got 3"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_rejects_snapshot_inflow_mismatch(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload(inflow_q=[6.0, 6.0])), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[6.0, 6.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)

    with pytest.raises(ValueError, match="cardiac output mismatch"):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
            inflow_path=inflow_path,
            impedance_config={"tune_space": _tune_space_with_xi()},
        )


def test_run_impedance_tuning_for_iteration_accepts_scaled_snapshot_inflow(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = _write_constant_inflow_csv(tmp_path, 6.0)
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload(inflow_q=[6.0, 6.0])), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[3.0, 3.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.construct_impedance_trees",
        lambda *args, **kwargs: None,
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        inflow_path=inflow_path,
        impedance_config={"tune_space": _tune_space_with_xi()},
    )

    assert Path(result["pa_config_snapshot"]).name == PA_CONFIG_SNAPSHOT_FILENAME


def test_run_impedance_tuning_for_iteration_uses_unscaled_seed_inflow_when_rescale_disabled(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload(inflow_q=[6.0, 6.0])), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[6.0, 6.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.construct_impedance_trees",
        lambda *args, **kwargs: None,
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        impedance_config={
            "rescale_inflow": False,
            "tune_space": _tune_space_with_xi(),
        },
    )

    assert Path(result["pa_config_snapshot"]).name == PA_CONFIG_SNAPSHOT_FILENAME


def test_run_impedance_tuning_for_iteration_uses_inflow_file_mean_source(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    inflow_path = tmp_path / "inflow.csv"
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload(inflow_q=[2.0, 2.0])), encoding="utf-8")
    inflow_path.write_text("t,q\n0.0,8.0\n1.0,8.0\n", encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyConfigHandler:
        @classmethod
        def from_json(cls, _path: str, is_pulmonary: bool = False):
            return cls()

        def to_json(self, path: str):
            Path(path).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=True,
                    number_of_time_pts=2,
                    number_of_time_pts_per_cardiac_cycle=2,
                )),
                encoding="utf-8",
            )

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def tune(self, nm_iter: int = 1):
            _ = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,xi,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,2.3,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                json.dumps(_impedance_artifact_payload(
                    bc_values={"z": [1.0], "Pd": 10.0},
                    coupled=False,
                    number_of_time_pts_per_cardiac_cycle=2,
                    inflow_q=[4.0, 4.0],
                )),
                encoding="utf-8",
            )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", DummyConfigHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ImpedanceTuner", DummyTuner)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.get_pa_outlet_scale", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration.construct_impedance_trees",
        lambda *args, **kwargs: None,
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        inflow_path=inflow_path,
        impedance_config={"tune_space": _tune_space_with_xi()},
    )

    assert Path(result["pa_config_snapshot"]).name == PA_CONFIG_SNAPSHOT_FILENAME


def test_run_impedance_tuning_for_iteration_requires_inflow_path_when_rescaling(tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    iteration_dir = tmp_path / "iter-01"
    seed.write_text(json.dumps(_seed_config_payload()), encoding="utf-8")
    mesh_surfaces.mkdir(parents=True, exist_ok=True)
    targets.write_text("target,value\n", encoding="utf-8")

    class DummyClinicalTargets:
        wedge_p = 10.0

        @classmethod
        def from_csv(cls, _path: str):
            return cls()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("svzerodtrees.tuning.iteration.ClinicalTargets", DummyClinicalTargets)
    with pytest.raises(ValueError, match="rescale_inflow=True requires inflow_path"):
        try:
            run_impedance_tuning_for_iteration(
                iteration_dir=iteration_dir,
                seed_config=seed,
                mesh_surfaces=mesh_surfaces,
                clinical_targets=targets,
                impedance_config={"tune_space": _tune_space_with_xi()},
            )
        finally:
            monkeypatch.undo()


def test_resolve_impedance_config_requires_explicit_tune_space():
    with pytest.raises(ValueError, match="must include explicit tune_space"):
        _resolve_impedance_config(None)

    with pytest.raises(ValueError, match="must include explicit tune_space"):
        _resolve_impedance_config({"solver": "Nelder-Mead"})


def test_resolve_impedance_config_supports_tuning_model_and_diameter_std_cap():
    cfg = _resolve_impedance_config(
        {
            "tuning_model": "full_pa",
            "diameter_std_cap": 1.25,
            "tune_space": _tune_space_with_xi(),
        }
    )
    assert cfg["tuning_model"] == "full_pa"
    assert cfg["diameter_std_cap"] == pytest.approx(1.25)


def test_resolve_impedance_config_rejects_invalid_tuning_model_and_diameter_std_cap():
    with pytest.raises(ValueError, match="tuning_model"):
        _resolve_impedance_config(
            {
                "tuning_model": "invalid",
                "tune_space": _tune_space_with_xi(),
            }
        )

    with pytest.raises(ValueError, match="diameter_std_cap"):
        _resolve_impedance_config(
            {
                "diameter_std_cap": -1.0,
                "tune_space": _tune_space_with_xi(),
            }
        )


def test_build_tune_space_from_config_supports_inf_and_transforms():
    cfg = {
        "free": [
            {
                "name": "lpa.inductance",
                "init": 1.0,
                "lb": 0.0,
                "ub": "inf",
                "to_native": "positive",
                "from_native": "log",
            }
        ],
        "fixed": [{"name": "lrr", "value": 10.0}],
        "tied": [{"name": "rpa.inductance", "other": "lpa.inductance", "fn": "identity"}],
    }
    tune_space = _build_tune_space_from_config(_resolve_impedance_config({"tune_space": cfg})["tune_space"])
    assert len(tune_space.free) == 1
    assert tune_space.free[0].name == "lpa.inductance"
    assert tune_space.free[0].ub == pytest.approx(float("inf"))
    assert tune_space.tied[0].name == "rpa.inductance"


def test_resolve_impedance_config_invalid_transform_raises():
    with pytest.raises(ValueError, match="to_native"):
        _resolve_impedance_config(
            {
                "tune_space": {
                    "free": [
                        {
                            "name": "lpa.inductance",
                            "init": 1.0,
                            "lb": 0.0,
                            "ub": "inf",
                            "to_native": "bad",
                        }
                    ],
                    "fixed": [],
                    "tied": [],
                }
            }
        )


def test_resolve_impedance_config_invalid_inf_string_raises():
    with pytest.raises(ValueError, match="bound strings"):
        _resolve_impedance_config(
            {
                "tune_space": {
                    "free": [
                        {
                            "name": "lpa.inductance",
                            "init": 1.0,
                            "lb": 0.0,
                            "ub": "infinity",
                        }
                    ],
                    "fixed": [],
                    "tied": [],
                }
            }
        )


def test_resolve_impedance_config_missing_tune_space_keys_raises():
    with pytest.raises(ValueError, match="must define keys: free, fixed, tied"):
        _resolve_impedance_config({"tune_space": {"free": []}})
