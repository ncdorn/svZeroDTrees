from __future__ import annotations

import json
from pathlib import Path

import pytest

from svzerodtrees.tuning.iteration import (
    OPTIMIZATION_LOG_FILENAME,
    OPTIMIZED_PARAMS_FILENAME,
    PA_CONFIG_SNAPSHOT_FILENAME,
    compute_centerline_mpa_metrics,
    compute_flow_split_metrics,
    evaluate_iteration_gate,
    generate_reduced_pa_from_iteration,
    run_impedance_tuning_for_iteration,
    write_iteration_decision,
    write_iteration_metrics,
)


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
    iteration_dir = tmp_path / "iter-01"
    seed.write_text("{\"seed\": true}", encoding="utf-8")
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
            Path(path).write_text("{\"tuned\": true}", encoding="utf-8")
            calls["to_json"] = path

    class DummyClinicalTargets:
        wedge_p = 12.0

        @classmethod
        def from_csv(cls, path: str):
            calls["targets_path"] = path
            return cls()

    class DummyTuner:
        def __init__(self, *_args, **kwargs):
            calls["tuner_kwargs"] = kwargs

        def tune(self, nm_iter: int = 1):
            calls["nm_iter"] = nm_iter
            out_dir = Path.cwd()
            (out_dir / OPTIMIZED_PARAMS_FILENAME).write_text(
                "pa,alpha,beta,d_min,lrr,diameter,compliance_model,C\n"
                "lpa,0.9,0.6,0.01,10.0,0.3,constant,66000\n"
                "rpa,0.9,0.6,0.01,10.0,0.3,constant,66000\n",
                encoding="utf-8",
            )
            (out_dir / PA_CONFIG_SNAPSHOT_FILENAME).write_text(
                "{\"snapshot\": true}",
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
    monkeypatch.setattr(
        "svzerodtrees.tuning.iteration._load_tree_params",
        lambda _path: ("lpa-params", "rpa-params"),
    )

    result = run_impedance_tuning_for_iteration(
        iteration_dir=iteration_dir,
        seed_config=seed,
        mesh_surfaces=mesh_surfaces,
        clinical_targets=targets,
        impedance_config={"nm_iter": 7, "n_procs": 12, "use_mean": False},
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


def test_run_impedance_tuning_for_iteration_missing_snapshot_raises(monkeypatch, tmp_path: Path):
    seed = tmp_path / "simplified_nonlinear_zerod.json"
    mesh_surfaces = tmp_path / "mesh-surfaces"
    targets = tmp_path / "clinical_targets.csv"
    iteration_dir = tmp_path / "iter-01"
    seed.write_text("{\"seed\": true}", encoding="utf-8")
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

    with pytest.raises(FileNotFoundError, match=PA_CONFIG_SNAPSHOT_FILENAME):
        run_impedance_tuning_for_iteration(
            iteration_dir=iteration_dir,
            seed_config=seed,
            mesh_surfaces=mesh_surfaces,
            clinical_targets=targets,
        )
