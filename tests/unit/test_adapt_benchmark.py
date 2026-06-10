from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from svzerodtrees.adaptation.benchmark import (
    run_adaptation_benchmark_study,
    run_reduced_pa_adaptation_case,
)
from svzerodtrees.config import AdaptBenchmarkConfig, AdaptBenchmarkScenarioConfig, load_config


class FakeStore:
    def __init__(self, diameters):
        self.d = np.asarray(diameters, dtype=float)

    def n_nodes(self):
        return self.d.size


class FakeTree:
    def __init__(self, diameters):
        self.store = FakeStore(diameters)


class FakeTargets:
    def __init__(self):
        self.rpa_split = 0.46
        self.mpa_p = [32.0, 12.0, 21.0]
        self.wedge_p = 8.0


class FakeBlock:
    def __init__(self, resistance):
        self.R = float(resistance)


class FakePA:
    def __init__(
        self,
        *,
        lpa_prox_r: float,
        rpa_prox_r: float,
        lpa_tree_d=(2.0, 2.5),
        rpa_tree_d=(1.4, 1.6),
    ):
        self.lpa_prox = FakeBlock(lpa_prox_r)
        self.rpa_prox = FakeBlock(rpa_prox_r)
        self.lpa_tree = FakeTree(lpa_tree_d)
        self.rpa_tree = FakeTree(rpa_tree_d)
        self.bcs = {"LPA_BC": FakeBlock(0.0), "RPA_BC": FakeBlock(0.0)}
        self.clinical_targets = FakeTargets()
        self.result = None
        self.P_mpa = [30.0, 10.0, 20.0]
        self.rpa_split = None
        self.update_bcs()
        self.simulate()

    def update_bcs(self):
        self.bcs["LPA_BC"].R = 120.0 / float(np.mean(self.lpa_tree.store.d))
        self.bcs["RPA_BC"].R = 120.0 / float(np.mean(self.rpa_tree.store.d))

    def simulate(self):
        lpa_total_r = float(self.lpa_prox.R) + float(self.bcs["LPA_BC"].R)
        rpa_total_r = float(self.rpa_prox.R) + float(self.bcs["RPA_BC"].R)
        lpa_g = 1.0 / max(lpa_total_r, 1e-12)
        rpa_g = 1.0 / max(rpa_total_r, 1e-12)
        total_q = 100.0
        total_g = lpa_g + rpa_g
        lpa_flow = total_q * lpa_g / total_g
        rpa_flow = total_q - lpa_flow
        self.rpa_split = float(rpa_flow / total_q)
        self.P_mpa = [30.0 + self.rpa_split, 10.0, 20.0 + self.rpa_split]
        self.result = pd.DataFrame(
            {
                "name": [
                    "branch0_seg0",
                    "branch0_seg0",
                    "branch2_seg0",
                    "branch2_seg0",
                    "branch4_seg0",
                    "branch4_seg0",
                ],
                "pressure_in": [
                    self.P_mpa[0] * 1333.2,
                    self.P_mpa[2] * 1333.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "flow_out": [0.0, 0.0, lpa_flow, lpa_flow, rpa_flow, rpa_flow],
            }
        )
        return self.result


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _write_benchmark_config(tmp_path: Path, models: str = "[M1, M2, M3]", scenarios: str | None = None) -> Path:
    preop = _touch(tmp_path / "preop.json")
    postop = _touch(tmp_path / "postop.json")
    tree_params = _touch(tmp_path / "optimized_params.csv")
    clinical = _touch(tmp_path / "clinical_targets.csv")
    if scenarios is None:
        scenarios = f"""
    - name: baseline
      preop_rri_config: {preop}
      postop_rri_config: {postop}
"""
    cfg = tmp_path / "adapt_benchmark.yml"
    cfg.write_text(
        f"""
version: 1
workflow: adapt_benchmark
paths:
  root: {tmp_path}
adapt_benchmark:
  study_id: tst-study
  output_dir: outputs
  models: {models}
  tree_params_csv: {tree_params}
  clinical_targets_csv: {clinical}
  scenarios:
{scenarios}
""",
        encoding="utf-8",
    )
    return cfg


def _fake_initialize_from_paths(preop_path, postop_path, optimized_tree_params_csv, clinical_targets_csv, *, max_nodes=100_000):
    del preop_path, postop_path, optimized_tree_params_csv, clinical_targets_csv, max_nodes
    preop = FakePA(lpa_prox_r=30.0, rpa_prox_r=55.0, lpa_tree_d=(2.3, 2.4), rpa_tree_d=(1.3, 1.4))
    postop = FakePA(lpa_prox_r=28.0, rpa_prox_r=75.0, lpa_tree_d=(2.0, 2.1), rpa_tree_d=(1.3, 1.4))
    return preop, postop


def _fake_run_adaptation(preop_pa, postop_pa, model_cls, gain_arr, **kwargs):
    del gain_arr, kwargs
    adapted = postop_pa
    if model_cls.__name__ == "CWSSAdaptation":
        adapted.lpa_tree.store.d = adapted.lpa_tree.store.d * 1.10
        adapted.rpa_tree.store.d = adapted.rpa_tree.store.d * 0.92
        termination_reason = "rpa_split_window_converged"
        final_split = 0.43
    else:
        adapted.lpa_tree.store.d = adapted.lpa_tree.store.d * 1.04
        adapted.rpa_tree.store.d = adapted.rpa_tree.store.d * 1.08
        termination_reason = "geometry_converged"
        final_split = 0.49
    adapted.update_bcs()
    adapted.simulate()
    adapted.rpa_split = final_split
    fig = plt.figure()
    result = {
        "stable": 1,
        "geom_err": 0.02,
        "t95": 15.0,
        "n_rhs": 9,
        "preop_rpa_split": float(preop_pa.rpa_split),
        "postop_rpa_split": 0.41,
        "final_rpa_split": final_split,
        "solver_diagnostics": {
            "termination_reason": termination_reason,
            "event_time": 24.0,
            "accepted_step_flow_split_history": [
                {"t": 0.0, "rpa_split": 0.41},
                {"t": 12.0, "rpa_split": 0.45},
                {"t": 24.0, "rpa_split": final_split},
            ],
        },
    }
    return result, list(result["solver_diagnostics"]["accepted_step_flow_split_history"]), None, adapted, [fig]


def _fake_run_adaptation_collapse(preop_pa, postop_pa, model_cls, gain_arr, **kwargs):
    result, flow_log, sol, adapted, hists = _fake_run_adaptation(
        preop_pa,
        postop_pa,
        model_cls,
        gain_arr,
        **kwargs,
    )
    if model_cls.__name__ == "CWSSIMSAdaptation":
        result["final_rpa_split"] = 0.004
        result["solver_diagnostics"]["radius_change"] = {
            "lpa_radius": {"mean_relative_change": 2.0, "max_abs_relative_change": 12.0},
            "rpa_radius": {"mean_relative_change": -0.98, "max_abs_relative_change": 0.98},
            "lpa_thickness": {"mean_relative_change": 0.0, "max_abs_relative_change": 0.0},
            "rpa_thickness": {"mean_relative_change": 0.0, "max_abs_relative_change": 0.0},
        }
    return result, flow_log, sol, adapted, hists


def test_load_config_parses_adapt_benchmark(tmp_path):
    cfg_path = _write_benchmark_config(tmp_path)
    cfg = load_config(str(cfg_path))

    assert cfg.workflow == "adapt_benchmark"
    assert cfg.adapt_benchmark is not None
    assert cfg.adapt_benchmark.study_id == "tst-study"
    assert cfg.adapt_benchmark.models == ["M1", "M2", "M3"]
    assert cfg.adapt_benchmark.scenarios[0].name == "baseline"


def test_load_config_parses_adapt_benchmark_stage3_metadata(tmp_path):
    preop = _touch(tmp_path / "preop.json")
    postop = _touch(tmp_path / "postop.json")
    tree_params = _touch(tmp_path / "optimized_params.csv")
    clinical = _touch(tmp_path / "clinical_targets.csv")
    cfg = tmp_path / "adapt_benchmark.yml"
    cfg.write_text(
        f"""
version: 1
workflow: adapt_benchmark
paths:
  root: {tmp_path}
adapt_benchmark:
  study_id: stage3
  output_dir: outputs
  models: [M3]
  tree_params_csv: {tree_params}
  clinical_targets_csv: {clinical}
  scenarios:
    - name: tst_stan_1_medium__cwss_ims_equal_gains
      patient_id: tst-stan-1
      scenario_group: medium
      perturbation_severity: medium
      preop_rri_config: {preop}
      postop_rri_config: {postop}
""",
        encoding="utf-8",
    )

    parsed = load_config(str(cfg)).adapt_benchmark

    assert parsed is not None
    assert parsed.scenarios[0].patient_id == "tst-stan-1"
    assert parsed.scenarios[0].scenario_group == "medium"
    assert parsed.scenarios[0].perturbation_severity == "medium"


def test_load_config_rejects_invalid_adapt_benchmark_models(tmp_path):
    cfg_path = _write_benchmark_config(tmp_path, models="[M1, BAD]")
    with pytest.raises(ValueError, match="adapt_benchmark.models"):
        load_config(str(cfg_path))


def test_load_config_rejects_empty_adapt_benchmark_scenarios(tmp_path):
    cfg_path = _write_benchmark_config(tmp_path, scenarios="")
    with pytest.raises(ValueError, match="at least one scenario"):
        load_config(str(cfg_path))


def test_run_reduced_pa_adaptation_case_validates_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="preop_rri_config"):
        run_reduced_pa_adaptation_case(
            scenario_name="baseline",
            model="M1",
            parameter_set={},
            preop_rri_config=str(tmp_path / "missing-preop.json"),
            postop_rri_config=str(tmp_path / "missing-postop.json"),
            tree_params_csv=str(tmp_path / "missing-params.csv"),
            clinical_targets_csv=str(tmp_path / "missing-clinical.csv"),
            output_dir=tmp_path / "out",
        )


def test_run_adaptation_benchmark_study_writes_outputs(monkeypatch, tmp_path):
    preop = _touch(tmp_path / "preop.json")
    postop = _touch(tmp_path / "postop.json")
    tree_params = _touch(tmp_path / "optimized_params.csv")
    clinical = _touch(tmp_path / "clinical_targets.csv")
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.initialize_from_paths",
        _fake_initialize_from_paths,
    )
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.run_adaptation",
        _fake_run_adaptation,
    )

    payload = run_adaptation_benchmark_study(
        AdaptBenchmarkConfig(
            study_id="study-001",
            output_dir=str(tmp_path / "benchmark"),
            models=["M1", "M2", "M3"],
            tree_params_csv=str(tree_params),
            clinical_targets_csv=str(clinical),
            scenarios=[
                AdaptBenchmarkScenarioConfig(
                    name="baseline",
                    preop_rri_config=str(preop),
                    postop_rri_config=str(postop),
                    parameter_overrides={
                        "M3": {
                            "benchmark_case": "cwss_ims_equal_gains",
                            "terminal_load_policy": "explicit_terminal_resistance",
                            "terminal_resistance": 1000.0,
                            "k_arr": [1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4],
                        }
                    },
                )
            ],
        )
    )

    assert payload["study_id"] == "study-001"
    assert len(payload["rows"]) == 3
    rows = {(row["scenario"], row["model"]): row for row in payload["rows"]}
    assert rows[("baseline", "M1")]["update_mode"] == "dynamic"
    assert rows[("baseline", "M2")]["update_mode"] == "algebraic"
    assert rows[("baseline", "M2")]["event_time"] is None
    assert rows[("baseline", "M3")]["termination_reason"] == "geometry_converged"
    assert rows[("baseline", "M3")]["case_name"] == "cwss_ims_equal_gains"
    assert rows[("baseline", "M3")]["terminal_resistance"] == 1000.0
    assert rows[("baseline", "M3")]["k_tau_h"] == pytest.approx(1.0e-4)
    assert rows[("baseline", "M3")]["final_lpa_split"] == pytest.approx(0.51)
    assert Path(payload["summary_csv"]).exists()
    assert Path(payload["summary_json"]).exists()
    assert Path(payload["artifacts"]["benchmark_stability_convergence_png"]).exists()
    assert Path(payload["artifacts"]["benchmark_final_rpa_split_png"]).exists()
    assert Path(payload["artifacts"]["benchmark_rpa_split_overlay_png"]).exists()
    assert Path(payload["artifacts"]["benchmark_lpa_split_overlay_png"]).exists()
    assert Path(payload["artifacts"]["benchmark_aggregate_final_rpa_split_png"]).exists()
    assert Path(payload["artifacts"]["benchmark_convergence_table_csv"]).exists()
    assert Path(payload["artifacts"]["benchmark_failure_table_csv"]).exists()
    summary_csv = pd.read_csv(payload["summary_csv"])
    assert {
        "case_name",
        "patient_id",
        "scenario_group",
        "perturbation_severity",
        "terminal_load_policy",
        "terminal_resistance",
        "k_tau_r",
        "k_sig_r",
        "k_tau_h",
        "k_sig_h",
        "final_lpa_split",
        "lpa_radius_mean_relative_change",
        "lpa_radius_max_abs_relative_change",
        "rpa_thickness_mean_relative_change",
        "rpa_thickness_max_abs_relative_change",
        "one_branch_collapse",
        "radius_bounds_violation",
        "thickness_bounds_violation",
        "nonfinite_state_detected",
        "nonphysical_terminal_load",
        "stability_screen_failed",
    }.issubset(set(summary_csv.columns))
    assert (
        tmp_path
        / "benchmark"
        / "baseline"
        / "m1"
        / "adaptation_summary.json"
    ).exists()
    assert (
        tmp_path
        / "benchmark"
        / "baseline"
        / "m3"
        / "flow_split_history.png"
    ).exists()


def test_run_adaptation_benchmark_study_writes_stage3_screening_outputs(monkeypatch, tmp_path):
    preop = _touch(tmp_path / "preop.json")
    postop = _touch(tmp_path / "postop.json")
    tree_params = _touch(tmp_path / "optimized_params.csv")
    clinical = _touch(tmp_path / "clinical_targets.csv")
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.initialize_from_paths",
        _fake_initialize_from_paths,
    )
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.run_adaptation",
        _fake_run_adaptation_collapse,
    )

    payload = run_adaptation_benchmark_study(
        AdaptBenchmarkConfig(
            study_id="study-stage3",
            output_dir=str(tmp_path / "benchmark"),
            models=["M3"],
            tree_params_csv=str(tree_params),
            clinical_targets_csv=str(clinical),
            scenarios=[
                AdaptBenchmarkScenarioConfig(
                    name="tst_stan_1_medium__cwss_ims_equal_gains",
                    patient_id="tst-stan-1",
                    scenario_group="medium",
                    perturbation_severity="medium",
                    preop_rri_config=str(preop),
                    postop_rri_config=str(postop),
                    parameter_overrides={
                        "M3": {
                            "benchmark_case": "cwss_ims_equal_gains",
                            "k_arr": [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2],
                        }
                    },
                )
            ],
        )
    )

    row = payload["rows"][0]
    assert row["patient_id"] == "tst-stan-1"
    assert row["scenario_group"] == "medium"
    assert row["perturbation_severity"] == "medium"
    assert row["one_branch_collapse"] == 1
    assert row["radius_bounds_violation"] == 1
    assert row["stability_screen_failed"] == 1
    assert Path(payload["artifacts"]["benchmark_tst-stan-1_rpa_split_overlay_png"]).exists()

    failure_table = pd.read_csv(payload["artifacts"]["benchmark_failure_table_csv"])
    assert failure_table.loc[0, "case_name"] == "cwss_ims_equal_gains"
    convergence_table = pd.read_csv(payload["artifacts"]["benchmark_convergence_table_csv"])
    assert convergence_table.loc[0, "n_screen_failed"] == 1


def test_run_adaptation_benchmark_study_supports_multiple_scenarios(monkeypatch, tmp_path):
    preop = _touch(tmp_path / "preop.json")
    postop = _touch(tmp_path / "postop.json")
    tree_params = _touch(tmp_path / "optimized_params.csv")
    clinical = _touch(tmp_path / "clinical_targets.csv")
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.initialize_from_paths",
        _fake_initialize_from_paths,
    )
    monkeypatch.setattr(
        "svzerodtrees.adaptation.benchmark.run_adaptation",
        _fake_run_adaptation,
    )

    payload = run_adaptation_benchmark_study(
        AdaptBenchmarkConfig(
            study_id="study-002",
            output_dir=str(tmp_path / "benchmark"),
            models=["M1", "M2", "M3"],
            tree_params_csv=str(tree_params),
            clinical_targets_csv=str(clinical),
            scenarios=[
                AdaptBenchmarkScenarioConfig(
                    name="baseline",
                    preop_rri_config=str(preop),
                    postop_rri_config=str(postop),
                ),
                AdaptBenchmarkScenarioConfig(
                    name="variant",
                    preop_rri_config=str(preop),
                    postop_rri_config=str(postop),
                ),
            ],
        )
    )

    assert len(payload["rows"]) == 6
