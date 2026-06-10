from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

import svzerodtrees.adaptation.models.cwss as cwss_module
import svzerodtrees.adaptation.workflow as workflow_module
from svzerodtrees.adaptation.experiment import run_minipa_cwss_debug, run_single_tree_cwss_debug
from svzerodtrees.adaptation.integrator import run_adaptation
from svzerodtrees.adaptation.models.cwss import CWSSAdaptation
from svzerodtrees.adaptation.utils import wrap_event
from svzerodtrees.adaptation.workflow import run_structured_tree_adaptation


class FakeStore:
    def __init__(self, diameters, ids):
        self.d = np.asarray(diameters, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=np.int32)

    def n_nodes(self):
        return self.d.size


class FakeTree:
    def __init__(self, diameters, ids, *, results=None):
        self.store = FakeStore(diameters, ids)
        self.results = results


class FakePA:
    def __init__(self, lpa_tree=None, rpa_tree=None):
        self.lpa_tree = lpa_tree
        self.rpa_tree = rpa_tree
        self.rpa_split = 0.55
        self.update_calls = 0
        self.sim_calls = 0
        self.clinical_targets = SimpleNamespace(wedge_p=8.0)
        self.result = pd.DataFrame(
            {
                "name": ["branch2_seg0", "branch4_seg0"],
                "flow_out": [2.0, 3.0],
            }
        )

    def update_bcs(self):
        self.update_calls += 1

    def simulate(self):
        self.sim_calls += 1


def _model():
    return CWSSAdaptation([1.0, 0.0, 0.0, 0.0])


def test_constructor_validates_gain_array_type_and_length():
    with pytest.raises(TypeError, match="K_arr"):
        CWSSAdaptation("bad")

    with pytest.raises(ValueError, match="exactly 4"):
        CWSSAdaptation([1.0, 2.0, 3.0])

    assert _model().K_arr == [1.0, 0.0, 0.0, 0.0]


def test_compute_rhs_rejects_invalid_state_shape_and_values():
    simple_pa = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    with pytest.raises(ValueError, match="1-D"):
        _model().compute_rhs(0.0, [[1.0, 2.0]], simple_pa, None, np.ones(2), [0.0], [], [])

    with pytest.raises(ValueError, match="non-positive"):
        _model().compute_rhs(0.0, [1.0, 0.0], simple_pa, None, np.ones(2), [0.0], [], [])


def test_compute_rhs_rejects_missing_or_unbuilt_trees():
    with pytest.raises(AttributeError, match="missing LPA or RPA"):
        _model().compute_rhs(0.0, [1.0, 0.1], FakePA(), None, np.ones(2), [0.0], [], [])

    with pytest.raises(AttributeError, match="LPA structured tree has not been built"):
        _model().compute_rhs(
            0.0,
            [1.0, 0.1, 1.0, 0.1],
            FakePA(SimpleNamespace(store=None), FakeTree([2.0], [20])),
            None,
            np.ones(4),
            [0.0],
            [],
            [],
        )


def test_compute_rhs_rejects_wrong_state_length():
    simple_pa = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    with pytest.raises(ValueError, match="State vector length"):
        _model().compute_rhs(0.0, [1.0, 0.1], simple_pa, None, np.ones(2), [0.0], [], [])


def test_compute_rhs_happy_path_uses_wss_only_and_freezes_thickness(monkeypatch):
    lpa = FakeTree([2.0, 4.0], [10, 11])
    rpa = FakeTree([6.0], [20])
    lpa.homeostatic_wss = np.array([1.0, 1.5])
    rpa.homeostatic_wss = np.array([2.0])
    simple_pa = FakePA(lpa, rpa)
    calls = []

    def fake_estimate(tree, *, root_flow, distal_pressure):
        calls.append((tree, root_flow, distal_pressure))
        if tree is lpa:
            return SimpleNamespace(wall_shear_stress=np.array([3.0, 7.0]))
        return SimpleNamespace(wall_shear_stress=np.array([11.0]))

    monkeypatch.setattr(cwss_module, "estimate_steady_tree_hemodynamics", fake_estimate)

    y = np.array([1.0, 0.2, 2.0, 0.4, 3.0, 0.6])
    dydt = _model().compute_rhs(
        0.0,
        y,
        simple_pa,
        None,
        y.copy(),
        [0.0],
        [],
        [],
    )

    tau = np.array([3.0, 7.0, 11.0])
    tau_err = tau - np.array([1.0, 1.5, 2.0])
    expected = np.zeros_like(y)
    expected[0::2] = tau_err * np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(dydt, expected)
    np.testing.assert_allclose(lpa.store.d, [2.0, 4.0])
    np.testing.assert_allclose(rpa.store.d, [6.0])
    assert simple_pa.update_calls == 1
    assert simple_pa.sim_calls == 1
    assert calls == [
        (lpa, 2.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (rpa, 3.0, simple_pa.clinical_targets.wedge_p * 1333.2),
    ]


def test_compute_rhs_uses_steady_hemodynamics_on_every_rhs_evaluation(monkeypatch):
    lpa = FakeTree([2.0], [10])
    rpa = FakeTree([4.0], [20])
    lpa.homeostatic_wss = np.array([1.0])
    rpa.homeostatic_wss = np.array([2.0])
    simple_pa = FakePA(lpa, rpa)
    calls = []

    def fake_estimate(tree, *, root_flow, distal_pressure):
        calls.append((tree, root_flow, distal_pressure))
        return SimpleNamespace(wall_shear_stress=np.array([3.0 if tree is lpa else 5.0]))

    monkeypatch.setattr(cwss_module, "estimate_steady_tree_hemodynamics", fake_estimate)

    y = np.array([1.0, 0.1, 2.0, 0.2])
    _model().compute_rhs(0.0, y, simple_pa, None, y.copy(), [0.0], [], [])
    _model().compute_rhs(0.0, y, simple_pa, None, y.copy(), [0.0], [], [])
    _model().compute_rhs(1.0, y, simple_pa, None, y.copy(), [0.0], [], [])

    assert calls == [
        (lpa, 2.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (rpa, 3.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (lpa, 2.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (rpa, 3.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (lpa, 2.0, simple_pa.clinical_targets.wedge_p * 1333.2),
        (rpa, 3.0, simple_pa.clinical_targets.wedge_p * 1333.2),
    ]


def test_event_uses_rpa_split_rolling_window_convergence():
    model = _model()
    simple_pa = SimpleNamespace(rpa_split=0.5)
    last_y = np.array([1.0, 0.2, 2.0, 0.4])

    assert model.event(
        0.0,
        last_y,
        simple_pa,
        last_y,
        [],
        {"triggered": False, "was_positive": False},
    ) == pytest.approx(model.min_convergence_time)

    assert model.event(
        6.0,
        last_y,
        simple_pa,
        last_y,
        [{"t": 5.5, "rpa_split": 0.5001}],
        {"triggered": False, "was_positive": False},
    ) == pytest.approx(model.min_convergence_time - 6.0)

    stable_window = [
        {"t": 7.4, "rpa_split": 0.50020},
        {"t": 8.5, "rpa_split": 0.50035},
        {"t": 10.0, "rpa_split": 0.50010},
        {"t": 11.0, "rpa_split": 0.50025},
        {"t": 12.0, "rpa_split": 0.50030},
    ]
    simple_pa.rpa_split = 0.50030
    assert model.event(
        12.0,
        last_y,
        simple_pa,
        last_y,
        stable_window,
        {"triggered": False, "was_positive": True},
    ) < 0.0

    simple_pa.rpa_split = 0.5030
    wide_window = [
        {"t": 7.0, "rpa_split": 0.5000},
        {"t": 9.0, "rpa_split": 0.5015},
        {"t": 10.5, "rpa_split": 0.5024},
        {"t": 11.5, "rpa_split": 0.5010},
        {"t": 12.0, "rpa_split": 0.5030},
    ]
    assert model.event(
        12.0,
        last_y,
        simple_pa,
        last_y,
        wide_window,
        {"triggered": False, "was_positive": True},
    ) > 0.0

    assert model.event_outsidesim(0.0, last_y, simple_pa, last_y) > 0.0
    assert model.event_outsidesim(1.0, last_y, simple_pa, last_y) < 0.0
    assert model.event_outsidesim(1.0, last_y * 2.0, simple_pa, last_y) > 0.0

    diag = model.convergence_diagnostics(12.0, stable_window)
    assert diag["window_coverage"] == pytest.approx(4.6)
    assert diag["window_samples"] == 5
    assert diag["window_span"] < model.split_window_tolerance
    assert diag["center_deviation"] < model.split_window_center_tolerance
    assert diag["converged"] is True


def test_convergence_diagnostics_requires_enough_samples_and_fractional_coverage():
    model = _model()

    sparse_window = [
        {"t": 7.0, "rpa_split": 0.50010},
        {"t": 9.5, "rpa_split": 0.50020},
        {"t": 12.0, "rpa_split": 0.50015},
    ]
    sparse_diag = model.convergence_diagnostics(12.0, sparse_window)
    assert sparse_diag["window_samples"] == 3
    assert sparse_diag["converged"] is False
    assert sparse_diag["margin"] > 0.0

    short_coverage_window = [
        {"t": 7.8, "rpa_split": 0.50010},
        {"t": 8.7, "rpa_split": 0.50020},
        {"t": 9.6, "rpa_split": 0.50005},
        {"t": 10.5, "rpa_split": 0.50015},
        {"t": 11.4, "rpa_split": 0.50012},
        {"t": 12.0, "rpa_split": 0.50018},
    ]
    short_diag = model.convergence_diagnostics(12.0, short_coverage_window)
    assert short_diag["window_samples"] == 6
    assert short_diag["window_coverage"] == pytest.approx(4.2)
    assert short_diag["converged"] is False
    assert short_diag["margin"] > 0.0


def test_event_terminates_solve_ivp_once_split_window_is_stable():
    model = _model()
    y0 = np.array([1.0, 0.2], dtype=np.float64)
    flow_log = []
    simple_pa = SimpleNamespace()
    event = wrap_event(
        model.event,
        simple_pa,
        y0.copy(),
        flow_log,
        {"triggered": False, "was_positive": False},
    )

    def rhs(t, y):
        flow_log.append(
            {
                "t": float(t),
                "rpa_split": float(0.18405 + 2.0e-4 * np.sin(2.0 * np.pi * t)),
            }
        )
        return np.zeros_like(y)

    sol = solve_ivp(
        rhs,
        (0.0, 20.0),
        y0,
        events=event,
        max_step=0.25,
        rtol=1e-9,
        atol=1e-12,
    )

    assert sol.status == 1
    assert len(sol.t_events[0]) == 1
    assert sol.t_events[0][0] >= model.min_convergence_time
    assert sol.t_events[0][0] < 20.0


def test_run_adaptation_stops_on_accepted_step_convergence():
    class AcceptedStepModel:
        event_reason_label = "accepted_step_window_converged"

        def __init__(self, K_arr):
            self.K_arr = K_arr

        def compute_rhs(self, t, y, simple_pa, _vessels, last_update_y, last_t_holder, flow_log, solver_trace):
            if t > max(last_t_holder[0], 1e-12):
                last_update_y[:] = y
                last_t_holder[0] = float(t)
                flow_log.append({"t": float(t), "rpa_split": float(simple_pa.rpa_split)})
                solver_trace.append(
                    {
                        "t": float(t),
                        "geom_change_mean": 0.0,
                        "rpa_split": float(simple_pa.rpa_split),
                        "rhs_l2": 0.0,
                    }
                )
            return np.zeros_like(y)

        def convergence_margin(self, t, flow_log):
            return -1.0 if t >= 3.0 else 1.0

    preop = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))
    postop = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    result, flow_log, sol, adapted_pa, hists = run_adaptation(
        preop,
        postop,
        AcceptedStepModel,
        [1.0, 0.0, 0.0, 0.0],
        t_end=10.0,
        max_step=0.5,
        method="RK23",
    )

    assert result["stable"] == 1
    assert result["solver_diagnostics"]["termination_reason"] == "accepted_step_window_converged"
    assert result["solver_diagnostics"]["event_time"] >= 3.0
    assert result["solver_diagnostics"]["convergence_check_mode"] == "accepted_step_window"
    assert result["solver_diagnostics"]["effective_max_step"] == pytest.approx(0.5)
    assert result["solver_diagnostics"]["accepted_step_convergence_history"][-1]["converged"] is True
    assert sol.status == 1
    assert len(flow_log) >= 2
    assert flow_log[-1]["t"] >= 3.0
    assert adapted_pa.sim_calls > 0
    assert len(hists) == 1


def test_run_adaptation_decodes_log_space_model_outputs():
    class LogSpaceAcceptedStepModel:
        event_reason_label = "accepted_step_window_converged"

        def __init__(self, K_arr):
            self.K_arr = K_arr

        def encode_state(self, y):
            return np.log(np.asarray(y, dtype=np.float64))

        def decode_state(self, y):
            return np.exp(np.asarray(y, dtype=np.float64))

        def compute_rhs(self, t, y, simple_pa, _vessels, last_update_y, last_t_holder, flow_log, solver_trace):
            if t > max(last_t_holder[0], 1e-12):
                last_update_y[:] = y
                last_t_holder[0] = float(t)
                flow_log.append({"t": float(t), "rpa_split": float(simple_pa.rpa_split)})
                solver_trace.append(
                    {
                        "t": float(t),
                        "geom_change_mean": 0.0,
                        "rpa_split": float(simple_pa.rpa_split),
                        "rhs_l2": 0.0,
                    }
                )
            return np.zeros_like(y)

        def convergence_margin(self, t, flow_log):
            return -1.0 if t >= 3.0 else 1.0

    preop = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))
    postop = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    _result, _flow_log, sol, adapted_pa, _hists = run_adaptation(
        preop,
        postop,
        LogSpaceAcceptedStepModel,
        [1.0, 0.0, 0.0, 0.0],
        t_end=10.0,
        max_step=0.5,
        method="RK23",
    )

    np.testing.assert_allclose(sol.y[:, 0], np.array([1.0, 0.1, 1.5, 0.15]))
    np.testing.assert_allclose(adapted_pa.lpa_tree.store.d, np.array([2.0]))
    np.testing.assert_allclose(adapted_pa.rpa_tree.store.d, np.array([3.0]))


def test_run_minipa_cwss_debug_converges_with_large_terminal_diameter():
    debug = run_minipa_cwss_debug(
        initial_d=0.3,
        d_min=0.1,
        q_total_preop=2.0,
        q_total_postop=2.0,
        wss_gain=0.01,
        t_end=200.0,
        max_step=1.0,
        method="RK23",
    )

    metrics = debug["metrics"]
    diagnostics = metrics["solver_diagnostics"]
    assert metrics["stable"] == 1
    assert diagnostics["termination_reason"] == "rpa_split_window_converged"
    assert diagnostics["event_time"] < 200.0
    assert diagnostics["accepted_step_convergence_history"][-1]["converged"] is True
    assert diagnostics["accepted_step_convergence_history"][-1]["window_span"] < 1e-3


def test_run_structured_tree_adaptation_m1_uses_stable_dispatch_and_exports_solver_metrics(
    monkeypatch,
    tmp_path,
):
    adapted_dir = tmp_path / "adapted"
    adapted_dir.mkdir()
    (adapted_dir / "svzerod_3Dcoupling.json").write_text(
        json.dumps({"kind": "adapted"}),
        encoding="utf-8",
    )

    class DummySimDir:
        def __init__(self, path, lpa_flow, rpa_flow, lpa_res, rpa_res):
            self.path = path
            self._lpa_flow = lpa_flow
            self._rpa_flow = rpa_flow
            self._lpa_res = lpa_res
            self._rpa_res = rpa_res

        def flow_split(self, get_mean=True, verbose=False):
            return ({"lpa": self._lpa_flow}, {"rpa": self._rpa_flow})

        def _compute_pressure_drops(self, get_mean=True):
            return (0.0, 0.0, 0.0, self._lpa_res, self._rpa_res)

    def fake_from_directory(path, convert_to_cm=False):
        path = str(path)
        if path.endswith("preop"):
            return DummySimDir(path, 10.0, 20.0, 100.0, 200.0)
        if path.endswith("postop"):
            return DummySimDir(path, 12.0, 18.0, 110.0, 190.0)
        return SimpleNamespace(path=path, svzerod_3Dcoupling=None)

    class DummyAdaptor:
        call_kwargs = None

        def __init__(self, preop, postop, adapted, targets, **kwargs):
            self.adapted_simdir = adapted

        def adapt_cwss(self, **kwargs):
            DummyAdaptor.call_kwargs = kwargs
            return {
                "stable": 1,
                "geom_err": 0.02,
                "t95": 12.5,
                "n_rhs": 34,
                "wss_gain": kwargs["wss_gain"],
                "solver_t_end": 7200.0,
                "solver_rtol": kwargs["rtol"],
                "solver_atol": kwargs["atol"],
                "solver_max_step": kwargs["max_step"],
                "tree_max_nodes": kwargs["max_nodes"],
                "flow_log_points": 5,
                "saved_history_figures": 1,
                "preop_rpa_split": 0.67,
                "postop_rpa_split": 0.56,
                "final_rpa_split": 0.54,
                "solver_diagnostics": {
                    "termination_reason": "geometry_converged",
                    "event_time": 240.0,
                    "rhs_l2_initial": 0.3,
                    "rhs_l2_final": 1e-5,
                    "accepted_step_flow_split_history": [
                        {"t": 0.0, "rpa_split": 0.56},
                        {"t": 120.0, "rpa_split": 0.55},
                        {"t": 240.0, "rpa_split": 0.54},
                    ],
                },
            }

    monkeypatch.setattr(workflow_module.SimulationDirectory, "from_directory", fake_from_directory)
    monkeypatch.setattr(
        workflow_module.ClinicalTargets,
        "from_csv",
        lambda _path: SimpleNamespace(rpa_split=0.58),
    )
    monkeypatch.setattr(workflow_module, "MicrovascularAdaptor", DummyAdaptor)

    summary = run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=str(tmp_path / "clinical_targets.csv"),
        reduced_order_pa=str(tmp_path / "reduced.json"),
        tree_params=str(tmp_path / "optimized_params.csv"),
        model="M1",
        parameter_set={
            "iterations": 2,
            "max_nodes": 200_000,
            "terminal_resistance": 50_000.0,
        },
        output_root=str(tmp_path / "results"),
    )

    assert DummyAdaptor.call_kwargs["n_iter"] == 2
    assert DummyAdaptor.call_kwargs["max_nodes"] == 200_000
    assert DummyAdaptor.call_kwargs["wss_gain"] == pytest.approx(0.01)
    assert DummyAdaptor.call_kwargs["terminal_resistance"] == pytest.approx(50_000.0)
    assert summary["solver_metrics"]["stable"] == 1
    assert summary["solver_metrics"]["solver_t_end"] == pytest.approx(7200.0)
    assert summary["solver_metrics"]["tree_max_nodes"] == 200_000
    assert summary["hemodynamics"]["threed"]["preop"]["rpa_split"] == pytest.approx(20.0 / 30.0)
    assert summary["hemodynamics"]["internal_zerod"]["adapted_final"]["rpa_split"] == pytest.approx(0.54)
    assert summary["hemodynamics"]["internal_zerod"]["target"]["rpa_split"] == pytest.approx(0.58)
    assert Path(summary["artifacts"]["adapted_coupler_json"]).exists()
    csv_path = Path(summary["artifacts"]["flow_split_convergence_csv"])
    png_path = Path(summary["artifacts"]["flow_split_convergence_png"])
    assert csv_path.exists()
    assert png_path.exists()

    summary_payload = json.loads(
        Path(summary["artifacts"]["adaptation_summary_json"]).read_text(encoding="utf-8")
    )
    metrics_payload = json.loads(
        Path(summary["artifacts"]["adaptation_metrics_json"]).read_text(encoding="utf-8")
    )
    assert summary_payload["artifacts"]["flow_split_convergence_csv"] == str(csv_path)
    assert summary_payload["artifacts"]["flow_split_convergence_png"] == str(png_path)
    assert summary_payload["hemodynamics"]["internal_zerod"]["target"]["rpa_split"] == pytest.approx(0.58)
    assert metrics_payload["artifacts"]["flow_split_convergence_csv"] == str(csv_path)
    assert metrics_payload["artifacts"]["flow_split_convergence_png"] == str(png_path)
    assert metrics_payload["hemodynamics"]["internal_zerod"]["target"]["rpa_split"] == pytest.approx(0.58)


def test_run_structured_tree_adaptation_m2_omits_flow_split_convergence_artifacts(
    monkeypatch,
    tmp_path,
):
    adapted_dir = tmp_path / "adapted"
    adapted_dir.mkdir()
    (adapted_dir / "svzerod_3Dcoupling.json").write_text(
        json.dumps({"kind": "adapted"}),
        encoding="utf-8",
    )

    class DummySimDir:
        def __init__(self, path, lpa_flow, rpa_flow, lpa_res, rpa_res):
            self.path = path
            self._lpa_flow = lpa_flow
            self._rpa_flow = rpa_flow
            self._lpa_res = lpa_res
            self._rpa_res = rpa_res

        def flow_split(self, get_mean=True, verbose=False):
            return ({"lpa": self._lpa_flow}, {"rpa": self._rpa_flow})

        def _compute_pressure_drops(self, get_mean=True):
            return (0.0, 0.0, 0.0, self._lpa_res, self._rpa_res)

    def fake_from_directory(path, convert_to_cm=False):
        path = str(path)
        if path.endswith("preop"):
            return DummySimDir(path, 10.0, 20.0, 100.0, 200.0)
        if path.endswith("postop"):
            return DummySimDir(path, 12.0, 18.0, 110.0, 190.0)
        return SimpleNamespace(path=path, svzerod_3Dcoupling=None)

    class DummyAdaptor:
        def __init__(self, preop, postop, adapted, targets, **kwargs):
            self.adapted_simdir = adapted
            self.lpa_tree = None
            self.rpa_tree = None

        def construct_impedance_trees(self, *, max_nodes):
            tree = lambda: SimpleNamespace(store=SimpleNamespace(d=np.asarray([1.0], dtype=float)))
            return tree(), tree()

        def _finalize_coupling_with_adapted_trees(self):
            return None

    monkeypatch.setattr(workflow_module.SimulationDirectory, "from_directory", fake_from_directory)
    monkeypatch.setattr(
        workflow_module.ClinicalTargets,
        "from_csv",
        lambda _path: SimpleNamespace(rpa_split=0.58),
    )
    monkeypatch.setattr(workflow_module, "MicrovascularAdaptor", DummyAdaptor)

    summary = run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=str(tmp_path / "clinical_targets.csv"),
        reduced_order_pa=str(tmp_path / "reduced.json"),
        tree_params=str(tmp_path / "optimized_params.csv"),
        model="M2",
        parameter_set={"iterations": 2},
        output_root=str(tmp_path / "results"),
    )

    assert "internal_zerod" not in summary["hemodynamics"]
    assert "flow_split_convergence_csv" not in summary["artifacts"]
    assert "flow_split_convergence_png" not in summary["artifacts"]

    metrics_payload = json.loads(
        Path(summary["artifacts"]["adaptation_metrics_json"]).read_text(encoding="utf-8")
    )
    assert "artifacts" not in metrics_payload
    assert "internal_zerod" not in metrics_payload["hemodynamics"]


def test_run_structured_tree_adaptation_m3_exports_solver_metrics(
    monkeypatch,
    tmp_path,
):
    adapted_dir = tmp_path / "adapted"
    adapted_dir.mkdir()
    (adapted_dir / "svzerod_3Dcoupling.json").write_text(
        json.dumps({"kind": "adapted"}),
        encoding="utf-8",
    )

    class DummySimDir:
        def __init__(self, path, lpa_flow, rpa_flow, lpa_res, rpa_res):
            self.path = path
            self._lpa_flow = lpa_flow
            self._rpa_flow = rpa_flow
            self._lpa_res = lpa_res
            self._rpa_res = rpa_res

        def flow_split(self, get_mean=True, verbose=False):
            return ({"lpa": self._lpa_flow}, {"rpa": self._rpa_flow})

        def _compute_pressure_drops(self, get_mean=True):
            return (0.0, 0.0, 0.0, self._lpa_res, self._rpa_res)

    def fake_from_directory(path, convert_to_cm=False):
        path = str(path)
        if path.endswith("preop"):
            return DummySimDir(path, 10.0, 20.0, 100.0, 200.0)
        if path.endswith("postop"):
            return DummySimDir(path, 12.0, 18.0, 110.0, 190.0)
        return SimpleNamespace(path=path, svzerod_3Dcoupling=None)

    class DummyAdaptor:
        call_args = None

        def __init__(self, preop, postop, adapted, targets, **kwargs):
            self.adapted_simdir = adapted

        def adapt_cwss_ims(self, k_arr, **kwargs):
            DummyAdaptor.call_args = {"k_arr": k_arr, **kwargs}
            return {
                "stable": 1,
                "geom_err": 0.03,
                "t95": 18.0,
                "n_rhs": 27,
                "k_arr": [float(value) for value in k_arr],
                "preop_rpa_split": 0.67,
                "postop_rpa_split": 0.56,
                "final_rpa_split": 0.51,
                "solver_method": kwargs["method"],
                "solver_diagnostics": {
                    "termination_reason": "geometry_converged",
                    "event_time": 180.0,
                    "accepted_step_flow_split_history": [
                        {"t": 0.0, "rpa_split": 0.56},
                        {"t": 90.0, "rpa_split": 0.53},
                        {"t": 180.0, "rpa_split": 0.51},
                    ],
                },
            }

    monkeypatch.setattr(workflow_module.SimulationDirectory, "from_directory", fake_from_directory)
    monkeypatch.setattr(
        workflow_module.ClinicalTargets,
        "from_csv",
        lambda _path: SimpleNamespace(rpa_split=0.58),
    )
    monkeypatch.setattr(workflow_module, "MicrovascularAdaptor", DummyAdaptor)

    summary = run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=str(tmp_path / "clinical_targets.csv"),
        reduced_order_pa=str(tmp_path / "reduced.json"),
        tree_params=str(tmp_path / "optimized_params.csv"),
        model="M3",
        parameter_set={
            "k_arr": [1.0, 2.0, 3.0, 4.0],
            "t_end": 5400.0,
            "solver_method": "BDF",
        },
        output_root=str(tmp_path / "results"),
    )

    assert DummyAdaptor.call_args["k_arr"] == [1.0, 2.0, 3.0, 4.0]
    assert DummyAdaptor.call_args["method"] == "BDF"
    assert DummyAdaptor.call_args["t_end"] == pytest.approx(5400.0)
    assert summary["solver_metrics"]["stable"] == 1
    assert summary["solver_metrics"]["k_arr"] == [1.0, 2.0, 3.0, 4.0]
    assert summary["solver_metrics"]["solver_diagnostics"]["termination_reason"] == "geometry_converged"
    assert summary["hemodynamics"]["internal_zerod"]["target"]["rpa_split"] == pytest.approx(0.58)
    metrics_payload = json.loads(
        Path(summary["artifacts"]["adaptation_metrics_json"]).read_text(encoding="utf-8")
    )
    assert metrics_payload["solver_metrics"]["k_arr"] == [1.0, 2.0, 3.0, 4.0]
    assert metrics_payload["hemodynamics"]["internal_zerod"]["adapted_final"]["rpa_split"] == pytest.approx(0.51)


def test_run_single_tree_cwss_debug_smoke():
    class DebugStore(FakeStore):
        def __init__(self, diameters, ids):
            super().__init__(diameters, ids)
            self.left = np.asarray([-1, -1], dtype=np.int32)
            self.right = np.asarray([-1, -1], dtype=np.int32)
            self.gen = np.asarray([0, 0], dtype=np.int32)
            self.lrr = 10.0
            self.eta = 0.04

    class DebugTree:
        def __init__(self):
            self.store = DebugStore([2.0, 3.0], [1, 2])
            self.Pd = 0.0
            self.homeostatic_wss = None

        def compute_homeostatic_state(self, q_homeostatic):
            radii = np.maximum(0.5 * self.store.d, 1e-6)
            q_val = float(q_homeostatic)
            self.homeostatic_wss = 4.0 * self.store.eta * q_val / (np.pi * radii ** 3)

    debug = run_single_tree_cwss_debug(
        DebugTree(),
        q_homeostatic=1.0,
        q_target=1.1,
        wss_gain=0.01,
        t_end=10.0,
        max_step=1.0,
    )

    assert debug["solution"].y.shape[0] == 4
    assert "geom_err" in debug["metrics"]
    assert "n_rhs" in debug["metrics"]
    assert debug["metrics"]["integration_method"] == "RK23"


def test_run_single_tree_cwss_debug_changes_geometry_when_target_flow_changes():
    class DebugStore(FakeStore):
        def __init__(self, diameters, ids):
            super().__init__(diameters, ids)
            self.left = np.asarray([-1, -1], dtype=np.int32)
            self.right = np.asarray([-1, -1], dtype=np.int32)
            self.gen = np.asarray([0, 0], dtype=np.int32)
            self.lrr = 10.0
            self.eta = 0.04

    class DebugTree:
        def __init__(self):
            self.store = DebugStore([2.0, 3.0], [1, 2])
            self.Pd = 0.0
            self.homeostatic_wss = None

        def compute_homeostatic_state(self, q_homeostatic):
            radii = np.maximum(0.5 * self.store.d, 1e-6)
            q_val = float(q_homeostatic)
            self.homeostatic_wss = 4.0 * self.store.eta * q_val / (np.pi * radii ** 3)

    debug = run_single_tree_cwss_debug(
        DebugTree(),
        q_homeostatic=1.0,
        q_target=1.1,
        wss_gain=0.01,
        t_end=10.0,
        max_step=1.0,
    )

    assert debug["metrics"]["geom_err"] > 0.0
