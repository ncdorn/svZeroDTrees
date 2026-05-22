from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import svzerodtrees.adaptation.models.cwss as cwss_module
import svzerodtrees.adaptation.workflow as workflow_module
from svzerodtrees.adaptation.experiment import run_single_tree_cwss_debug
from svzerodtrees.adaptation.models.cwss import CWSSAdaptation
from svzerodtrees.adaptation.workflow import run_structured_tree_adaptation


class FakeStore:
    def __init__(self, diameters, ids):
        self.d = np.asarray(diameters, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=np.int32)

    def n_nodes(self):
        return self.d.size


class FakeResults:
    def __init__(self, wss):
        self._wss = np.asarray(wss, dtype=np.float64)

    def wss_timeseries(self):
        return self._wss


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

    def fake_simulate_outlet_trees(pa):
        pa.lpa_tree.results = FakeResults(
            wss=[[2.0, 4.0], [6.0, 8.0]],
        )
        pa.rpa_tree.results = FakeResults(
            wss=[[10.0, 12.0]],
        )

    monkeypatch.setattr(cwss_module, "simulate_outlet_trees", fake_simulate_outlet_trees)

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


def test_event_and_event_outsidesim_convergence_behavior():
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
    ) == pytest.approx(1.0)
    assert model.event(
        1.0,
        last_y * 1.1,
        simple_pa,
        last_y,
        [],
        {"triggered": False, "was_positive": False},
    ) == pytest.approx(1.0)
    assert model.event(
        1.0,
        last_y,
        simple_pa,
        last_y,
        [],
        {"triggered": False, "was_positive": True},
    ) == pytest.approx(-1.0)

    assert model.event_outsidesim(0.0, last_y, simple_pa, last_y) > 0.0
    assert model.event_outsidesim(1.0, last_y, simple_pa, last_y) < 0.0
    assert model.event_outsidesim(1.0, last_y * 2.0, simple_pa, last_y) > 0.0


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
                },
            }

    monkeypatch.setattr(workflow_module.SimulationDirectory, "from_directory", fake_from_directory)
    monkeypatch.setattr(workflow_module.ClinicalTargets, "from_csv", lambda _path: object())
    monkeypatch.setattr(workflow_module, "MicrovascularAdaptor", DummyAdaptor)

    summary = run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=str(tmp_path / "clinical_targets.csv"),
        reduced_order_pa=str(tmp_path / "reduced.json"),
        tree_params=str(tmp_path / "optimized_params.csv"),
        model="M1",
        parameter_set={"iterations": 2},
        output_root=str(tmp_path / "results"),
    )

    assert DummyAdaptor.call_kwargs["n_iter"] == 2
    assert DummyAdaptor.call_kwargs["wss_gain"] == pytest.approx(0.01)
    assert summary["solver_metrics"]["stable"] == 1
    assert summary["solver_metrics"]["solver_t_end"] == pytest.approx(7200.0)
    assert summary["hemodynamics"]["threed"]["preop"]["rpa_split"] == pytest.approx(20.0 / 30.0)
    assert summary["hemodynamics"]["internal_zerod"]["adapted_final"]["rpa_split"] == pytest.approx(0.54)
    assert Path(summary["artifacts"]["adapted_coupler_json"]).exists()


def test_run_single_tree_cwss_debug_smoke():
    class DebugTree:
        def __init__(self):
            self.store = FakeStore([2.0, 3.0], [1, 2])
            self.Pd = 0.0
            self.results = None
            self.homeostatic_wss = None

        def compute_homeostatic_state(self, q_homeostatic):
            self.simulate(Q_in=[q_homeostatic, q_homeostatic], Pd=self.Pd)
            self.homeostatic_wss = np.mean(self.results.wss_timeseries(), axis=1)

        def simulate(self, Q_in, Pd):
            q_val = float(Q_in[0])
            radii = np.maximum(0.5 * self.store.d, 1e-6)
            tau = (q_val / radii)[:, None]
            self.results = FakeResults(np.repeat(tau, 2, axis=1))

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
    assert debug["tree"].results is not None
