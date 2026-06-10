from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import svzerodtrees.adaptation.models.cwss_ims as cwss_module
from svzerodtrees.adaptation.models.cwss_ims import CWSSIMSAdaptation


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
    return CWSSIMSAdaptation([1.0, 2.0, 3.0, 4.0])


def test_constructor_validates_gain_array_type_and_length():
    with pytest.raises(TypeError, match="K_arr"):
        CWSSIMSAdaptation("bad")

    with pytest.raises(ValueError, match="exactly 4"):
        CWSSIMSAdaptation([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="K_tau_r"):
        CWSSIMSAdaptation([-1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="K_sig_h"):
        CWSSIMSAdaptation([1.0, 2.0, 3.0, -1.0])

    assert _model().K_arr == [1.0, 2.0, 3.0, 4.0]
    assert CWSSIMSAdaptation([0.0, 2.0, 3.0, 4.0]).K_arr == [0.0, 2.0, 3.0, 4.0]
    assert CWSSIMSAdaptation([1.0, 0.0, 3.0, 4.0]).K_arr == [1.0, 0.0, 3.0, 4.0]
    assert CWSSIMSAdaptation([1.0, 2.0, 3.0, 0.0]).K_arr == [1.0, 2.0, 3.0, 0.0]


def test_compute_rhs_rejects_invalid_state_shape_and_values():
    simple_pa = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    with pytest.raises(ValueError, match="1-D"):
        _model().compute_rhs(0.0, [[1.0, 2.0]], simple_pa, None, np.ones(2), [0.0], [], [])

    with pytest.raises(ValueError, match="finite"):
        _model().compute_rhs(0.0, [0.0, np.inf], simple_pa, None, np.ones(2), [0.0], [], [])


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


def test_compute_rhs_happy_path_uses_results_and_homeostatic_references(monkeypatch):
    lpa = FakeTree([2.0, 4.0], [10, 11])
    rpa = FakeTree([6.0], [20])
    lpa.homeostatic_wss = np.array([1.0, 1.5])
    rpa.homeostatic_wss = np.array([2.0])
    lpa.homeostatic_ims = np.array([80.0, 120.0])
    rpa.homeostatic_ims = np.array([150.0])
    simple_pa = FakePA(lpa, rpa)

    def fake_estimate(tree, *, root_flow, distal_pressure):
        if tree is lpa:
            return SimpleNamespace(
                wall_shear_stress=np.array([3.0, 7.0]),
                pressure_in=np.array([12.0, 22.0]),
            )
        return SimpleNamespace(
            wall_shear_stress=np.array([11.0]),
            pressure_in=np.array([32.0]),
        )

    monkeypatch.setattr(cwss_module, "estimate_steady_tree_hemodynamics", fake_estimate)

    y_physical = np.array([1.0, 0.2, 2.0, 0.4, 3.0, 0.6])
    y = _model().encode_state(y_physical)
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
    sig = np.array([60.0, 110.0, 160.0])
    r = np.array([1.0, 2.0, 3.0])
    h = np.array([0.2, 0.4, 0.6])
    tau_err = np.log(tau / np.array([1.0, 1.5, 2.0]))
    sig_err = np.log(sig / np.array([80.0, 120.0, 150.0]))
    expected = np.empty_like(y)
    expected[0::2] = tau_err + 2.0 * sig_err
    expected[1::2] = 3.0 * tau_err + 4.0 * sig_err

    np.testing.assert_allclose(dydt, expected)
    np.testing.assert_allclose(lpa.store.d, [2.0, 4.0])
    np.testing.assert_allclose(rpa.store.d, [6.0])
    assert simple_pa.update_calls == 1
    assert simple_pa.sim_calls == 1


def test_compute_rhs_uses_homeostatic_maps_and_validates_reference_sizes(monkeypatch):
    lpa = FakeTree([2.0], [10])
    rpa = FakeTree([4.0], [20])
    lpa._homeostatic_wss_map = {10: 1.0}
    rpa._homeostatic_wss_map = {20: 1.0}
    lpa._homeostatic_ims_map = {10: 50.0}
    rpa._homeostatic_ims_map = {20: 50.0}

    monkeypatch.setattr(
        cwss_module,
        "estimate_steady_tree_hemodynamics",
        lambda tree, **_: SimpleNamespace(
            wall_shear_stress=np.array([2.0 if tree is lpa else 3.0]),
            pressure_in=np.array([10.0 if tree is lpa else 20.0]),
        ),
    )
    dydt = _model().compute_rhs(
        0.0,
        _model().encode_state([1.0, 0.2, 2.0, 0.4]),
        FakePA(lpa, rpa),
        None,
        np.ones(4),
        [0.0],
        [],
        [],
    )
    assert dydt.shape == (4,)

    lpa.homeostatic_wss = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="homeostatic_wss"):
        _model().compute_rhs(
            0.0,
            [1.0, 0.2, 2.0, 0.4],
            FakePA(lpa, rpa),
            None,
            np.ones(4),
            [0.0],
            [],
            [],
        )


def test_log_stimulus_uses_symmetric_log_ratio():
    model = _model()
    stimulus = model._log_stimulus(np.array([12.0, 10.0]), np.array([10.0, 12.0]))
    np.testing.assert_allclose(stimulus, np.array([np.log(1.2), -np.log(1.2)]))


def test_encode_decode_state_round_trips_physical_geometry():
    model = _model()
    physical = np.array([1.0, 0.2, 2.0, 0.4])
    encoded = model.encode_state(physical)
    decoded = model.decode_state(encoded)
    np.testing.assert_allclose(decoded, physical)


def test_convergence_diagnostics_matches_moving_window_split_logic():
    model = _model()
    stable_window = [
        {"t": 40.0, "rpa_split": 0.50000},
        {"t": 46.0, "rpa_split": 0.500006},
        {"t": 52.0, "rpa_split": 0.500010},
        {"t": 58.0, "rpa_split": 0.500004},
        {"t": 64.0, "rpa_split": 0.500008},
        {"t": 70.0, "rpa_split": 0.500002},
    ]
    diag = model.convergence_diagnostics(70.0, stable_window)
    assert diag["window_samples"] == 6
    assert diag["window_coverage"] == pytest.approx(30.0)
    assert diag["window_span"] < model.split_window_tolerance
    assert diag["center_deviation"] < model.split_window_center_tolerance
    assert diag["converged"] is True
    assert model.convergence_margin(70.0, stable_window) <= 0.0

    sparse_window = [
        {"t": 40.0, "rpa_split": 0.5001},
        {"t": 70.0, "rpa_split": 0.5000},
    ]
    sparse_diag = model.convergence_diagnostics(70.0, sparse_window)
    assert sparse_diag["converged"] is False
    assert sparse_diag["margin"] > 0.0

    early_diag = model.convergence_diagnostics(50.0, stable_window)
    assert early_diag["converged"] is False
    assert early_diag["margin"] > 0.0


def test_event_and_event_outsidesim_convergence_behavior():
    model = _model()
    simple_pa = SimpleNamespace(rpa_split=0.5)
    last_y = model.encode_state(np.array([1.0, 0.2, 2.0, 0.4]))
    flow_log = [
        {"t": 40.0, "rpa_split": 0.50000},
        {"t": 46.0, "rpa_split": 0.500006},
        {"t": 52.0, "rpa_split": 0.500010},
        {"t": 58.0, "rpa_split": 0.500004},
        {"t": 64.0, "rpa_split": 0.500008},
        {"t": 70.0, "rpa_split": 0.500002},
    ]

    assert model.event(
        70.0,
        last_y,
        simple_pa,
        last_y,
        flow_log,
        {"triggered": False, "was_positive": False},
    ) <= 0.0

    assert model.event_outsidesim(0.0, last_y, simple_pa, last_y) > 0.0
    assert model.event_outsidesim(1.0, last_y, simple_pa, last_y) < 0.0
    assert model.event_outsidesim(1.0, model.encode_state(model.decode_state(last_y) * 2.0), simple_pa, last_y) > 0.0
