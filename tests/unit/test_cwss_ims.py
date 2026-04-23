from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import svzerodtrees.adaptation.models.cwss_ims as cwss_module
from svzerodtrees.adaptation.models.cwss_ims import CWSSIMSAdaptation


class FakeStore:
    def __init__(self, diameters, ids):
        self.d = np.asarray(diameters, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=np.int32)

    def n_nodes(self):
        return self.d.size


class FakeResults:
    def __init__(self, wss, pressure):
        self._wss = np.asarray(wss, dtype=np.float64)
        self.pressure_in = np.asarray(pressure, dtype=np.float64)

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
    return CWSSIMSAdaptation([1.0, 2.0, 3.0, 4.0])


def test_constructor_validates_gain_array_type_and_length():
    with pytest.raises(TypeError, match="K_arr"):
        CWSSIMSAdaptation("bad")

    with pytest.raises(ValueError, match="exactly 4"):
        CWSSIMSAdaptation([1.0, 2.0, 3.0])

    assert _model().K_arr == [1.0, 2.0, 3.0, 4.0]


def test_compute_rhs_rejects_invalid_state_shape_and_values():
    simple_pa = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    with pytest.raises(ValueError, match="1-D"):
        _model().compute_rhs(0.0, [[1.0, 2.0]], simple_pa, None, np.ones(2), [0.0], [])

    with pytest.raises(ValueError, match="non-positive"):
        _model().compute_rhs(0.0, [1.0, 0.0], simple_pa, None, np.ones(2), [0.0], [])


def test_compute_rhs_rejects_missing_or_unbuilt_trees():
    with pytest.raises(AttributeError, match="missing LPA or RPA"):
        _model().compute_rhs(0.0, [1.0, 0.1], FakePA(), None, np.ones(2), [0.0], [])

    with pytest.raises(AttributeError, match="LPA structured tree has not been built"):
        _model().compute_rhs(
            0.0,
            [1.0, 0.1, 1.0, 0.1],
            FakePA(SimpleNamespace(store=None), FakeTree([2.0], [20])),
            None,
            np.ones(4),
            [0.0],
            [],
        )


def test_compute_rhs_rejects_wrong_state_length():
    simple_pa = FakePA(FakeTree([2.0], [10]), FakeTree([3.0], [20]))

    with pytest.raises(ValueError, match="State vector length"):
        _model().compute_rhs(0.0, [1.0, 0.1], simple_pa, None, np.ones(2), [0.0], [])


def test_compute_rhs_happy_path_uses_results_and_homeostatic_references(monkeypatch):
    lpa = FakeTree([2.0, 4.0], [10, 11])
    rpa = FakeTree([6.0], [20])
    lpa.homeostatic_wss = np.array([1.0, 1.5])
    rpa.homeostatic_wss = np.array([2.0])
    lpa.homeostatic_ims = np.array([80.0, 120.0])
    rpa.homeostatic_ims = np.array([150.0])
    simple_pa = FakePA(lpa, rpa)

    def fake_simulate_outlet_trees(pa):
        pa.lpa_tree.results = FakeResults(
            wss=[[2.0, 4.0], [6.0, 8.0]],
            pressure=[[10.0, 14.0], [20.0, 24.0]],
        )
        pa.rpa_tree.results = FakeResults(
            wss=[[10.0, 12.0]],
            pressure=[[30.0, 34.0]],
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
    )

    tau = np.array([3.0, 7.0, 11.0])
    sig = np.array([60.0, 110.0, 160.0])
    tau_err = tau - np.array([1.0, 1.5, 2.0])
    sig_err = sig - np.array([80.0, 120.0, 150.0])
    expected = np.empty_like(y)
    expected[0::2] = tau_err + 2.0 * sig_err
    expected[1::2] = -3.0 * tau_err + 4.0 * sig_err

    np.testing.assert_allclose(dydt, expected)
    np.testing.assert_allclose(lpa.store.d, [2.0, 4.0])
    np.testing.assert_allclose(rpa.store.d, [6.0])
    assert simple_pa.update_calls == 1
    assert simple_pa.sim_calls == 1


def test_compute_rhs_uses_homeostatic_maps_and_validates_reference_sizes(monkeypatch):
    lpa = FakeTree([2.0], [10], results=FakeResults([[2.0, 2.0]], [[10.0, 10.0]]))
    rpa = FakeTree([4.0], [20], results=FakeResults([[3.0, 3.0]], [[20.0, 20.0]]))
    lpa._homeostatic_wss_map = {10: 1.0}
    rpa._homeostatic_wss_map = {20: 1.0}
    lpa._homeostatic_ims_map = {10: 50.0}
    rpa._homeostatic_ims_map = {20: 50.0}

    monkeypatch.setattr(cwss_module, "simulate_outlet_trees", lambda _pa: None)
    dydt = _model().compute_rhs(
        0.0,
        [1.0, 0.2, 2.0, 0.4],
        FakePA(lpa, rpa),
        None,
        np.ones(4),
        [0.0],
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
        )


def test_event_and_event_outsidesim_convergence_behavior():
    model = _model()
    simple_pa = SimpleNamespace(rpa_split=0.5)
    last_y = np.array([1.0, 0.2, 2.0, 0.4])
    flow_log = [(0.0, 0.5), (1.0, 0.50001)]

    assert model.event(
        0.0,
        last_y * 1.1,
        simple_pa,
        last_y,
        flow_log,
        triggered=[False],
        was_positive=[False],
    ) == pytest.approx(1.0)
    assert model.event(
        0.0,
        last_y,
        simple_pa,
        last_y,
        flow_log,
        triggered=[False],
        was_positive=[True],
    ) == pytest.approx(-1.0)

    assert model.event_outsidesim(0.0, last_y, simple_pa, last_y) < 0.0
    assert model.event_outsidesim(0.0, last_y * 2.0, simple_pa, last_y) > 0.0
