from types import SimpleNamespace

import numpy as np
import pytest

from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tune_bcs.tune_space import FreeParam, TuneSpace


def _tuner(tmp_path):
    return ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path=str(tmp_path / "mesh-surfaces"),
        clinical_targets=SimpleNamespace(mpa_p=[30.0, 12.0, 20.0], rpa_split=0.5, wedge_p=8.0),
        tune_space=TuneSpace(free=[FreeParam("lpa.alpha", init=0.9, lb=0.7, ub=0.99)], fixed=[], tied=[]),
        compliance_model="constant",
        grid_search_init=False,
        rescale_inflow=False,
        log_file=str(tmp_path / "tuning.log"),
        maxiter=5,
    )


def _raise(exc):
    def _evaluate(*_args, **_kwargs):
        raise exc
    return _evaluate


def test_loss_fn_aborts_on_solver_capability_error(tmp_path):
    tuner = _tuner(tmp_path)
    tuner._evaluate_model = _raise(RuntimeError("Invalid block type IMPEDANCE"))

    with pytest.raises(RuntimeError, match="IMPEDANCE boundary-condition support"):
        tuner.loss_fn(np.array([0.9]), pa_config=None)


def test_loss_fn_penalizes_candidate_specific_failure(tmp_path):
    tuner = _tuner(tmp_path)
    tuner._evaluate_model = _raise(ValueError("impedance BC contains non-finite values"))

    assert tuner.loss_fn(np.array([0.9]), pa_config=None) == 1e9
    assert "non-finite" in tuner._last_evaluation_error


def test_tune_raises_when_no_evaluation_succeeds(tmp_path):
    tuner = _tuner(tmp_path)
    stale_csv = tmp_path / "optimized_params.csv"
    stale_csv.write_text("pa\nlpa\nrpa\n", encoding="utf-8")
    tuner._prepare_geometry_defaults = lambda: None
    tuner._make_tuning_model = lambda: SimpleNamespace(bcs={"INFLOW": SimpleNamespace(Q=[6.0, 6.0])})
    tuner._evaluate_model = _raise(ValueError("unstable impedance kernel"))

    with pytest.raises(RuntimeError, match="no optimizer evaluation simulated successfully"):
        tuner.tune(nm_iter=1)

    assert "unstable impedance kernel" in (tmp_path / "tuning.log").read_text(encoding="utf-8")
    assert stale_csv.read_text(encoding="utf-8") == "pa\nlpa\nrpa\n"


def test_full_pa_tune_checks_inflow_before_any_evaluation(tmp_path):
    tuner = _tuner(tmp_path)
    tuner.tuning_model = "full_pa"
    evaluations = []
    tuner._prepare_geometry_defaults = lambda: None
    # The serialized config disagrees with the in-memory BC, as a stale
    # cached Inflow would; every candidate would fail the snapshot check.
    tuner._make_tuning_model = lambda: SimpleNamespace(
        bcs={"INFLOW": SimpleNamespace(Q=[6.0, 6.0])},
        config={"boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [5.0, 5.0], "t": [0.0, 1.0]}}
        ]},
    )
    tuner._evaluate_model = lambda *args, **kwargs: evaluations.append(args)

    with pytest.raises(ValueError, match="cardiac output mismatch"):
        tuner.tune(nm_iter=1)

    assert evaluations == []
