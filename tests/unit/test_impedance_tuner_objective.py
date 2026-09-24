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
        tune_space=TuneSpace(
            free=[
                FreeParam("comp.lpa.C", init=1.0, lb=0.0, ub=10.0),
                FreeParam("comp.rpa.C", init=1.0, lb=0.0, ub=10.0),
            ],
            fixed=[],
            tied=[],
        ),
        compliance_model="constant",
        grid_search_init=False,
        rescale_inflow=False,
        log_file=str(tmp_path / "tuning.log"),
    )


def test_objective_terms_match_documented_formula(tmp_path):
    tuner = _tuner(tmp_path)

    terms = tuner.objective_terms([33.0, 6.0, 20.0], 0.6, {"comp.lpa.C": 1.0, "comp.rpa.C": 2.0})

    expected = {
        "sys": 1.5 * (3.0 / 30.0) ** 2 * 100.0,
        "dia": 1.0 * (6.0 / 12.0) ** 2 * 100.0,
        "mean": 0.0,
        "flow": (0.1 / 0.5) ** 2 * 100.0,
    }
    assert terms["components"] == pytest.approx(expected)
    assert terms["unweighted_loss"] == pytest.approx(sum(expected.values()))
    assert terms["weighted_loss"] == pytest.approx(terms["unweighted_loss"])


def test_evaluate_candidate_reports_metrics_at_unit_weights(tmp_path):
    tuner = _tuner(tmp_path)
    tuner._loss_weights = {"sys": 9.0, "dia": 9.0, "mean": 9.0, "flow": 9.0}
    tuner._evaluate_model = lambda x, provided_model=None: (
        None,
        {"P_mpa": [33.0, 6.0, 20.0], "rpa_split": 0.6},
        None,
        None,
        {"comp.lpa.C": float(x[0]), "comp.rpa.C": float(x[1])},
    )

    result = tuner.evaluate_candidate([1.0, 2.0], pa_config=None)

    expected = tuner.objective_terms([33.0, 6.0, 20.0], 0.6, {"comp.lpa.C": 1.0, "comp.rpa.C": 2.0})
    # Scored at unit weights, and the caller's weights are restored.
    assert tuner._loss_weights["sys"] == 9.0
    assert result["error"] is None
    assert result["loss"] == pytest.approx(expected["unweighted_loss"])
    assert result["metrics"] == {"sys_pressure": 33.0, "dia_pressure": 6.0, "mean_pressure": 20.0, "rpa_split": 0.6}
    assert result["params"] == {"comp.lpa.C": 1.0, "comp.rpa.C": 2.0}


def test_evaluate_candidate_reports_failed_simulation(tmp_path):
    tuner = _tuner(tmp_path)

    def _fail(*_args, **_kwargs):
        raise ValueError("unstable impedance kernel")

    tuner._evaluate_model = _fail

    result = tuner.evaluate_candidate(np.array([1.0, 2.0]), pa_config=None)

    assert result["loss"] == float("inf")
    assert "unstable impedance kernel" in result["error"]
