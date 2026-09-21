import pytest

from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tune_bcs.tune_space import FixedParam, FreeParam, TiedParam, TuneSpace


def test_olufsen_compliance_uses_optional_k3_parameter():
    tuner = object.__new__(ImpedanceTuner)
    tuner.compliance_model = "olufsen"

    compliance = tuner._build_compliance(
        "lpa",
        {
            "comp.lpa.k2": -25.0,
            "comp.lpa.k3": 100000.0,
        },
    )

    assert compliance.k1 == pytest.approx(19992500.0)
    assert compliance.k2 == pytest.approx(-25.0)
    assert compliance.k3 == pytest.approx(100000.0)


def test_tree_params_apply_tied_olufsen_k3_to_both_pas():
    tuner = object.__new__(ImpedanceTuner)
    tuner.compliance_model = "olufsen"
    tuner._geom_defaults = {
        "lpa.default_diameter": 0.24,
        "rpa.default_diameter": 0.36,
    }
    tune_space = TuneSpace(
        free=[
            FreeParam("comp.lpa.k2", init=-25.0, lb=-35.0, ub=-1.0),
            FreeParam("comp.lpa.k3", init=100000.0, lb=0.0, ub=1000000.0),
        ],
        fixed=[FixedParam("lrr", 10.0), FixedParam("d_min", 0.01)],
        tied=[
            TiedParam("comp.rpa.k2", other="comp.lpa.k2"),
            TiedParam("comp.rpa.k3", other="comp.lpa.k3"),
        ],
    )

    x0, _ = tune_space.pack_init_and_bounds()
    lpa, rpa = tuner._build_tree_params(tune_space.vector_to_param_dict(x0))

    assert lpa.compliance_model.k3 == pytest.approx(100000.0)
    assert rpa.compliance_model.k3 == pytest.approx(100000.0)
