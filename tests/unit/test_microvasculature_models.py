from __future__ import annotations

import pandas as pd
import pytest

from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.compliance.olufsen import OlufsenCompliance
from svzerodtrees.microvasculature.structured_tree.asymmetry import (
    alpha_beta_from_xi_eta,
    xi_from_alpha_beta,
)
from svzerodtrees.microvasculature.treeparams import TreeParameters


def test_compliance_models_evaluate_radius():
    assert ConstantCompliance(123.0).evaluate(0.5) == pytest.approx(123.0)

    olufsen = OlufsenCompliance(k1=1.0, k2=0.0, k3=2.0)
    assert olufsen.evaluate(0.25) == pytest.approx(3.0)


def test_asymmetry_parameter_round_trip():
    xi = 2.3
    alpha, beta = alpha_beta_from_xi_eta(xi, 0.65)

    assert xi_from_alpha_beta(alpha, beta) == pytest.approx(xi)


def test_tree_parameters_from_row_and_to_csv_row_include_derived_values():
    xi = 2.1
    alpha, beta = alpha_beta_from_xi_eta(xi, 0.6)
    row = pd.DataFrame(
        [
            {
                "pa": "lpa",
                "lrr": 10.0,
                "diameter": 0.3,
                "d_min": 0.01,
                "alpha": alpha,
                "beta": beta,
                "xi": None,
                "eta_sym": beta / alpha,
                "inductance": 0.02,
                "compliance model": "ConstantCompliance",
                "Eh/r": 66000.0,
            }
        ]
    )

    params = TreeParameters.from_row(row)
    serialized = params.to_csv_row(loss=1.0, flow_split=0.5, p_mpa=[10.0, 5.0, 8.0])

    assert params.xi == pytest.approx(xi)
    assert params.inductance == pytest.approx(0.02)
    assert serialized["xi"] == pytest.approx(xi)
    assert serialized["compliance model"] == "ConstantCompliance"
