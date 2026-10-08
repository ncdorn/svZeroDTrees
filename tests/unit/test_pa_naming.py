"""LPA/RPA side from cap and surface names."""

from __future__ import annotations

import pytest

from svzerodtrees.pa_naming import pa_side
from svzerodtrees.tune_bcs.outlet_mapping import cap_side


@pytest.mark.parametrize(
    ("name", "side"),
    [
        ("lpa_2.vtp", "lpa"),
        ("/mesh-surfaces/RPA_1.vtp", "rpa"),
        ("l_pa_1_x.vtp", "lpa"),
        ("r_pa_x_2.vtp", "rpa"),
        ("r-pa_3", "rpa"),
        ("segmental_pa.vtp", None),  # 'l' + 'pa' only as a separated token
        ("wall_blend_r_pa_x_l_pa_x.vtp", None),  # both sides
        ("inflow.vtp", None),
    ],
)
def test_pa_side(name, side):
    assert pa_side(name) == side


def test_cap_side_accepts_separated_names_and_rejects_ambiguous_ones():
    assert cap_side("l_pa_4_1_x.vtp") == "lpa"
    with pytest.raises(ValueError, match="exactly one pulmonary side"):
        cap_side("segmental_pa.vtp")
