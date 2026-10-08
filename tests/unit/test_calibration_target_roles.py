"""MPA/LPA/RPA target roles for target-focused full-PA calibration."""

from __future__ import annotations

import json

import pytest

from svzerodtrees.calibration.target_roles import full_pa_calibration_targets, pa_target_roles
from svzerodtrees.config import _parse_calibration_targets


def _model():
    # MPA(0) -> J0 -> {LPA branch: 1 -> J1 -> (2, 3)}, {RPA branch: 4 (outlet)}
    def vessel(i, name, bcs=None):
        v = {"vessel_id": i, "vessel_name": name}
        if bcs:
            v["boundary_conditions"] = bcs
        return v

    return {
        "vessels": [
            vessel(0, "branch0_seg0", {"inlet": "INFLOW"}),
            vessel(1, "branch1_seg0_connectorEL"),
            vessel(2, "branch2_seg0_connectorEL", {"outlet": "RESISTANCE_0"}),
            vessel(3, "branch3_seg0_connectorEL", {"outlet": "RESISTANCE_1"}),
            vessel(4, "branch4_seg0_connectorEL", {"outlet": "RESISTANCE_2"}),
        ],
        "junctions": [
            {"junction_name": "J0", "inlet_vessels": [0], "outlet_vessels": [4, 1]},
            {"junction_name": "J1", "inlet_vessels": [1], "outlet_vessels": [2, 3]},
        ],
    }


def _mapping(sides):
    return {"pairs": [{"bc_name": f"RESISTANCE_{i}", "side": s} for i, s in enumerate(sides)]}


def test_roles_follow_outlet_sides_not_vessel_order():
    roles = pa_target_roles(_model(), _mapping(["lpa", "lpa", "rpa"]))
    assert roles == {
        "mpa_vessel": "branch0_seg0",
        "lpa_vessel": "branch1_seg0_connectorEL",
        "rpa_vessel": "branch4_seg0_connectorEL",
    }


@pytest.mark.parametrize(
    ("sides", "message"),
    [
        (["lpa", "rpa", "rpa"], "expected one side"),
        (["rpa", "rpa", "rpa"], "both MPA junction outlets drain the RPA"),
    ],
)
def test_mixed_or_duplicate_sides_are_errors(sides, message):
    with pytest.raises(ValueError, match=message):
        pa_target_roles(_model(), _mapping(sides))


def test_targets_block_parses_as_calibration_targets(tmp_path):
    model = tmp_path / "svzerod_3d_coupling_tuned.json"
    mapping = tmp_path / "outlet_cap_mapping.json"
    model.write_text(json.dumps(_model()), encoding="utf-8")
    mapping.write_text(json.dumps(_mapping(["lpa", "lpa", "rpa"])), encoding="utf-8")
    targets = full_pa_calibration_targets(model, mapping)
    assert targets["rpa_flow_split"]["rpa_vessel"] == "branch4_seg0_connectorEL"
    assert targets["mpa_pressure"]["interface"] == "external_upstream"
    assert _parse_calibration_targets(targets) is not None
