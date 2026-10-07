"""Calibration topology for raw learnedZeroD vessel names."""

from __future__ import annotations

import pytest

from svzerodtrees.calibration.workflow import (
    _branch_and_segment_for_vessel,
    _network_topology,
)


def test_learned_connector_names_map_to_branch_segments():
    assert _branch_and_segment_for_vessel("branch3_seg1") == (3, 1)
    assert _branch_and_segment_for_vessel("branch1_seg0_connectorEL") == (1, 0)
    assert _branch_and_segment_for_vessel("branch2_seg0_1_2_connectorEL") == (2, 0)
    assert _branch_and_segment_for_vessel("branch1_seg0_connector0") == (1, 1)
    assert _branch_and_segment_for_vessel("branch1_seg0_connector4") == (1, 5)
    with pytest.raises(ValueError, match="branch<id>_seg<id>"):
        _branch_and_segment_for_vessel("lpa_main")


def _vessel(vessel_id, name, length, bcs=None):
    vessel = {"vessel_id": vessel_id, "vessel_name": name, "vessel_length": length}
    if bcs:
        vessel["boundary_conditions"] = bcs
    return vessel


def _learned_config():
    # MPA -> J0 -> branch1 connectorEL -> J1 -> (outlet, split connector -> outlet)
    return {
        "vessels": [
            _vessel(0, "branch0_seg0", 0.5, {"inlet": "INFLOW"}),
            _vessel(1, "branch1_seg0_connectorEL", 0.0),
            _vessel(2, "branch1_seg0_connector0", 0.1),
            _vessel(3, "branch2_seg0_connectorEL", 0.0, {"outlet": "OUT_A"}),
            _vessel(4, "branch3_seg0_connectorEL", 0.0, {"outlet": "OUT_B"}),
            _vessel(5, "branch4_seg0_connectorEL", 0.0, {"outlet": "OUT_C"}),
        ],
        "junctions": [
            {"junction_name": "J0", "inlet_vessels": [0], "outlet_vessels": [1, 5]},
            {"junction_name": "J1", "inlet_vessels": [1], "outlet_vessels": [3, 2]},
            {"junction_name": "J2", "inlet_vessels": [2], "outlet_vessels": [4]},
        ],
    }


def test_folded_connector_length_comes_from_observed_branch_extent():
    derived: dict[str, float] = {}
    topology, upstream, downstream = _network_topology(
        _learned_config(),
        branch_path_extents={0: 0.5, 1: 1.5, 2: 0.8, 3: 0.6, 4: 0.9},
        derived_lengths=derived,
    )
    assert derived == pytest.approx(
        {
            "branch1_seg0_connectorEL": 1.4,
            "branch2_seg0_connectorEL": 0.8,
            "branch3_seg0_connectorEL": 0.6,
            "branch4_seg0_connectorEL": 0.9,
        }
    )
    el = topology["branch1_seg0_connectorEL"]
    split = topology["branch1_seg0_connector0"]
    assert (el.seg_id, el.start_path, el.end_path) == (0, 0.0, pytest.approx(1.4))
    assert (split.seg_id, split.start_path, split.end_path) == (1, pytest.approx(1.4), pytest.approx(1.5))
    assert upstream["branch1_seg0_connectorEL"] == "J0"
    assert downstream["branch1_seg0_connector0"] == "J2"


def test_folded_connector_without_observed_extent_still_fails():
    with pytest.raises(ValueError, match="vessel_length must be positive"):
        _network_topology(_learned_config())
