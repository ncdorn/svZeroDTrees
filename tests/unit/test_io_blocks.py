from __future__ import annotations

import math

import pytest

from svzerodtrees.io.blocks import Chamber, CouplingBlock, Junction, Valve, Vessel
from svzerodtrees.io.blocks.boundary_condition import BoundaryCondition


def test_chamber_and_valve_round_trip():
    chamber = Chamber.from_config(
        {"name": "left_heart", "type": "HEART", "values": {"Emax": 1.0}}
    )
    valve = Valve.from_config(
        {"name": "mitral", "type": "VALVE", "params": {"Rmax": 10.0}}
    )

    assert chamber.to_dict() == {
        "name": "left_heart",
        "type": "HEART",
        "values": {"Emax": 1.0},
    }
    assert valve.to_dict() == {
        "name": "mitral",
        "type": "VALVE",
        "params": {"Rmax": 10.0},
    }


def test_vessel_round_trip_and_unit_conversion(minimal_svzerod_payload):
    vessel = Vessel.from_config(minimal_svzerod_payload["vessels"][0])

    vessel.convert_to_cm()
    serialized = vessel.to_dict()

    assert serialized["vessel_id"] == 0
    assert serialized["boundary_conditions"] == {"inlet": "INFLOW", "outlet": "OUT"}
    assert serialized["zero_d_element_values"]["R_poiseuille"] == pytest.approx(1000.0)
    assert serialized["zero_d_element_values"]["L"] == pytest.approx(0.0)


def test_junction_from_vessel_computes_child_areas(minimal_svzerod_payload):
    parent = Vessel.from_config(minimal_svzerod_payload["vessels"][0])
    child = Vessel.from_config(
        {
            **minimal_svzerod_payload["vessels"][0],
            "vessel_id": 1,
            "vessel_name": "branch1_seg0",
        }
    )
    child.diameter = 2.0
    parent.children = [child]

    junction = Junction.from_vessel(parent)

    assert junction.to_dict()["junction_name"] == "J0"
    assert junction.to_dict()["inlet_vessels"] == [0]
    assert junction.to_dict()["outlet_vessels"] == [1]
    assert junction.to_dict()["areas"] == pytest.approx([math.pi])


def test_coupling_block_from_boundary_condition_omits_result_by_default():
    bc = BoundaryCondition.from_config(
        {
            "bc_name": "OUT_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {"R": 100.0, "Pd": 12.0},
        }
    )

    block = CouplingBlock.from_bc(bc, surface="outlet.vtp")
    serialized = block.to_dict()

    assert serialized["name"] == "OUTBC"
    assert serialized["connected_block"] == "OUT_BC"
    assert serialized["surface"] == "outlet.vtp"
    assert "result" not in serialized
