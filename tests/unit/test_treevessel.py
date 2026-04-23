from __future__ import annotations

import numpy as np
import pytest

from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.treevessel import TreeVessel


def _vessel(vessel_id: int, *, diameter: float = 1.0, gen: int = 0) -> TreeVessel:
    return TreeVessel.create_vessel(
        id=vessel_id,
        gen=gen,
        diameter=diameter,
        density=1.06,
        lrr=2.0,
        compliance_model=ConstantCompliance(1.0),
    )


def test_create_vessel_derives_geometry_and_zero_d_values():
    vessel = _vessel(3, diameter=0.8, gen=2)
    radius = 0.4
    area = np.pi * radius**2
    length = 2.0 * radius
    expected_r = 8 * 0.049 * length / (np.pi * radius**4)

    assert vessel.id == 3
    assert vessel.gen == 2
    assert vessel.name == "branch3_seg0"
    assert vessel.l == pytest.approx(length)
    assert vessel.a == pytest.approx(area)
    assert vessel.R == pytest.approx(expected_r)
    assert vessel.params["zero_d_element_values"]["C"] == pytest.approx(3 * area / 2)
    assert vessel.params["zero_d_element_values"]["L"] == pytest.approx(1.06 * length / area)


def test_to_dict_serializes_current_vessel_state():
    vessel = _vessel(1, diameter=0.5)
    serialized = vessel.to_dict()

    assert serialized["vessel_id"] == 1
    assert serialized["vessel_name"] == "branch1_seg0"
    assert serialized["vessel_D"] == pytest.approx(vessel.d)
    assert serialized["zero_d_element_values"]["R_poiseuille"] == pytest.approx(vessel.R)
    assert serialized["generation"] == 0


def test_equivalent_resistance_updates_for_parallel_children():
    root = _vessel(0, diameter=1.0)
    left = _vessel(1, diameter=0.8, gen=1)
    right = _vessel(2, diameter=0.6, gen=1)

    root.left = left
    root.right = right

    expected = root.R + 1.0 / (1.0 / left.R + 1.0 / right.R)
    assert root.R_eq == pytest.approx(expected)

    left.R = left.R * 2.0
    expected = root.R + 1.0 / (1.0 / left.R + 1.0 / right.R)
    assert root.R_eq == pytest.approx(expected)


def test_diameter_and_radius_setters_update_zero_d_values():
    root = _vessel(0, diameter=1.0)
    root.left = _vessel(1, diameter=0.8, gen=1)
    root.right = _vessel(2, diameter=0.6, gen=1)

    root.d = 0.7
    assert root.r == pytest.approx(0.35)
    assert root.params["vessel_D"] == pytest.approx(0.7)
    assert root.params["vessel_length"] == pytest.approx(2.0 * 0.35)
    assert root.params["zero_d_element_values"]["R_poiseuille"] == pytest.approx(root.R)

    root.r = 0.25
    assert root.d == pytest.approx(0.5)
    assert root.params["vessel_D"] == pytest.approx(0.5)
    assert root.params["vessel_length"] == pytest.approx(2.0 * 0.25)


def test_wall_shear_and_intramural_stress_formulas():
    vessel = _vessel(1, diameter=1.0)
    q = np.array([1.0, 2.0])
    p = np.array([10.0, 20.0])

    expected_wss = q * 4 * vessel.eta / (np.pi * vessel.r**3)
    np.testing.assert_allclose(vessel.wall_shear_stress(Q=q, mean=False), expected_wss)
    assert vessel.wall_shear_stress(Q=q) == pytest.approx(np.mean(expected_wss))

    expected_ims = p * vessel.r / vessel.h
    np.testing.assert_allclose(vessel.intramural_stress(P=p, mean=False), expected_ims)
    assert vessel.intramural_stress(P=p) == pytest.approx(np.mean(expected_ims))


def test_collapsed_vessel_adds_distal_pressure_boundary_condition():
    vessel = _vessel(4, diameter=0.5)

    vessel.collapsed = True

    assert vessel.params["boundary_conditions"] == {"outlet": "P_d4"}
