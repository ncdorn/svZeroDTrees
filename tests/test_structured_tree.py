import numpy as np
import pytest

from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree
from svzerodtrees.io.blocks.simulation_parameters import SimParams
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance
from svzerodtrees.io.blocks.boundary_condition import BoundaryCondition


@pytest.fixture
def simple_tree():
    """StructuredTree with a shallow binary tree (root + two leaves)."""
    sim_params = SimParams({})
    tree = StructuredTree(
        name="unit_test_tree",
        time=[0.0, 0.5, 1.0],
        simparams=sim_params,
        compliance_model=ConstantCompliance(1.0),
    )
    tree.build(initial_d=1.0, d_min=0.6, alpha=0.5, beta=0.5, lrr=2.0)
    return tree


def test_segment_resistances_requires_built_store():
    sim_params = SimParams({})
    tree = StructuredTree(
        name="unbuilt_tree",
        time=[0.0, 1.0],
        simparams=sim_params,
        compliance_model=ConstantCompliance(1.0),
    )
    with pytest.raises(RuntimeError):
        tree.segment_resistances()


def test_segment_resistances_matches_poiseuille(simple_tree):
    resistances = simple_tree.segment_resistances()
    store = simple_tree.store
    radii = 0.5 * np.asarray(store.d, dtype=np.float64)
    lengths = float(store.lrr) * radii
    expected = 8.0 * float(store.eta) * lengths / (np.pi * np.maximum(radii**4, np.finfo(np.float64).tiny))
    np.testing.assert_allclose(resistances, expected)


def test_equivalent_resistance_reduces_series_and_parallel(simple_tree):
    seg_R = simple_tree.segment_resistances()
    expected_root = float(seg_R[0] + 1.0 / (1.0 / seg_R[1] + 1.0 / seg_R[2]))
    assert simple_tree.equivalent_resistance() == pytest.approx(expected_root)


def test_create_resistance_bc_uses_equivalent_resistance(simple_tree):
    pd_value = 5.0
    bc = simple_tree.create_resistance_bc("test_res_bc", Pd=pd_value)
    assert isinstance(bc, BoundaryCondition)
    assert bc.type == "RESISTANCE"
    assert bc.values["Pd"] == pytest.approx(pd_value)
    assert bc.values["R"] == pytest.approx(simple_tree.equivalent_resistance())


def test_create_impedance_bc_serializes_kernel(simple_tree):
    kernel = np.array([1.0, 0.5, 0.25])
    simple_tree.Z_t = kernel
    tree_id = 7
    pd_value = 12.0
    bc = simple_tree.create_impedance_bc("test_imp_bc", tree_id=tree_id, Pd=pd_value)
    assert isinstance(bc, BoundaryCondition)
    assert bc.type == "IMPEDANCE"
    assert bc.values["tree"] == tree_id
    assert bc.values["Pd"] == pytest.approx(pd_value)
    assert bc.values["Z"] == pytest.approx(kernel.tolist())
    assert bc.values["t"] == simple_tree.time


