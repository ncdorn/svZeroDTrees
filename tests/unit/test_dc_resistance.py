import numpy as np
import pytest

from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.structured_tree.dc_resistance import (
    conductance_matched_diameter,
    structured_tree_dc_resistance,
    structured_tree_node_count,
)
from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree


@pytest.mark.parametrize(
    ("initial_d", "alpha", "beta"),
    [
        (0.005, 0.9, 0.6),
        (0.05, 0.9, 0.6),
        (0.12, 0.9, 0.6),
        (0.3, 0.9, 0.6),
        (0.05, 0.8, 0.8),
        (0.12, 0.8, 0.8),
    ],
)
def test_dc_resistance_matches_built_tree(initial_d, alpha, beta):
    tree = StructuredTree(
        name="t",
        time=[0.0, 1.0],
        simparams=None,
        compliance_model=ConstantCompliance(6.6e4),
    )
    tree.build(initial_d=initial_d, d_min=0.01, lrr=10.0, alpha=alpha, beta=beta)

    assert structured_tree_node_count(
        initial_d, d_min=0.01, alpha=alpha, beta=beta
    ) == tree.store.d.size
    expected = tree.equivalent_resistance()
    actual = structured_tree_dc_resistance(
        initial_d,
        d_min=0.01,
        alpha=alpha,
        beta=beta,
        lrr=10.0,
        eta=tree.viscosity,
    )

    assert actual == pytest.approx(expected, rel=1e-7)


def test_conductance_matched_diameter_matches_total_conductance():
    diameters = np.array([0.08, 0.12, 0.15, 0.2, 0.3, 0.35])

    d_ref, residual = conductance_matched_diameter(
        diameters, d_min=0.01, alpha=0.9, beta=0.6
    )

    def conductance(d):
        return 1.0 / structured_tree_dc_resistance(d, d_min=0.01, alpha=0.9, beta=0.6)

    total = sum(conductance(d) for d in diameters)
    assert diameters.min() <= d_ref <= diameters.max()
    assert abs(residual) < 1e-9
    assert diameters.size * conductance(d_ref) == pytest.approx(total, rel=1e-9)
    # The arithmetic-mean tree under-represents the per-outlet conductance,
    # which is the bias conductance matching removes.
    assert diameters.size * conductance(diameters.mean()) < 0.9 * total


def test_conductance_matched_diameter_of_identical_caps_is_that_diameter():
    d_ref, residual = conductance_matched_diameter(
        [0.2, 0.2, 0.2], d_min=0.01, alpha=0.9, beta=0.6
    )

    assert d_ref == pytest.approx(0.2)
    assert residual == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("diameters", "kwargs", "message"),
    [
        ([], {}, "at least one diameter"),
        ([0.1, -0.2], {}, "finite diameters > 0"),
        ([0.1], {"alpha": 1.2}, "alpha must be in"),
        ([0.1], {"d_min": 0.0}, "d_min must be"),
    ],
)
def test_conductance_matched_diameter_rejects_invalid_inputs(diameters, kwargs, message):
    options = {"d_min": 0.01, "alpha": 0.9, "beta": 0.6, **kwargs}

    with pytest.raises(ValueError, match=message):
        conductance_matched_diameter(diameters, **options)


def test_conductance_matched_diameter_warns_when_trees_would_be_truncated():
    with pytest.warns(UserWarning, match="max_nodes=1000"):
        conductance_matched_diameter(
            [0.1, 0.3], d_min=0.01, alpha=0.9, beta=0.6, max_nodes=1000
        )
