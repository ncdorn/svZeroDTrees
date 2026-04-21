import numpy as np
import pytest

from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree
from svzerodtrees.io.blocks.simulation_parameters import SimParams
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance
from svzerodtrees.io.blocks.boundary_condition import (
    BoundaryCondition,
    validate_flow_cardiac_output_config,
    validate_impedance_timing_config,
)


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


def _legacy_zero_termination_kernel(tree: StructuredTree, *, tsteps: int) -> np.ndarray:
    """Reference the pre-vectorized recursive tree impedance implementation."""
    store = tree.store
    diameter = np.asarray(store.d, dtype=np.float64)
    area = np.pi * (diameter / 2.0) ** 2
    length = float(store.lrr) * (diameter / 2.0)
    left = np.asarray(store.left, dtype=np.int32)
    right = np.asarray(store.right, dtype=np.int32)
    collapsed = np.asarray(store.collapsed, dtype=bool)
    density = float(store.density)
    viscosity = float(store.eta)
    compliance_model = store.compliance_model

    period = max(tree.time) * tree.q / tree.Lr**3
    df = 1.0 / period
    omega = np.array(
        [i * df * 2.0 * np.pi for i in range(-tsteps // 2, tsteps // 2)],
        dtype=np.float64,
    )

    def _z0_dc(idx: int) -> float:
        radius = diameter[idx] / 2.0
        segment_resistance = 8.0 * viscosity * length[idx] / (np.pi * radius**4)
        if collapsed[idx]:
            return float(segment_resistance)
        z_left = _z0_dc(int(left[idx]))
        z_right = _z0_dc(int(right[idx]))
        return float(segment_resistance + 1.0 / (1.0 / z_left + 1.0 / z_right))

    def _z0(idx: int, omega_value: float) -> complex:
        if collapsed[idx]:
            z_load = 0.0
        else:
            z_left = _z0(int(left[idx]), omega_value)
            z_right = _z0(int(right[idx]), omega_value)
            z_load = 1.0 / (1.0 / z_left + 1.0 / z_right)

        if omega_value == 0.0:
            return complex(_z0_dc(idx), 0.0)

        eh_over_r = compliance_model.evaluate(diameter[idx] / 2.0)
        compliance = 3.0 * area[idx] / (2.0 * eh_over_r)
        wom = diameter[idx] / 2.0 * np.sqrt(omega_value * density / viscosity)
        if wom > 3.0:
            term = np.sqrt(
                1.0
                - 2.0 / 1j**0.5 / wom * (1.0 + 1.0 / (2.0 * wom))
            )
        elif wom > 2.0:
            term = (
                (3.0 - wom) * np.sqrt(1j * wom**2.0 / 8.0 + wom**4.0 / 48.0)
                + (wom - 2.0)
                * np.sqrt(
                    1.0
                    - 2.0 / 1j**0.5 / wom * (1.0 + 1.0 / (2.0 * wom))
                )
            )
        elif wom == 0.0:
            term = 0.0
        else:
            term = np.sqrt(1j * wom**2 / 8.0 + wom**4 / 48.0)

        g_omega = np.sqrt(compliance * area[idx] / density) * term
        c_omega = np.sqrt(area[idx] / compliance / density) * term
        kappa = omega_value * length[idx] / c_omega
        t1 = 1j * np.sin(kappa) / g_omega + np.cos(kappa) * z_load
        t2 = np.cos(kappa) + 1j * g_omega * z_load * np.sin(kappa)
        return t1 / t2

    z_omega = np.zeros(len(omega), dtype=complex)
    for idx in range(0, tsteps // 2 + 1):
        z_omega[idx] = np.conjugate(_z0(0, abs(float(omega[idx]))))
    z_omega_half = z_omega[: tsteps // 2]
    z_omega[tsteps // 2 + 1 :] = np.conjugate(np.flipud(z_omega_half[:-1]))
    return np.real(np.fft.ifft(np.fft.ifftshift(z_omega)))


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


def test_tree_metadata_round_trip_preserves_rebuild_inputs(simple_tree):
    simple_tree.inductance = 0.05
    simple_tree.generation_mode = "per_outlet"
    simple_tree.outlet_mapping = {
        "mode": "per_outlet",
        "side": "lpa",
        "bc_names": ["IMPEDANCE_0"],
        "outlet_names": ["LPA_0.vtp"],
    }

    metadata = simple_tree.to_dict()
    rebuilt = StructuredTree.from_tree_metadata(
        metadata,
        time=[0.0, 0.5, 1.0],
        simparams=SimParams({}),
    )

    assert rebuilt.initial_d == pytest.approx(simple_tree.initial_d)
    assert rebuilt.d_min == pytest.approx(simple_tree.d_min)
    assert rebuilt.lrr == pytest.approx(simple_tree.lrr)
    assert rebuilt.alpha == pytest.approx(simple_tree.alpha)
    assert rebuilt.beta == pytest.approx(simple_tree.beta)
    assert rebuilt.inductance == pytest.approx(0.05)
    assert rebuilt.generation_mode == "per_outlet"
    assert rebuilt.outlet_mapping["bc_names"] == ["IMPEDANCE_0"]


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
    assert bc.values["Pd"] == pytest.approx(pd_value)
    assert bc.values["z"] == pytest.approx(kernel.tolist())
    assert set(bc.values) == {"z", "Pd"}


def test_impedance_bc_accepts_solver_schema():
    bc = BoundaryCondition.from_config(
        {
            "bc_name": "OUT",
            "bc_type": "IMPEDANCE",
            "bc_values": {
                "z": [10.0, 2.0, 1.0],
                "Pd": 3.0,
            },
        }
    )

    assert bc.Z == pytest.approx([10.0, 2.0, 1.0])
    assert bc.values["z"] == pytest.approx([10.0, 2.0, 1.0])
    assert "Z" not in bc.values


def test_impedance_bc_accepts_solver_options():
    bc = BoundaryCondition.from_config(
        {
            "bc_name": "OUT",
            "bc_type": "IMPEDANCE",
            "bc_values": {
                "z": [10.0, 2.0, 1.0],
                "Pd": 3.0,
                "convolution_mode": "truncated",
                "num_kernel_terms": 3,
            },
        }
    )

    assert bc.values["convolution_mode"] == "truncated"
    assert bc.values["num_kernel_terms"] == 3


@pytest.mark.parametrize(
    "bc_values, message",
    [
        ({"Z": [10.0, 2.0, 1.0], "Pd": 3.0}, "unsupported keys: Z"),
        ({"z": [10.0, 2.0, 1.0], "Pd": 3.0, "tree": 0}, "unsupported keys: tree"),
        ({"z": [10.0, 2.0, 1.0], "Pd": 3.0, "t": [0.0, 1.0]}, "unsupported keys: t"),
        ({"Pd": 3.0}, "missing required keys: z"),
        ({"z": [10.0, 2.0, 1.0]}, "missing required keys: Pd"),
        (
            {"z": [10.0, 2.0, 1.0], "Pd": 3.0, "convolution_mode": "exact", "num_kernel_terms": 3},
            "requires convolution_mode='truncated'",
        ),
        (
            {"z": [10.0, 2.0, 1.0], "Pd": 3.0, "extra": 1},
            "unsupported keys: extra",
        ),
    ],
)
def test_impedance_bc_rejects_legacy_or_invalid_schema(bc_values, message):
    with pytest.raises(ValueError, match=message):
        BoundaryCondition.from_config(
            {
                "bc_name": "OUT",
                "bc_type": "IMPEDANCE",
                "bc_values": bc_values,
            }
        )


def test_validate_impedance_timing_config_rejects_noncoupled_mismatch():
    with pytest.raises(ValueError, match="number_of_time_pts_per_cardiac_cycle = len\\(z\\) \\+ 1"):
        validate_impedance_timing_config(
            {
                "simulation_parameters": {
                    "number_of_cardiac_cycles": 1,
                    "number_of_time_pts_per_cardiac_cycle": 2,
                },
                "boundary_conditions": [
                    {
                        "bc_name": "OUT",
                        "bc_type": "IMPEDANCE",
                        "bc_values": {"z": [10.0, 2.0], "Pd": 3.0},
                    }
                ],
            }
        )


def test_validate_impedance_timing_config_rejects_coupled_wrong_number_of_time_pts():
    with pytest.raises(ValueError, match="number_of_time_pts = 2; got 3"):
        validate_impedance_timing_config(
            {
                "simulation_parameters": {
                    "coupled_simulation": True,
                    "number_of_time_pts": 3,
                    "output_all_cycles": True,
                    "steady_initial": False,
                    "density": 1.06,
                    "viscosity": 0.04,
                    "external_step_size": 1.0,
                    "cardiac_period": 1.0,
                },
                "boundary_conditions": [
                    {
                        "bc_name": "INFLOW",
                        "bc_type": "FLOW",
                        "bc_values": {"Q": [1.0, 1.0], "t": [0.0, 1.0]},
                    },
                    {
                        "bc_name": "OUT",
                        "bc_type": "IMPEDANCE",
                        "bc_values": {"z": [10.0], "Pd": 3.0},
                    }
                ],
            }
        )


def test_validate_impedance_timing_config_rejects_coupled_extra_simparam_keys():
    with pytest.raises(ValueError, match="does not allow extra simulation_parameters keys"):
        validate_impedance_timing_config(
            {
                "simulation_parameters": {
                    "coupled_simulation": True,
                    "number_of_time_pts": 2,
                    "output_all_cycles": True,
                    "steady_initial": False,
                    "density": 1.06,
                    "viscosity": 0.04,
                    "external_step_size": 1.0,
                    "cardiac_period": 1.0,
                    "number_of_time_pts_per_cardiac_cycle": 2,
                },
                "boundary_conditions": [
                    {
                        "bc_name": "INFLOW",
                        "bc_type": "FLOW",
                        "bc_values": {"Q": [1.0, 1.0], "t": [0.0, 1.0]},
                    },
                    {
                        "bc_name": "OUT",
                        "bc_type": "IMPEDANCE",
                        "bc_values": {"z": [1.0], "Pd": 3.0},
                    },
                ],
            }
        )


def test_validate_impedance_timing_config_rejects_coupled_missing_cardiac_period():
    with pytest.raises(ValueError, match="simulation_parameters keys: cardiac_period"):
        validate_impedance_timing_config(
            {
                "simulation_parameters": {
                    "coupled_simulation": True,
                    "number_of_time_pts": 2,
                    "output_all_cycles": True,
                    "steady_initial": False,
                    "density": 1.06,
                    "viscosity": 0.04,
                    "external_step_size": 1.0,
                },
                "boundary_conditions": [
                    {
                        "bc_name": "INFLOW",
                        "bc_type": "FLOW",
                        "bc_values": {"Q": [1.0, 1.0], "t": [0.0, 1.0]},
                    },
                    {
                        "bc_name": "OUT",
                        "bc_type": "IMPEDANCE",
                        "bc_values": {"z": [1.0], "Pd": 3.0},
                    },
                ],
            }
        )


def test_validate_flow_cardiac_output_config_accepts_matching_inflow():
    measured = validate_flow_cardiac_output_config(
        {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": [2.0, 2.0],
                        "t": [0.0, 1.0],
                    },
                }
            ]
        },
        expected_cardiac_output=2.0,
    )

    assert measured == pytest.approx(2.0)


def test_validate_flow_cardiac_output_config_rejects_mismatched_output():
    with pytest.raises(ValueError, match="expected 1.5, got 2"):
        validate_flow_cardiac_output_config(
            {
                "boundary_conditions": [
                    {
                        "bc_name": "INFLOW",
                        "bc_type": "FLOW",
                        "bc_values": {
                            "Q": [2.0, 2.0],
                            "t": [0.0, 1.0],
                        },
                    }
                ]
            },
            expected_cardiac_output=1.5,
        )


def test_validate_flow_cardiac_output_config_rejects_malformed_inflow():
    with pytest.raises(ValueError, match="must have the same length"):
        validate_flow_cardiac_output_config(
            {
                "boundary_conditions": [
                    {
                        "bc_name": "INFLOW",
                        "bc_type": "FLOW",
                        "bc_values": {
                            "Q": [2.0, 2.0],
                            "t": [0.0],
                        },
                    }
                ]
            },
            expected_cardiac_output=2.0,
        )


def test_validate_flow_cardiac_output_config_uses_mean_for_nonunit_period():
    measured = validate_flow_cardiac_output_config(
        {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": [2.0, 2.0, 2.0],
                        "t": [0.0, 0.4, 0.8],
                    },
                }
            ]
        },
        expected_cardiac_output=2.0,
    )

    assert measured == pytest.approx(2.0)


def test_compute_olufsen_impedance_defaults_to_legacy_zero_termination(simple_tree):
    tsteps = 2
    expected = _legacy_zero_termination_kernel(simple_tree, tsteps=tsteps)
    actual, _ = simple_tree.compute_olufsen_impedance(tsteps=tsteps)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_compute_olufsen_impedance_reflectionless_is_opt_in(simple_tree):
    zero_kernel, _ = simple_tree.compute_olufsen_impedance(tsteps=2, leaf_termination="zero")
    reflectionless_kernel, _ = simple_tree.compute_olufsen_impedance(
        tsteps=2,
        leaf_termination="reflectionless",
    )
    assert np.max(np.abs(np.asarray(zero_kernel) - np.asarray(reflectionless_kernel))) > 1e-6
