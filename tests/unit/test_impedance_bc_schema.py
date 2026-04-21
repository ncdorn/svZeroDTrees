from __future__ import annotations

import pytest

from svzerodtrees.io.blocks.boundary_condition import (
    resolve_coupled_impedance_timepoint_contract,
    validate_impedance_timing_config,
)


def _noncoupled_impedance_config() -> dict[str, object]:
    return {
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 3,
            "number_of_cardiac_cycles": 1,
        },
        "boundary_conditions": [
            {
                "bc_name": "OUT",
                "bc_type": "IMPEDANCE",
                "bc_values": {"z": [1.0, 0.5], "Pd": 8.0},
            }
        ],
    }


def _coupled_impedance_config() -> dict[str, object]:
    return {
        "simulation_parameters": {
            "coupled_simulation": True,
            "number_of_time_pts": 2,
            "output_all_cycles": True,
            "steady_initial": False,
            "density": 1.06,
            "viscosity": 0.04,
            "external_step_size": 0.5,
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
                "bc_values": {"z": [1.0, 0.5], "Pd": 8.0},
            },
        ],
    }


def test_noncoupled_impedance_config_uses_canonical_solver_schema():
    validate_impedance_timing_config(_noncoupled_impedance_config())


def test_coupled_impedance_config_uses_canonical_solver_schema():
    sample_count, kernel_steps = resolve_coupled_impedance_timepoint_contract(
        _coupled_impedance_config()
    )

    assert sample_count == 2
    assert kernel_steps == 2


def test_coupled_dirichlet_inlet_config_may_omit_flow_bc():
    config = _coupled_impedance_config()
    config["boundary_conditions"] = [
        bc
        for bc in config["boundary_conditions"]
        if bc["bc_type"] != "FLOW"
    ]

    sample_count, kernel_steps = resolve_coupled_impedance_timepoint_contract(config)

    assert sample_count == 2
    assert kernel_steps == 2


@pytest.mark.parametrize("legacy_key", ["Z", "tree", "t"])
def test_impedance_config_rejects_legacy_schema_keys(legacy_key):
    config = _noncoupled_impedance_config()
    values = config["boundary_conditions"][0]["bc_values"]
    values[legacy_key] = values.pop("z") if legacy_key == "Z" else 1

    with pytest.raises(ValueError, match=f"unsupported keys: {legacy_key}"):
        validate_impedance_timing_config(config)
