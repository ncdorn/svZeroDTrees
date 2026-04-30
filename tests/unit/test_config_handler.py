from __future__ import annotations

import numpy as np
import pytest

import svzerodtrees._pysvzerod as pysvzerod_loader
from svzerodtrees.io.config_handler import ConfigHandler


def _vessel(vessel_id, name, *, resistance=10.0, bc=None, length=1.0):
    config = {
        "vessel_id": vessel_id,
        "vessel_name": name,
        "vessel_length": length,
        "zero_d_element_type": "BloodVessel",
        "zero_d_element_values": {
            "R_poiseuille": resistance,
            "C": 1.0,
            "L": 0.0,
            "stenosis_coefficient": 0.0,
        },
    }
    if bc is not None:
        config["boundary_conditions"] = bc
    return config


def _bifurcation_config(*, coupled=False, trees=None):
    simparams = (
        {
            "coupled_simulation": True,
            "number_of_time_pts": 2,
            "output_all_cycles": True,
            "steady_initial": False,
            "density": 1.06,
            "viscosity": 0.04,
            "external_step_size": 0.5,
            "cardiac_period": 1.0,
        }
        if coupled
        else {
            "number_of_time_pts_per_cardiac_cycle": 3,
            "number_of_cardiac_cycles": 1,
        }
    )
    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": [2.0, 2.0, 2.0], "t": [0.0, 0.5, 1.0]},
            },
            {
                "bc_name": "LPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 100.0, "Pd": 8.0},
            },
            {
                "bc_name": "RPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 200.0, "Pd": 8.0},
            },
        ],
        "simulation_parameters": simparams,
        "vessels": [
            _vessel(0, "branch0_seg0", resistance=10.0, bc={"inlet": "INFLOW"}),
            _vessel(1, "branch1_seg0", resistance=20.0, bc={"outlet": "LPA_BC"}),
            _vessel(2, "branch2_seg0", resistance=30.0, bc={"outlet": "RPA_BC"}),
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "areas": [1.0, 1.0],
            }
        ],
        **({"external_solver_coupling_blocks": []} if coupled else {}),
        **({"trees": trees} if trees is not None else {}),
    }


def test_branch_and_vessel_maps_with_pulmonary_labels():
    handler = ConfigHandler(_bifurcation_config(), is_pulmonary=True)

    assert set(handler.vessel_map) == {0, 1, 2}
    assert set(handler.branch_map) == {0, 1, 2}
    assert handler.root.id == 0
    assert [child.id for child in handler.root.children] == [1, 2]
    assert handler.mpa is handler.root
    assert handler.lpa.id == 1
    assert handler.rpa.id == 2
    assert handler.get_segments("lpa")[0].id == 1


def test_get_time_series_for_standard_and_coupled_configs():
    standard = ConfigHandler(_bifurcation_config())
    np.testing.assert_allclose(standard.get_time_series(), [0.0, 0.5, 1.0])

    coupled = ConfigHandler(_bifurcation_config(coupled=True))
    np.testing.assert_allclose(coupled.get_time_series(), [0.0, 0.5, 1.0])


def test_canonical_coupled_simulation_parameters_resolve_from_inflow():
    handler = ConfigHandler(_bifurcation_config(coupled=True), is_threed_interface=True)
    handler.simparams.external_step_size = None
    handler.simparams.cardiac_period = None
    handler.simparams.number_of_time_pts_per_cardiac_cycle = 3

    resolved = handler._canonical_coupled_simulation_parameters()

    assert resolved == {
        "coupled_simulation": True,
        "number_of_time_pts": 2,
        "output_all_cycles": True,
        "steady_initial": False,
        "density": 1.06,
        "viscosity": 0.04,
        "external_step_size": pytest.approx(0.5),
        "cardiac_period": pytest.approx(1.0),
    }


def test_resolve_impedance_tree_metadata_success_and_failures():
    config = _bifurcation_config(
        trees=[
            {
                "name": "lpa_tree",
                "inductance": 0.05,
                "outlet_mapping": {"bc_names": ["LPA_BC"]},
            }
        ]
    )
    config["boundary_conditions"][1] = {
        "bc_name": "LPA_BC",
        "bc_type": "IMPEDANCE",
        "bc_values": {"z": [1.0, 0.5], "Pd": 8.0},
    }
    handler = ConfigHandler(config)

    entries = handler._resolve_impedance_tree_metadata()
    assert len(entries) == 1
    assert entries[0][0] == "lpa_tree"
    assert entries[0][3] == ["LPA_BC"]
    assert handler.bc_inductance["LPA_BC"] == pytest.approx(0.05)

    config_without_trees = _bifurcation_config()
    config_without_trees["boundary_conditions"][1] = config["boundary_conditions"][1]
    with pytest.raises(RuntimeError, match="top-level 'trees'"):
        ConfigHandler(config_without_trees)._resolve_impedance_tree_metadata()

    missing_mapping = _bifurcation_config(trees=[{"name": "bad_tree"}])
    missing_mapping["boundary_conditions"][1] = config["boundary_conditions"][1]
    with pytest.raises(RuntimeError, match="Missing outlet mapping"):
        ConfigHandler(missing_mapping)._resolve_impedance_tree_metadata()


def test_branch_resistance_mutation_and_equivalent_resistance():
    payload = _bifurcation_config()
    payload["vessels"].insert(
        1,
        _vessel(3, "branch0_seg1", resistance=15.0, length=1.5),
    )
    payload["junctions"] = [
        {
            "junction_name": "J_internal",
            "junction_type": "NORMAL_JUNCTION",
            "inlet_vessels": [0],
            "outlet_vessels": [3],
            "areas": [1.0],
        },
        {
            "junction_name": "J0",
            "junction_type": "NORMAL_JUNCTION",
            "inlet_vessels": [3],
            "outlet_vessels": [1, 2],
            "areas": [1.0, 1.0],
        },
    ]
    handler = ConfigHandler(payload)

    assert handler.get_branch_resistance(0) == pytest.approx(25.0)

    handler.change_branch_resistance(0, 50.0)
    assert handler.branch_map[0].R == pytest.approx(50.0)
    assert handler.get_branch_resistance(0) == pytest.approx(50.0)

    handler.change_branch_resistance(0, [5.0, 7.0])
    assert [vessel.R for vessel in handler.get_segments(0)] == pytest.approx([5.0, 7.0])

    handler.compute_R_eq()
    expected_lpa = handler.vessel_map[1].R + handler.bcs["LPA_BC"].R
    expected_rpa = handler.vessel_map[2].R + handler.bcs["RPA_BC"].R
    expected_parallel = 1.0 / (1.0 / expected_lpa + 1.0 / expected_rpa)
    assert handler.root.id == 3
    assert handler.root.R_eq == pytest.approx(7.0 + expected_parallel)


def test_equivalent_resistance_includes_terminal_boundary_conditions_for_connectors():
    payload = _bifurcation_config()
    payload["vessels"][1]["zero_d_element_values"]["R_poiseuille"] = 0.0
    payload["vessels"][2]["zero_d_element_values"]["R_poiseuille"] = 0.0
    handler = ConfigHandler(payload)

    assert handler.vessel_map[1].R_eq == pytest.approx(100.0)
    assert handler.vessel_map[2].R_eq == pytest.approx(200.0)
    expected_parallel = 1.0 / (1.0 / 100.0 + 1.0 / 200.0)
    assert handler.root.R_eq == pytest.approx(10.0 + expected_parallel)


def test_simulate_defers_missing_pysvzerod_until_solver_call(monkeypatch):
    pysvzerod_loader.require_pysvzerod.cache_clear()

    def _missing_solver(_name):
        exc = ModuleNotFoundError("No module named 'pysvzerod'")
        exc.name = "pysvzerod"
        raise exc

    monkeypatch.setattr(pysvzerod_loader.importlib, "import_module", _missing_solver)

    handler = ConfigHandler(_bifurcation_config())
    assert handler.root.id == 0

    with pytest.raises(ModuleNotFoundError, match="Install the sibling svZeroDSolver checkout first"):
        handler.simulate()
