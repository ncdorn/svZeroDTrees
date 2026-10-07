"""Coupling-block surfaces must follow the cap each outlet BC was built for.

``MeshComplete.mesh_surfaces`` lists the inflow cap first and then the outlet
caps alphabetically, while a tuned full-PA config keeps its learned seed's BC
order (centerline order).  The 3D coupler must therefore pair IMPEDANCE BCs
with caps through the tree ``outlet_mapping``, never by list position.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from svzerodtrees.adaptation.tuned_trees import (
    check_coupler_matches_tuned_model,
    load_tuned_tree_model,
)
from svzerodtrees.io.config_handler import ConfigHandler

# Centerline-ordered seed BCs against alphabetically sorted caps: pairing by
# position would put IMPEDANCE_0's RPA tree on LPA_1 and an LPA tree on RPA_1.
CAPS = {
    "IMPEDANCE_0": ("/tuning/mesh-surfaces/RPA_1.vtp", "rpa"),
    "IMPEDANCE_1": ("/tuning/mesh-surfaces/LPA_2.vtp", "lpa"),
    "IMPEDANCE_2": ("/tuning/mesh-surfaces/LPA_1.vtp", "lpa"),
}
EXPECTED_SURFACES = {"IMPEDANCE_0": "RPA_1.vtp", "IMPEDANCE_1": "LPA_2.vtp", "IMPEDANCE_2": "LPA_1.vtp"}
MESH_CAPS = ("inflow.vtp", "LPA_1.vtp", "LPA_2.vtp", "RPA_1.vtp")


def _mesh_complete(filenames=MESH_CAPS):
    return SimpleNamespace(
        mesh_surfaces={
            name: SimpleNamespace(filename=name, path=f"/sim/mesh-complete/mesh-surfaces/{name}")
            for name in filenames
        }
    )


def _tree_entry(name, side, bc_names, outlet_names):
    return {
        "name": name,
        "initial_d": 0.05,
        "d_min": 0.01,
        "lrr": 10.0,
        "compliance": {"model": "constant", "value": 6.6e4},
        "inductance": 0.05,
        "outlet_mapping": {
            "mode": "per_outlet" if len(bc_names) == 1 else "shared_by_side",
            "side": side,
            "bc_names": list(bc_names),
            "outlet_names": list(outlet_names),
        },
    }


def _per_cap_trees(caps=CAPS):
    return [_tree_entry(cap, side, [bc], [cap]) for bc, (cap, side) in caps.items()]


def _shared_trees(caps=CAPS):
    trees = []
    for side in ("lpa", "rpa"):
        pairs = [(bc, cap) for bc, (cap, cap_side) in caps.items() if cap_side == side]
        trees.append(_tree_entry(side.upper(), side, [bc for bc, _ in pairs], [cap for _, cap in pairs]))
    return trees


OUTLET_BC_VALUES = {
    "IMPEDANCE": {"z": [1.0, 0.5], "Pd": 8.0},
    "RCR": {"Rp": 100.0, "C": 1e-4, "Rd": 900.0, "Pd": 8.0},
    "RESISTANCE": {"R": 1000.0, "Pd": 8.0},
}


def _outlet_bc(name, bc_type):
    return {"bc_name": name, "bc_type": bc_type, "bc_values": dict(OUTLET_BC_VALUES[bc_type])}


def _config(outlets, trees):
    """0D config: INFLOW -> junction -> one vessel per (bc_name, bc_type) outlet."""
    return {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [10.0, 12.0, 10.0], "t": [0.0, 0.4, 0.8]}}
        ]
        + [_outlet_bc(name, bc_type) for name, bc_type in outlets],
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 3,
            "number_of_cardiac_cycles": 1,
            "density": 1.06,
            "viscosity": 0.04,
        },
        "vessels": [
            {
                "boundary_conditions": {"inlet": "INFLOW"} if vessel_id == 0 else {"outlet": outlets[vessel_id - 1][0]},
                "vessel_id": vessel_id,
                "vessel_length": 10.0,
                "vessel_name": f"branch{vessel_id}_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"C": 0.0, "L": 0.0, "R_poiseuille": 1.0, "stenosis_coefficient": 0.0},
            }
            for vessel_id in range(len(outlets) + 1)
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [0],
                "outlet_vessels": list(range(1, len(outlets) + 1)),
                "areas": [1.0] * len(outlets),
            }
        ],
        "trees": trees,
    }


def _impedance_config(trees, bc_names=tuple(CAPS)):
    return _config([(name, "IMPEDANCE") for name in bc_names], trees)


def _surfaces(coupler):
    return {name: Path(str(block.surface)).name for name, block in coupler.coupling_blocks.items()}


@pytest.mark.parametrize("trees", [_per_cap_trees, _shared_trees], ids=["per_cap", "shared_by_side"])
@pytest.mark.parametrize("include_distal_vessel", [False, True], ids=["no_distal", "distal_vessel"])
def test_impedance_blocks_couple_to_the_tree_mapped_caps(tmp_path, trees, include_distal_vessel):
    handler = ConfigHandler(_impedance_config(trees()), is_pulmonary=False)

    coupler, _ = handler.generate_threed_coupler(
        str(tmp_path),
        inflow_from_0d=True,
        mesh_complete=_mesh_complete(),
        include_distal_vessel=include_distal_vessel,
    )

    surfaces = _surfaces(coupler)
    assert surfaces.pop("inflow") == "INFLOW.vtp"
    assert surfaces == EXPECTED_SURFACES
    # The adaptation stage accepts both the in-memory coupler and the written JSON.
    check_coupler_matches_tuned_model(coupler, load_tuned_tree_model(handler))
    written = ConfigHandler.from_json(str(tmp_path / "svzerod_3Dcoupling.json"), is_pulmonary=False)
    check_coupler_matches_tuned_model(written, load_tuned_tree_model(written))


def test_dirichlet_coupler_uses_tree_mapped_caps(tmp_path):
    handler = ConfigHandler(_impedance_config(_per_cap_trees()), is_pulmonary=False)

    coupler, _ = handler.generate_threed_coupler(
        str(tmp_path), inflow_from_0d=False, mesh_complete=_mesh_complete()
    )

    assert _surfaces(coupler) == EXPECTED_SURFACES


def test_impedance_bc_without_tree_mapping_fails(tmp_path):
    trees = [tree for tree in _per_cap_trees() if tree["outlet_mapping"]["bc_names"] != ["IMPEDANCE_1"]]
    handler = ConfigHandler(_impedance_config(trees), is_pulmonary=False)

    with pytest.raises(ValueError, match=r"IMPEDANCE_1.*no cap in the tree outlet_mapping"):
        handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())
    assert not (tmp_path / "svzerod_3Dcoupling.json").exists()


def test_impedance_bc_with_malformed_tree_mapping_fails(tmp_path):
    trees = _per_cap_trees()
    trees[0]["outlet_mapping"]["outlet_names"] = []
    handler = ConfigHandler(_impedance_config(trees), is_pulmonary=False)

    with pytest.raises(ValueError, match="one outlet_names entry per bc_names entry"):
        handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())


def test_mapped_cap_missing_from_mesh_fails(tmp_path):
    handler = ConfigHandler(_impedance_config(_per_cap_trees()), is_pulmonary=False)

    with pytest.raises(ValueError, match=r"IMPEDANCE_1.*'LPA_2.vtp'.*not a mesh surface"):
        handler.generate_threed_coupler(
            str(tmp_path), mesh_complete=_mesh_complete(("inflow.vtp", "LPA_1.vtp", "LPA_3.vtp", "RPA_1.vtp"))
        )


def test_two_bcs_mapped_to_one_cap_fail(tmp_path):
    caps = dict(CAPS, IMPEDANCE_1=CAPS["IMPEDANCE_2"])
    handler = ConfigHandler(_impedance_config(_shared_trees(caps)), is_pulmonary=False)

    with pytest.raises(ValueError, match=r"more than one outlet BC: 'LPA_1.vtp' <- \['IMPEDANCE_1', 'IMPEDANCE_2'\]"):
        handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())


def test_uncoupled_mesh_cap_fails(tmp_path):
    handler = ConfigHandler(_impedance_config(_per_cap_trees()), is_pulmonary=False)

    with pytest.raises(ValueError, match=r"no outlet BC for mesh caps \['RPA_2.vtp'\]"):
        handler.generate_threed_coupler(
            str(tmp_path), mesh_complete=_mesh_complete(MESH_CAPS + ("RPA_2.vtp",))
        )


def test_outlet_bcs_require_mesh_complete(tmp_path):
    handler = ConfigHandler(_impedance_config(_per_cap_trees()), is_pulmonary=False)

    with pytest.raises(ValueError, match="mesh_complete"):
        handler.generate_threed_coupler(str(tmp_path))


def test_cap_named_non_impedance_bcs_couple_by_name(tmp_path):
    # BC order differs from the alphabetical cap order; the names decide.
    outlets = [("RPA_1", "RCR"), ("LPA_2", "RESISTANCE"), ("LPA_1", "RCR")]
    handler = ConfigHandler(_config(outlets, []), is_pulmonary=False)

    coupler, _ = handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())

    surfaces = _surfaces(coupler)
    surfaces.pop("inflow")
    assert surfaces == {"RPA_1": "RPA_1.vtp", "LPA_2": "LPA_2.vtp", "LPA_1": "LPA_1.vtp"}


def test_unmapped_non_impedance_bcs_fall_back_to_cap_order_with_warning(tmp_path, capsys):
    outlets = [("RCR_0", "RCR"), ("RCR_1", "RCR"), ("RCR_2", "RCR")]
    handler = ConfigHandler(_config(outlets, []), is_pulmonary=False)

    coupler, _ = handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())

    surfaces = _surfaces(coupler)
    surfaces.pop("inflow")
    assert surfaces == {"RCR_0": "LPA_1.vtp", "RCR_1": "LPA_2.vtp", "RCR_2": "RPA_1.vtp"}
    assert "WARNING" in capsys.readouterr().out


def test_fallback_bcs_only_take_caps_no_mapped_bc_claimed(tmp_path):
    # IMPEDANCE_0 claims RPA_1 through its tree; the RCR BCs share what is left.
    trees = [_tree_entry(CAPS["IMPEDANCE_0"][0], "rpa", ["IMPEDANCE_0"], [CAPS["IMPEDANCE_0"][0]])]
    outlets = [("IMPEDANCE_0", "IMPEDANCE"), ("RCR_1", "RCR"), ("RCR_2", "RCR")]
    handler = ConfigHandler(_config(outlets, trees), is_pulmonary=False)

    coupler, _ = handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())

    surfaces = _surfaces(coupler)
    surfaces.pop("inflow")
    assert surfaces == {"IMPEDANCE_0": "RPA_1.vtp", "RCR_1": "LPA_1.vtp", "RCR_2": "LPA_2.vtp"}


def test_more_outlet_bcs_than_caps_fails(tmp_path):
    outlets = [("RCR_0", "RCR"), ("RCR_1", "RCR"), ("RCR_2", "RCR"), ("RCR_3", "RCR")]
    handler = ConfigHandler(_config(outlets, []), is_pulmonary=False)

    with pytest.raises(ValueError, match=r"RCR_3.*no mesh cap left"):
        handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())


def test_generate_threed_coupler_does_not_mutate_source_tree_metadata(tmp_path):
    trees = _per_cap_trees()
    handler = ConfigHandler(_impedance_config(copy.deepcopy(trees)), is_pulmonary=False)

    handler.generate_threed_coupler(str(tmp_path), mesh_complete=_mesh_complete())

    assert list(handler.tree_params.values()) == trees
