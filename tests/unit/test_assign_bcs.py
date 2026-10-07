from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from svzerodtrees.io.config_handler import ConfigHandler
from svzerodtrees.tune_bcs.assign_bcs import assign_rcr_bcs, validate_cap_to_bc_mapping
from svzerodtrees.tune_bcs.outlet_mapping import resolve_outlet_cap_mapping


class _BC:
    def __init__(self, name: str, bc_type: str = "RESISTANCE"):
        self.name = name
        self.type = bc_type


def _config() -> SimpleNamespace:
    names = ["OUT_0", "OUT_1"]
    return SimpleNamespace(
        _config={
            "boundary_conditions": [
                {"bc_name": "INFLOW", "bc_type": "FLOW"},
                *[{"bc_name": name, "bc_type": "RESISTANCE"} for name in names],
            ]
        },
        bcs={"INFLOW": _BC("INFLOW", "FLOW"), **{name: _BC(name) for name in names}},
        tree_params={},
        vessel_map={},
    )


def _patch_vtp_info(monkeypatch):
    caps = {
        "/mesh/lpa_terminal.vtp": 1.0,
        "/mesh/rpa_terminal.vtp": 4.0,
    }

    def fake_vtp_info(_mesh_path, *, convert_to_cm, pulmonary):
        assert convert_to_cm is False
        assert pulmonary is False
        return caps

    monkeypatch.setattr("svzerodtrees.tune_bcs.assign_bcs.vtp_info", fake_vtp_info)
    return caps


def test_validator_exposes_only_canonical_mapping_inputs():
    parameters = inspect.signature(validate_cap_to_bc_mapping).parameters

    assert tuple(parameters) == (
        "config_handler",
        "mesh_surfaces_path",
        "outlet_mapping_mode",
        "outlet_mapping",
        "resolved_mapping",
        "centerline",
        "seed_payload",
        "convert_to_cm",
        "is_pulmonary",
        "bc_prefix",
    )
    assert parameters["outlet_mapping_mode"].default == "auto"
    assert parameters["centerline"].default is None
    assert parameters["is_pulmonary"].default is False
    assert parameters["bc_prefix"].default is None


def test_validator_accepts_canonical_mode_and_explicit_mapping(monkeypatch):
    caps = _patch_vtp_info(monkeypatch)

    resolved = validate_cap_to_bc_mapping(
        _config(),
        "/mesh-surfaces",
        outlet_mapping_mode="explicit",
        outlet_mapping={
            "/mesh/rpa_terminal.vtp": "OUT_0",
            "/mesh/lpa_terminal.vtp": "OUT_1",
        },
        bc_prefix="IMPEDANCE",
    )

    assert resolved.pairs == (
        ("/mesh/lpa_terminal.vtp", "OUT_1"),
        ("/mesh/rpa_terminal.vtp", "OUT_0"),
    )
    assert tuple(resolved.cap_order) == tuple(caps)


def test_validator_reuses_supplied_resolved_mapping_identity(monkeypatch):
    caps = _patch_vtp_info(monkeypatch)
    config = _config()
    resolved = resolve_outlet_cap_mapping(
        config,
        caps,
        mode="explicit",
        explicit_mapping={
            "/mesh/lpa_terminal.vtp": "OUT_1",
            "/mesh/rpa_terminal.vtp": "OUT_0",
        },
    )

    validated = validate_cap_to_bc_mapping(
        config,
        "/mesh-surfaces",
        resolved_mapping=resolved,
        bc_prefix="IMPEDANCE",
    )

    assert validated is resolved


def test_validator_rejects_supplied_and_resolved_mapping_conflict(monkeypatch):
    _patch_vtp_info(monkeypatch)
    config = _config()
    resolved = resolve_outlet_cap_mapping(
        config,
        {
            "/mesh/lpa_terminal.vtp": 1.0,
            "/mesh/rpa_terminal.vtp": 4.0,
        },
        mode="serialized_cap_order",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_cap_to_bc_mapping(
            config,
            "/mesh-surfaces",
            outlet_mapping_mode="explicit",
            outlet_mapping={
                "/mesh/lpa_terminal.vtp": "OUT_1",
                "/mesh/rpa_terminal.vtp": "OUT_0",
            },
            resolved_mapping=resolved,
        )


def test_validator_rejects_order_fallback_in_auto_mode(monkeypatch):
    _patch_vtp_info(monkeypatch)

    with pytest.raises(ValueError, match="could not deterministically resolve"):
        validate_cap_to_bc_mapping(_config(), "/mesh-surfaces")


# assign_rcr_bcs scales each side's tuned resistance by side_total_area / cap_area.
RCR_LPA_CAPS = {"/mesh/lpa_1.vtp": 1.0, "/mesh/lpa_2.vtp": 3.0}
RCR_RPA_CAPS = {"/mesh/rpa_1.vtp": 2.0, "/mesh/rpa_2.vtp": 6.0}
RCR_PARAMS = [100.0, 1.0e-5, 200.0, 2.0e-5]  # R_LPA, C_LPA, R_RPA, C_RPA
WEDGE_P_MMHG = 8.0


def _expected_rcr(cap):
    lpa = "lpa" in cap
    caps = RCR_LPA_CAPS if lpa else RCR_RPA_CAPS
    resistance, capacitance = RCR_PARAMS[:2] if lpa else RCR_PARAMS[2:]
    return resistance * sum(caps.values()) / caps[cap], capacitance


def _patch_pulmonary_vtp_info(monkeypatch):
    def fake_vtp_info(_mesh_path, *, convert_to_cm, pulmonary):
        assert pulmonary is True
        return dict(RCR_RPA_CAPS), dict(RCR_LPA_CAPS), {"/mesh/inflow.vtp": 1.0}

    monkeypatch.setattr("svzerodtrees.tune_bcs.assign_bcs.vtp_info", fake_vtp_info)


def _pa_seed(outlet_bc_names, trees=()):
    """INFLOW -> junction -> one vessel per outlet RESISTANCE BC, in the given seed order."""
    return {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [10.0, 12.0, 10.0], "t": [0.0, 0.4, 0.8]}}
        ]
        + [
            {"bc_name": name, "bc_type": "RESISTANCE", "bc_values": {"R": 1000.0, "Pd": 0.0}}
            for name in outlet_bc_names
        ],
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 3,
            "number_of_cardiac_cycles": 1,
            "density": 1.06,
            "viscosity": 0.04,
        },
        "vessels": [
            {
                "boundary_conditions": {"inlet": "INFLOW"} if vessel_id == 0 else {"outlet": outlet_bc_names[vessel_id - 1]},
                "vessel_id": vessel_id,
                "vessel_length": 10.0,
                "vessel_name": f"branch{vessel_id}_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"C": 0.0, "L": 0.0, "R_poiseuille": 1.0, "stenosis_coefficient": 0.0},
            }
            for vessel_id in range(len(outlet_bc_names) + 1)
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [0],
                "outlet_vessels": list(range(1, len(outlet_bc_names) + 1)),
                "areas": [1.0] * len(outlet_bc_names),
            }
        ],
        "trees": list(trees),
    }


# Neither seed pairs vessel k's BC with the k-th cap of cap_info (lpa_1, lpa_2, rpa_1, rpa_2).
CAP_NAMED_SEED = {"rpa_2": "/mesh/rpa_2.vtp", "lpa_1": "/mesh/lpa_1.vtp", "rpa_1": "/mesh/rpa_1.vtp", "lpa_2": "/mesh/lpa_2.vtp"}
RCR_K_SEED = {"RCR_0": "/mesh/lpa_2.vtp", "RCR_1": "/mesh/rpa_2.vtp", "RCR_2": "/mesh/lpa_1.vtp", "RCR_3": "/mesh/rpa_1.vtp"}


def _rcr_k_trees():
    """Tree outlet_mapping metadata (the format construct_impedance_trees writes) pairing RCR_k with its cap."""
    return [
        {
            "name": side.upper(),
            "outlet_mapping": {
                "mode": "shared_by_side",
                "side": side,
                "bc_names": [bc for bc, cap in RCR_K_SEED.items() if side in cap],
                "outlet_names": [cap for cap in RCR_K_SEED.values() if side in cap],
            },
        }
        for side in ("lpa", "rpa")
    ]


@pytest.mark.parametrize(
    ("seed", "trees"),
    [(CAP_NAMED_SEED, ()), (RCR_K_SEED, _rcr_k_trees())],
    ids=["cap_named_bcs", "rcr_k_bcs_with_tree_mapping"],
)
def test_assign_rcr_bcs_serializes_each_vessels_bc_for_its_mapped_cap(monkeypatch, tmp_path, capsys, seed, trees):
    _patch_pulmonary_vtp_info(monkeypatch)
    handler = ConfigHandler(_pa_seed(list(seed), trees))

    assign_rcr_bcs(handler, "/mesh", WEDGE_P_MMHG, RCR_PARAMS)

    for key, bc in handler.bcs.items():
        assert bc.name == key

    # the BC a vessel references after a save/load round trip is the one built for its cap
    reloaded = ConfigHandler(handler.config)
    for vessel in reloaded.vessel_map.values():
        outlet = (vessel.bc or {}).get("outlet")
        if outlet is None:
            continue
        resistance, capacitance = _expected_rcr(seed[outlet])
        bc = reloaded.bcs[outlet]
        assert bc.type == "RCR"
        assert bc.R == pytest.approx(resistance)
        assert bc.C == pytest.approx(capacitance)
        assert bc.values["Pd"] == pytest.approx(WEDGE_P_MMHG * 1333.2)

    # and the 3D coupler pairs it with that cap without the mesh-order fallback
    mesh_caps = ("inflow.vtp", "lpa_1.vtp", "lpa_2.vtp", "rpa_1.vtp", "rpa_2.vtp")
    mesh_complete = SimpleNamespace(
        mesh_surfaces={name: SimpleNamespace(filename=name, path=f"/sim/mesh-surfaces/{name}") for name in mesh_caps}
    )
    capsys.readouterr()
    coupler, _ = reloaded.generate_threed_coupler(str(tmp_path), mesh_complete=mesh_complete)
    assert "WARNING" not in capsys.readouterr().out
    surfaces = {name: Path(str(block.surface)).name for name, block in coupler.coupling_blocks.items()}
    surfaces.pop("inflow")
    assert surfaces == {bc: Path(cap).name for bc, cap in seed.items()}
