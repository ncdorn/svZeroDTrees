from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from svzerodtrees.tune_bcs.assign_bcs import validate_cap_to_bc_mapping
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
        "convert_to_cm",
        "is_pulmonary",
        "bc_prefix",
    )
    assert parameters["outlet_mapping_mode"].default == "auto"
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
