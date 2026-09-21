from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from svzerodtrees.tune_bcs.outlet_mapping import (
    ResolvedOutletCapMapping,
    resolve_outlet_cap_mapping,
)


class _BC:
    def __init__(self, name, bc_type="RESISTANCE"):
        self.name = name
        self.type = bc_type


def _config(names, *, metadata=None, graph_sides=None):
    boundary_conditions = [
        {"bc_name": "INFLOW", "bc_type": "FLOW"},
        *[
            {"bc_name": name, "bc_type": "RESISTANCE"}
            for name in names
        ],
    ]
    tree_params = {}
    if metadata is not None:
        tree_params["tree"] = {"outlet_mapping": metadata}
    vessel_map = {}
    for name, side in (graph_sides or {}).items():
        vessel_map[name] = SimpleNamespace(
            bc={"outlet": name},
            label=side,
        )
    return SimpleNamespace(
        _config={"boundary_conditions": boundary_conditions},
        bcs={"INFLOW": _BC("INFLOW", "FLOW"), **{name: _BC(name) for name in names}},
        tree_params=tree_params,
        vessel_map=vessel_map,
    )


def test_auto_metadata_precedes_cap_name_and_preserves_canonical_cap_order():
    config = _config(
        ["OUT_0", "OUT_1"],
        metadata={
            "outlet_names": ["/mesh/lpa_cap.vtp", "/mesh/rpa_cap.vtp"],
            "bc_names": ["OUT_1", "OUT_0"],
        },
    )
    caps = {
        "/mesh/lpa_cap.vtp": 1.0,
        "/mesh/rpa_cap.vtp": 4.0,
    }

    resolved = resolve_outlet_cap_mapping(config, caps)

    assert isinstance(resolved, ResolvedOutletCapMapping)
    assert resolved.strategy == "metadata"
    assert resolved.pairs == (
        ("/mesh/lpa_cap.vtp", "OUT_1"),
        ("/mesh/rpa_cap.vtp", "OUT_0"),
    )
    assert [record.cap_index for record in resolved.records] == [0, 1]
    assert [record.bc_index for record in resolved.records] == [1, 0]
    assert resolved.records[0].raw_diameter == pytest.approx(2.0 / (3.141592653589793**0.5))


def test_cap_name_mode_matches_normalized_stems():
    config = _config(["LPA-cap", "RPA-cap"])
    resolved = resolve_outlet_cap_mapping(
        config,
        {"/mesh/LPA-cap.vtp": 2.0, "/mesh/RPA-cap.vtp": 3.0},
        mode="cap_name",
    )

    assert resolved.strategy == "cap_name"
    assert resolved["/mesh/LPA-cap.vtp"] == "LPA-cap"
    assert resolved["RPA-cap.vtp"] == "RPA-cap"


def test_auto_rejects_serialized_order_for_generic_full_model_names():
    config = _config(["RESISTANCE_0", "RESISTANCE_1"])
    caps = {"/mesh/rpa_dist.vtp": 1.0, "/mesh/lpa_dist.vtp": 1.0}

    with pytest.raises(ValueError, match="serialized_cap_order"):
        resolve_outlet_cap_mapping(config, caps)

    resolved = resolve_outlet_cap_mapping(
        config,
        caps,
        mode="serialized_cap_order",
    )
    assert resolved.strategy == "serialized_cap_order"
    assert resolved.pairs == (
        ("/mesh/rpa_dist.vtp", "RESISTANCE_0"),
        ("/mesh/lpa_dist.vtp", "RESISTANCE_1"),
    )
    assert [record.side for record in resolved.records] == ["rpa", "lpa"]


def test_explicit_mapping_requires_complete_bijection_and_is_order_independent():
    config = _config(["OUT_0", "OUT_1"])
    caps = {"/mesh/lpa_0.vtp": 1.0, "/mesh/rpa_0.vtp": 1.0}
    forward = resolve_outlet_cap_mapping(
        config,
        caps,
        mode="explicit",
        explicit_mapping={"rpa_0": "OUT_0", "lpa_0": "OUT_1"},
    )
    reverse = resolve_outlet_cap_mapping(
        config,
        caps,
        mode="explicit",
        explicit_mapping={"lpa_0": "OUT_1", "rpa_0": "OUT_0"},
    )

    assert forward.pairs == reverse.pairs
    assert forward.serialize() == reverse.serialize()
    with pytest.raises(ValueError, match="duplicate outlet BC"):
        resolve_outlet_cap_mapping(
            config,
            caps,
            mode="explicit",
            explicit_mapping={"lpa_0": "OUT_0", "rpa_0": "OUT_0"},
        )
    with pytest.raises(ValueError, match="exactly one entry per cap"):
        resolve_outlet_cap_mapping(
            config,
            caps,
            mode="explicit",
            explicit_mapping={"lpa_0": "OUT_0"},
        )


def test_ambiguous_name_and_metadata_modes_fail_clearly():
    caps = {"/mesh/lpa_terminal.vtp": 1.0, "/mesh/rpa_terminal.vtp": 1.0}
    ambiguous_names = _config(["LPA-terminal", "LPA_terminal"])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_outlet_cap_mapping(ambiguous_names, caps, mode="cap_name")

    ambiguous_metadata = _config(
        ["OUT_0", "OUT_1"],
        metadata={
            "cap_to_bc": {
                "lpa_terminal": "OUT_0",
                "LPA-terminal.vtp": "OUT_1",
            }
        },
    )
    with pytest.raises(ValueError, match="duplicate cap"):
        resolve_outlet_cap_mapping(ambiguous_metadata, caps, mode="metadata")


def test_cap_side_and_graph_side_are_separate_diagnostic_fields():
    config = _config(
        ["OUT_0", "OUT_1"],
        graph_sides={"OUT_0": "rpa", "OUT_1": "lpa"},
    )
    resolved = resolve_outlet_cap_mapping(
        config,
        {"/mesh/lpa_terminal.vtp": 1.0, "/mesh/rpa_terminal.vtp": 1.0},
        mode="serialized_cap_order",
    )

    assert resolved.records[0].side == "lpa"
    assert resolved.records[0].graph_side == "rpa"
    assert resolved.records[0].graph_side_disagreement is True


def test_payload_is_json_serializable_and_deterministic():
    config = _config(["OUT_0", "OUT_1"])
    resolved = resolve_outlet_cap_mapping(
        config,
        {"/mesh/lpa_terminal.vtp": 1.0, "/mesh/rpa_terminal.vtp": 2.0},
        mode="serialized_cap_order",
    )
    payload = resolved.to_payload()

    assert json.loads(resolved.serialize()) == payload
    assert resolved.to_json() == resolved.to_json()
    assert payload["version"] == 1
    assert payload["strategy"] == "serialized_cap_order"
    assert payload["pairs"][0]["area"] == 1.0
