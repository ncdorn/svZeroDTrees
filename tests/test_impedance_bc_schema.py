from __future__ import annotations

import json
from pathlib import Path


FIXTURE_PATHS = [
    Path(__file__).with_name("example_pa_config.json"),
    Path(__file__).with_name("example_svzerod_3Dcoupling.json"),
]


def _iter_impedance_bcs(payload):
    for bc_config in payload.get("boundary_conditions", []):
        if bc_config.get("bc_type") == "IMPEDANCE":
            yield bc_config


def test_active_impedance_json_fixtures_use_canonical_schema():
    for fixture_path in FIXTURE_PATHS:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        impedance_bcs = list(_iter_impedance_bcs(payload))
        assert impedance_bcs, f"{fixture_path.name} should contain at least one IMPEDANCE BC"
        simparams = payload["simulation_parameters"]
        points_per_cycle = simparams.get("number_of_time_pts_per_cardiac_cycle")
        assert points_per_cycle is not None, (
            f"{fixture_path.name} missing number_of_time_pts_per_cardiac_cycle for IMPEDANCE BCs"
        )
        for bc_config in impedance_bcs:
            values = bc_config["bc_values"]
            assert "z" in values, f"{fixture_path.name}:{bc_config['bc_name']} missing z"
            assert "Pd" in values, f"{fixture_path.name}:{bc_config['bc_name']} missing Pd"
            assert "Z" not in values, f"{fixture_path.name}:{bc_config['bc_name']} contains legacy Z"
            assert "tree" not in values, f"{fixture_path.name}:{bc_config['bc_name']} contains legacy tree"
            assert "t" not in values, f"{fixture_path.name}:{bc_config['bc_name']} contains legacy t"
            assert points_per_cycle == len(values["z"]) + 1, (
                f"{fixture_path.name}:{bc_config['bc_name']} must satisfy "
                "number_of_time_pts_per_cardiac_cycle = len(z) + 1"
            )
        if simparams.get("coupled_simulation"):
            assert simparams.get("number_of_time_pts") == 2, (
                f"{fixture_path.name} coupled IMPEDANCE config must keep number_of_time_pts = 2"
            )
