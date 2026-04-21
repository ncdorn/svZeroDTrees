from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath))
        if "unit" in path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in path.parts:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).with_name("fixtures")


@pytest.fixture
def minimal_svzerod_payload() -> dict[str, object]:
    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {"Q": [1.0, 1.0], "t": [0.0, 1.0]},
            },
            {
                "bc_name": "OUT",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 100.0, "Pd": 12.0},
            },
        ],
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 2,
            "number_of_cardiac_cycles": 1,
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {
                    "R_poiseuille": 1.0,
                    "C": 0.0,
                    "L": 0.0,
                    "stenosis_coefficient": 0.0,
                },
                "boundary_conditions": {"inlet": "INFLOW", "outlet": "OUT"},
            }
        ],
        "junctions": [],
    }


@pytest.fixture
def write_json(tmp_path):
    def _write(name: str, payload: dict[str, object]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def mesh_complete_stub():
    return SimpleNamespace(
        volume_mesh="mesh-complete/mesh-complete.mesh.vtu",
        mesh_surfaces={
            "inflow": SimpleNamespace(
                filename="inflow.vtp",
                path="mesh-complete/mesh-surfaces/inflow.vtp",
            ),
            "outlet": SimpleNamespace(
                filename="outlet.vtp",
                path="mesh-complete/mesh-surfaces/outlet.vtp",
            ),
        },
        walls_combined=SimpleNamespace(path="mesh-complete/walls_combined.vtp"),
    )
