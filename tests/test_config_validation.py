import json
from pathlib import Path

import pytest


CONFIG_FIXTURES = [
    "example_pa_config.json",
    "example_svzerod_3Dcoupling.json",
]


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _validate_config_connectivity(config):
    errors = []

    vessels = config.get("vessels", [])
    junctions = config.get("junctions", [])
    bcs = config.get("boundary_conditions", [])
    coupling_blocks = config.get("external_solver_coupling_blocks", [])

    simparams = config.get("simulation_parameters", {})
    is_threed = bool(coupling_blocks) or bool(simparams.get("coupled_simulation"))

    vessel_ids = set()
    vessel_names = set()
    for vessel in vessels:
        vessel_ids.add(vessel.get("vessel_id"))
        if vessel.get("vessel_name"):
            vessel_names.add(vessel.get("vessel_name"))

    bc_names = {bc.get("bc_name") for bc in bcs}

    junction_vessel_ids = set()
    for junction in junctions:
        for field in ("inlet_vessels", "outlet_vessels"):
            for vessel_id in junction.get(field, []):
                if vessel_id not in vessel_ids:
                    errors.append(
                        f"Junction {junction.get('junction_name')} references unknown vessel id {vessel_id}."
                    )
                else:
                    junction_vessel_ids.add(vessel_id)

    used_bc_names = set()
    for vessel in vessels:
        for bc_name in vessel.get("boundary_conditions", {}).values():
            used_bc_names.add(bc_name)
            if bc_name not in bc_names:
                errors.append(
                    f"Vessel {vessel.get('vessel_name')} references missing boundary condition {bc_name}."
                )

    coupling_connected = set()
    for block in coupling_blocks:
        connected_block = block.get("connected_block")
        if not connected_block:
            errors.append("Coupling block missing connected_block.")
            continue
        coupling_connected.add(connected_block)
        if connected_block not in vessel_names and connected_block not in bc_names:
            errors.append(
                f"Coupling block {block.get('name')} connects to unknown block {connected_block}."
            )

    for vessel in vessels:
        vessel_id = vessel.get("vessel_id")
        vessel_name = vessel.get("vessel_name")
        connected = vessel_id in junction_vessel_ids
        if not is_threed:
            bc_map = vessel.get("boundary_conditions", {})
            inlet_bc = bc_map.get("inlet")
            outlet_bc = bc_map.get("outlet")
            if inlet_bc and outlet_bc and inlet_bc in bc_names and outlet_bc in bc_names:
                connected = True
        if is_threed and vessel_name in coupling_connected:
            connected = True
        if not connected:
            errors.append(
                f"Vessel {vessel_name} (id {vessel_id}) is not connected to a junction or coupling block."
            )

    for bc in bcs:
        bc_name = bc.get("bc_name")
        connected = bc_name in used_bc_names
        if is_threed and bc_name in coupling_connected:
            connected = True
        if not connected:
            errors.append(
                f"Boundary condition {bc_name} is not referenced by any vessel or coupling block."
            )

    return errors


@pytest.mark.parametrize("config_path", CONFIG_FIXTURES)
def test_config_json_is_valid_and_connected(config_path):
    path = Path(config_path)
    config = _load_json(path)
    errors = _validate_config_connectivity(config)
    assert not errors, "\n".join(errors)
