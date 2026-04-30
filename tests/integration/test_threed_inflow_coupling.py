from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from svzerodtrees.io.config_handler import ConfigHandler
from svzerodtrees.simulation.input_builders.svmp_xml import SvMPxml


def _sample_config_payload() -> dict[str, object]:
    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {
                    "Q": [10.0, 12.0, 10.0],
                    "t": [0.0, 0.4, 0.8],
                },
            },
            {
                "bc_name": "OUTLET",
                "bc_type": "IMPEDANCE",
                "bc_values": {
                    "z": [1.0, 0.5],
                    "Pd": 8.0,
                },
            },
        ],
        "simulation_parameters": {
            "number_of_time_pts_per_cardiac_cycle": 3,
            "number_of_cardiac_cycles": 1,
            "density": 1.06,
            "viscosity": 0.04,
        },
        "vessels": [
            {
                "boundary_conditions": {
                    "inlet": "INFLOW",
                    "outlet": "OUTLET",
                },
                "vessel_id": 0,
                "vessel_length": 10.0,
                "vessel_name": "branch0_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {
                    "C": 0.0,
                    "L": 0.0,
                    "R_poiseuille": 1.0,
                    "stenosis_coefficient": 0.0,
                },
            }
        ],
        "junctions": [],
        "external_solver_coupling_blocks": [],
        "trees": [],
    }


def _mesh_complete():
    return SimpleNamespace(
        volume_mesh="mesh-complete/mesh-complete.mesh.vtu",
        mesh_surfaces={
            "inflow": SimpleNamespace(filename="inflow.vtp", path="mesh-complete/mesh-surfaces/inflow.vtp"),
            "outlet": SimpleNamespace(filename="outlet.vtp", path="mesh-complete/mesh-surfaces/outlet.vtp"),
        },
        walls_combined=SimpleNamespace(path="mesh-complete/walls_combined.vtp"),
    )


def test_generate_threed_coupler_dirichlet_drops_inflow_bc(tmp_path: Path):
    handler = ConfigHandler(_sample_config_payload(), is_pulmonary=True)
    mesh_complete = _mesh_complete()

    coupler, coupling_blocks = handler.generate_threed_coupler(
        str(tmp_path),
        inflow_from_0d=False,
        mesh_complete=mesh_complete,
    )

    payload = json.loads((tmp_path / "svzerod_3Dcoupling.json").read_text(encoding="utf-8"))
    bc_names = {bc["bc_name"] for bc in payload["boundary_conditions"]}

    assert "INFLOW" not in bc_names
    assert "OUTLET" in bc_names
    assert all("inflow" not in name.lower() for name in coupling_blocks)
    assert all("inflow" not in block.name.lower() for block in coupler.coupling_blocks.values())
    assert "INFLOW" in handler.bcs

    handler.generate_inflow_file(str(tmp_path), period=0.8, n_tsteps=3)
    assert (tmp_path / "inflow.flow").exists()


def test_generate_inflow_file_writes_reference_style_header(tmp_path: Path):
    handler = ConfigHandler(_sample_config_payload(), is_pulmonary=True)

    period = handler.generate_inflow_file(str(tmp_path), period=0.8, n_tsteps=2000)

    flow_path = tmp_path / "inflow.flow"
    lines = flow_path.read_text(encoding="utf-8").splitlines()

    assert period == 0.8
    assert lines[0] == "2000 16"
    assert len(lines) == 2001
    assert lines[1].startswith("0.0 -10.0")


def test_svmp_xml_dirichlet_inflow_writes_prescribed_inlet(tmp_path: Path):
    xml_path = tmp_path / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    threed_coupler = SimpleNamespace(
        coupling_blocks={
            "OUTLET": SimpleNamespace(surface="mesh-complete/mesh-surfaces/outlet.vtp", name="OUTLET")
        }
    )

    writer.write(
        _mesh_complete(),
        wall_model="rigid",
        threed_coupler=threed_coupler,
        inflow_boundary_condition="dirichlet",
        inflow_file_path=str(tmp_path / "inflow.flow"),
    )

    root = ET.parse(xml_path).getroot()
    inflow_bc = root.find(".//Add_BC[@name='inflow']")
    outlet_bc = root.find(".//Add_BC[@name='outlet']")

    assert inflow_bc is not None
    assert inflow_bc.findtext("Type") == "Dir"
    assert inflow_bc.findtext("Time_dependence") == "Unsteady"
    assert inflow_bc.findtext("Temporal_values_file_path") == "inflow.flow"
    assert inflow_bc.findtext("Profile") == "Parabolic"
    assert inflow_bc.findtext("Impose_flux") == "true"

    assert outlet_bc is not None
    assert outlet_bc.findtext("Type") == "Neu"
    assert outlet_bc.findtext("Time_dependence") == "Coupled"
    assert outlet_bc.findtext("svZeroDSolver_block") == "OUTLET"
