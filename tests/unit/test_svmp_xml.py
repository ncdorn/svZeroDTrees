import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from svzerodtrees.simulation.input_builders.svmp_xml import SvMPxml


def _make_mesh_complete():
    return SimpleNamespace(
        volume_mesh="mesh-complete/mesh-complete.mesh.vtu",
        mesh_surfaces={
            "inflow": SimpleNamespace(filename="inflow.vtp", path="mesh-complete/mesh-surfaces/inflow.vtp"),
            "outlet": SimpleNamespace(filename="outlet.vtp", path="mesh-complete/mesh-surfaces/outlet.vtp"),
        },
        walls_combined=SimpleNamespace(path="mesh-complete/walls_combined.vtp"),
    )


def _find(root, tag):
    return root.find(f".//{tag}")


def test_svmp_xml_rigid_wall_defaults(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(_make_mesh_complete(), wall_model="rigid")

    root = ET.parse(xml_path).getroot()
    add_eqn = _find(root, "Add_equation")
    assert add_eqn is not None
    assert add_eqn.get("type") == "fluid"

    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    assert wall_bc.findtext("Type") == "Dir"
    assert wall_bc.findtext("Time_dependence") == "Steady"
    assert wall_bc.findtext("Value") == "0.0"


def test_svmp_xml_deformable_wall_writes_cmm_tags(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(
        _make_mesh_complete(),
        wall_model="deformable",
        elasticity_modulus=123.0,
        poisson_ratio=0.49,
        shell_thickness=0.2,
    )

    root = ET.parse(xml_path).getroot()
    add_eqn = _find(root, "Add_equation")
    assert add_eqn is not None
    assert add_eqn.get("type") == "CMM"
    assert add_eqn.findtext("Min_iterations") == "3"
    assert add_eqn.findtext("Poisson_ratio") == "0.49"
    assert add_eqn.findtext("Shell_thickness") == "0.2"
    assert add_eqn.findtext("Elasticity_modulus") == "123.0"

    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    assert wall_bc.findtext("Type") == "CMM"


def test_svmp_xml_deformable_with_prestress_writes_path(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(
        _make_mesh_complete(),
        wall_model="deformable",
        prestress_file_path="/tmp/prestress.vtu",
    )

    root = ET.parse(xml_path).getroot()
    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    assert wall_bc.findtext("Prestress_file_path") == "/tmp/prestress.vtu"


def test_svmp_xml_deformable_without_prestress_omits_path(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(_make_mesh_complete(), wall_model="deformable")

    root = ET.parse(xml_path).getroot()
    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    assert wall_bc.find("Prestress_file_path") is None


def test_svmp_xml_deformable_writes_uniform_tissue_support(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(
        _make_mesh_complete(),
        wall_model="deformable",
        tissue_support={
            "enabled": True,
            "type": "uniform",
            "stiffness": 1000.0,
            "damping": 10000.0,
            "apply_along_normal_direction": True,
        },
    )

    root = ET.parse(xml_path).getroot()
    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    support = wall_bc.find("Tissue_support")
    assert support is not None
    assert support.findtext("Stiffness") == "1000.0"
    assert support.findtext("Damping") == "10000.0"
    assert support.findtext("Apply_along_normal_direction") == "true"


def test_svmp_xml_deformable_writes_spatial_tissue_support(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(
        _make_mesh_complete(),
        wall_model="deformable",
        tissue_support={
            "enabled": True,
            "type": "spatial",
            "spatial_values_file_path": "robin_values.vtp",
            "apply_along_normal_direction": False,
        },
    )

    root = ET.parse(xml_path).getroot()
    wall_bc = root.find(".//Add_BC[@name='wall']")
    assert wall_bc is not None
    support = wall_bc.find("Tissue_support")
    assert support is not None
    assert support.findtext("Spatial_values_file_path") == "robin_values.vtp"
    assert support.find("Stiffness") is None
    assert support.find("Damping") is None
    assert support.findtext("Apply_along_normal_direction") == "false"


def test_svmp_xml_rigid_rejects_enabled_tissue_support(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))

    with pytest.raises(ValueError, match="tissue_support"):
        writer.write(
            _make_mesh_complete(),
            wall_model="rigid",
            tissue_support={"enabled": True, "stiffness": 1.0, "damping": 1.0},
        )


def test_svmp_xml_rejects_mixed_spatial_tissue_support(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))

    with pytest.raises(ValueError, match="spatial tissue_support forbids"):
        writer.write(
            _make_mesh_complete(),
            wall_model="deformable",
            tissue_support={
                "enabled": True,
                "type": "spatial",
                "stiffness": 1.0,
                "spatial_values_file_path": "robin_values.vtp",
            },
        )


def test_svmp_xml_prestress_mode_writes_shell_cmm_setup(tmp_path):
    xml_path = Path(tmp_path) / "svFSIplus.xml"
    writer = SvMPxml(str(xml_path))
    writer.write(
        _make_mesh_complete(),
        simulation_mode="prestress",
        traction_file_path="/tmp/rigid_wall_mean_traction.vtp",
        wall_model="deformable",
    )

    root = ET.parse(xml_path).getroot()
    add_mesh = root.find(".//Add_mesh")
    assert add_mesh is not None
    assert add_mesh.get("name") == "wall"
    assert add_mesh.findtext("Set_mesh_as_shell") == "true"

    add_eqn = root.find(".//Add_equation")
    assert add_eqn is not None
    assert add_eqn.get("type") == "CMM"
    assert add_eqn.findtext("Prestress") == "true"
    assert add_eqn.findtext("Initialize") == "prestress"

    add_bf = root.find(".//Add_BF")
    assert add_bf is not None
    assert add_bf.get("mesh") == "wall"
    assert add_bf.findtext("Type") == "traction"
    assert add_bf.findtext("Spatial_values_file_path") == "/tmp/rigid_wall_mean_traction.vtp"
