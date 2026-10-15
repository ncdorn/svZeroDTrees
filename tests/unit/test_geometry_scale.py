import json
import math

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svzerodtrees.simulation.geometry_scale import (
    SCALED_GEOMETRY_MANIFEST,
    scale_centerline,
    scale_mesh_complete,
    scale_zerod_payload,
    stage_scaled_geometry,
)

S = 1.5


def _polydata(points, arrays=None):
    data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
    data.SetPoints(vtk_points)
    for name, values in (arrays or {}).items():
        array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
        array.SetName(name)
        data.GetPointData().AddArray(array)
    return data


def _write(data, path):
    writer = vtk.vtkXMLPolyDataWriter() if str(path).endswith(".vtp") else vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    assert writer.Write() == 1


def _read(path):
    reader = vtk.vtkXMLPolyDataReader() if str(path).endswith(".vtp") else vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def test_scale_zerod_payload_follows_poiseuille_rules():
    payload = {
        "vessels": [
            {
                "vessel_length": 2.0,
                "zero_d_element_values": {"R_poiseuille": 8.0, "L": 3.0, "C": 2.0, "stenosis_coefficient": 0.5},
                "geometric_params": {"inlet_area": 1.0, "outlet_area": 2.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            }
        ],
        "junctions": [{"areas": [1.0, 0.5], "lengths": [1.0, 2.0], "tangents": [[0, 0, 1]]}],
        "boundary_conditions": [{"bc_name": "INFLOW", "bc_values": {"Q": [1.0, 2.0]}}],
    }
    scaled = scale_zerod_payload(payload, S)
    vessel = scaled["vessels"][0]
    values = vessel["zero_d_element_values"]
    assert vessel["vessel_length"] == pytest.approx(2.0 * S)
    assert values["R_poiseuille"] == pytest.approx(8.0 / S**3)
    assert values["L"] == pytest.approx(3.0 / S)
    assert values["C"] == pytest.approx(2.0 * S**3)
    infinite = scale_zerod_payload({"vessels": [{"zero_d_element_values": {"C": math.inf}}]}, S)
    assert math.isinf(infinite["vessels"][0]["zero_d_element_values"]["C"])
    assert values["stenosis_coefficient"] == pytest.approx(0.5 / S**4)
    assert vessel["geometric_params"] == pytest.approx({"inlet_area": S**2, "outlet_area": 2 * S**2})
    assert scaled["junctions"][0]["areas"] == pytest.approx([S**2, 0.5 * S**2])
    assert scaled["junctions"][0]["lengths"] == pytest.approx([S, 2 * S])
    assert scaled["junctions"][0]["tangents"] == [[0, 0, 1]]
    assert scaled["boundary_conditions"] == payload["boundary_conditions"]
    assert payload["vessels"][0]["vessel_length"] == 2.0  # input untouched


def test_scale_centerline_scales_lengths_and_areas(tmp_path):
    source = tmp_path / "centerlines.vtp"
    _write(
        _polydata(
            [[0, 0, 0], [0, 0, 2]],
            {"Path": [0, 2], "MaximumInscribedSphereRadius": [1, 1], "CenterlineSectionArea": [3, 3], "BranchId": [0, 0]},
        ),
        source,
    )
    scale_centerline(source, tmp_path / "scaled.vtp", S)
    data = _read(tmp_path / "scaled.vtp")
    arrays = data.GetPointData()
    np.testing.assert_allclose(vtk_to_numpy(data.GetPoints().GetData())[1], [0, 0, 2 * S])
    np.testing.assert_allclose(vtk_to_numpy(arrays.GetArray("Path")), [0, 2 * S])
    np.testing.assert_allclose(vtk_to_numpy(arrays.GetArray("MaximumInscribedSphereRadius")), [S, S])
    np.testing.assert_allclose(vtk_to_numpy(arrays.GetArray("CenterlineSectionArea")), [3 * S**2] * 2)
    np.testing.assert_allclose(vtk_to_numpy(arrays.GetArray("BranchId")), [0, 0])


def _mesh_complete(root):
    (root / "mesh-surfaces").mkdir(parents=True)
    _write(_polydata([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), root / "mesh-surfaces" / "cap.vtp")
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    for point in ([0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]):
        points.InsertNextPoint(*point)
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_TETRA, 4, [0, 1, 2, 3])
    _write(grid, root / "mesh-complete.mesh.vtu")
    (root / "notes.txt").write_text("kept", encoding="utf-8")


def test_scale_mesh_complete_scales_every_vtk_file(tmp_path):
    _mesh_complete(tmp_path / "src")
    assert scale_mesh_complete(tmp_path / "src", tmp_path / "out", S) == 2
    cap = vtk_to_numpy(_read(tmp_path / "out" / "mesh-surfaces" / "cap.vtp").GetPoints().GetData())
    mesh = vtk_to_numpy(_read(tmp_path / "out" / "mesh-complete.mesh.vtu").GetPoints().GetData())
    np.testing.assert_allclose(cap[0], [S, 0, 0])
    np.testing.assert_allclose(mesh[1], [2 * S, 0, 0])
    assert (tmp_path / "out" / "notes.txt").read_text(encoding="utf-8") == "kept"


def test_stage_scaled_geometry_reuses_matching_copy_and_refuses_others(tmp_path):
    _mesh_complete(tmp_path / "patient" / "preop-mesh-complete")
    seed = tmp_path / "patient" / "baseline_0d.json"
    seed.write_text(json.dumps({"vessels": [{"vessel_length": 1.0}]}), encoding="utf-8")
    root = tmp_path / "staged" / "TST" / "scale_1.5"
    sources = {"preop_mesh_complete": str(tmp_path / "patient" / "preop-mesh-complete"), "zerod_config": str(seed)}
    targets = {"preop_mesh_complete": str(root / "preop-mesh-complete"), "zerod_config": str(root / "baseline_0d.json")}

    first = stage_scaled_geometry(root, S, sources=sources, targets=targets)
    second = stage_scaled_geometry(root, S, sources=sources, targets=targets)

    assert first["reused"] is False and second["reused"] is True
    assert json.loads((root / "baseline_0d.json").read_text())["vessels"][0]["vessel_length"] == pytest.approx(S)
    assert (root / SCALED_GEOMETRY_MANIFEST).exists()
    with pytest.raises(ValueError, match="other inputs or another scale"):
        stage_scaled_geometry(root, 2.0, sources=sources, targets=targets)
    with pytest.raises(ValueError):
        stage_scaled_geometry(
            root, S, sources={"zerod_config": str(seed)}, targets={"zerod_config": str(tmp_path / "elsewhere.json")}
        )
