import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svzerodtrees.post_processing.centerline_resample import densify_short_branches


def _write_centerline(path):
    # Branch 0 (MPA): two points 0.3 apart; branch 1: four points, joined to branch 0.
    points = np.array(
        [[0, 0, 0], [0, 0, 0.3], [0, 0, 0.4], [0, 0, 0.5], [0, 0, 0.6], [0, 0, 0.7]],
        dtype=float,
    )
    branch = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    path_values = points[:, 2].copy()
    normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    normals[1] = [0.0, 1.0, 0.0]
    node_ids = np.arange(6, dtype=np.int32)
    bifurcation = np.array([-1, 1, 1, -1, -1, -1], dtype=np.int64)

    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    polydata.SetPoints(vtk_points)
    lines = vtk.vtkCellArray()
    for a in range(5):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(a)
        lines.InsertCellPoint(a + 1)
    polydata.SetLines(lines)
    for name, values, vtk_type in [
        ("BranchId", branch, vtk.VTK_LONG_LONG),
        ("Path", path_values, vtk.VTK_DOUBLE),
        ("CenterlineSectionNormal", normals, vtk.VTK_DOUBLE),
        ("GlobalNodeId", node_ids, vtk.VTK_INT),
        ("BifurcationId", bifurcation, vtk.VTK_LONG_LONG),
    ]:
        array = numpy_to_vtk(values, deep=True, array_type=vtk_type)
        array.SetName(name)
        polydata.GetPointData().AddArray(array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def _read(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def test_densify_short_branches_subdivides_only_short_branches(tmp_path):
    source, output = tmp_path / "centerlines.vtp", tmp_path / "resampled.vtp"
    _write_centerline(source)

    summary = densify_short_branches(source, output, min_points_per_branch=4)

    assert summary["points_added"] == 2
    assert summary["branches"] == {"0": {"before": 2, "after": 4}}
    result = _read(output)
    data = result.GetPointData()
    branch = vtk_to_numpy(data.GetArray("BranchId"))
    path_values = vtk_to_numpy(data.GetArray("Path"))
    assert result.GetNumberOfPoints() == 8
    assert np.count_nonzero(branch == 0) == 4
    # Original points keep their ids; new points sit evenly inside the MPA.
    np.testing.assert_allclose(path_values[:6], [0, 0.3, 0.4, 0.5, 0.6, 0.7])
    np.testing.assert_allclose(path_values[6:], [0.1, 0.2])
    np.testing.assert_allclose(vtk_to_numpy(result.GetPoints().GetData())[6:, 2], [0.1, 0.2])
    normals = vtk_to_numpy(data.GetArray("CenterlineSectionNormal"))[6:]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert list(vtk_to_numpy(data.GetArray("GlobalNodeId"))[6:]) == [6, 7]
    # Integer flags come from the nearer endpoint (0.1 -> point 0, 0.2 -> point 1).
    assert list(vtk_to_numpy(data.GetArray("BifurcationId"))[6:]) == [-1, 1]
    assert result.GetNumberOfLines() == 7


def test_densify_short_branches_leaves_long_enough_centerlines_alone(tmp_path):
    source, output = tmp_path / "centerlines.vtp", tmp_path / "resampled.vtp"
    _write_centerline(source)

    summary = densify_short_branches(source, output, min_points_per_branch=2)

    assert summary["points_added"] == 0
    assert _read(output).GetNumberOfPoints() == 6
    with pytest.raises(ValueError, match="at least 2"):
        densify_short_branches(source, output, min_points_per_branch=1)
