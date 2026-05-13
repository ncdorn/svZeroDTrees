from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.post_processing.resistance_map import (
    _branch_geometry,
    _load_frames_csv,
    _select_last_cycle_frames,
)


def _write_centerline(path: Path, *, branch_ids, path_values) -> None:
    points = vtk.vtkPoints()
    for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)):
        points.InsertNextPoint(*point)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    lines = vtk.vtkCellArray()
    for start in (0, 2):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, start)
        line.GetPointIds().SetId(1, start + 1)
        lines.InsertNextCell(line)
    poly.SetLines(lines)

    for name, values in {"BranchId": branch_ids, "Path": path_values}.items():
        array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def test_load_frames_csv_resolves_relative_paths(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "result_0001.vtu"
    frame_path.write_text("dummy", encoding="utf-8")

    manifest = tmp_path / "frames.csv"
    manifest.write_text("path,time_s\nframes/result_0001.vtu,0.5\n", encoding="utf-8")

    df = _load_frames_csv(manifest)

    assert df.iloc[0]["path"] == str(frame_path.resolve())
    assert df.iloc[0]["time_s"] == pytest.approx(0.5)


def test_select_last_cycle_frames_uses_strict_cutoff():
    frames = pd.DataFrame(
        [
            {"path": "/tmp/a.vtu", "time_s": 0.0},
            {"path": "/tmp/b.vtu", "time_s": 0.8},
            {"path": "/tmp/c.vtu", "time_s": 1.4},
        ]
    )

    selected = _select_last_cycle_frames(frames, cycle_duration_s=1.0)

    assert selected["path"].tolist() == ["/tmp/b.vtu", "/tmp/c.vtu"]


def test_branch_geometry_warns_on_duplicate_path_values(tmp_path: Path):
    centerline = tmp_path / "centerline.vtp"
    _write_centerline(
        centerline,
        branch_ids=[1, 1, 2, 2],
        path_values=[0.0, 0.0, 0.0, 1.0],
    )

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(centerline))
    reader.Update()
    geometry = _branch_geometry(
        reader.GetOutput(),
        branch_id_array="BranchId",
        path_array="Path",
    )

    assert geometry[1].warnings == ["duplicate Path values within branch ordering"]
    assert geometry[2].length_cm == pytest.approx(1.0)
