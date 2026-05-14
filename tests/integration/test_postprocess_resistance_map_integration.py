from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.post_processing import compute_pulmonary_resistance_map


def _write_centerline(path: Path) -> None:
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

    arrays = {
        "BranchId": [1.0, 1.0, 2.0, 2.0],
        "Path": [0.0, 1.0, 0.0, 1.0],
    }
    for name, values in arrays.items():
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def _write_mapped_centerline(path: Path, *, pressure, velocity) -> None:
    _write_centerline(path)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())

    for name, values in {"pressure": pressure, "velocity": velocity}.items():
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def test_compute_pulmonary_resistance_map_with_mocked_svslicer(monkeypatch, tmp_path: Path):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    frame_paths = []
    for name in ("result_0001.vtu", "result_0002.vtu", "result_0003.vtu"):
        frame = tmp_path / name
        frame.write_text("dummy", encoding="utf-8")
        frame_paths.append(frame)

    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        "\n".join(
            [
                "path,time_s",
                f"{frame_paths[0].name},0.0",
                f"{frame_paths[1].name},0.8",
                f"{frame_paths[2].name},1.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    datasets = {
        "result_0002": {
            "pressure": [100.0, 80.0, 100.0, 70.0],
            "velocity": [4.0, 4.0, 2.0, 2.0],
        },
        "result_0003": {
            "pressure": [110.0, 90.0, 110.0, 75.0],
            "velocity": [5.0, 5.0, 2.5, 2.5],
        },
    }

    def fake_run(cmd, capture_output, text, check):
        output = Path(cmd[3])
        key = Path(cmd[1]).stem
        payload = datasets.get(key)
        if payload is None:
            payload = {"pressure": [0.0, 0.0, 0.0, 0.0], "velocity": [1.0, 1.0, 1.0, 1.0]}
        _write_mapped_centerline(
            output,
            pressure=payload["pressure"],
            velocity=payload["velocity"],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=1.0,
    )

    summary = pd.read_csv(result["summary_csv"])
    ranked = pd.read_csv(result["ranked_csv"])
    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    branch1 = summary.loc[summary["branch_id"] == 1].iloc[0]
    branch2 = summary.loc[summary["branch_id"] == 2].iloc[0]

    assert branch1["pressure_drop_mean"] == pytest.approx(20.0)
    assert branch1["flow_mean"] == pytest.approx(4.5)
    assert branch1["resistance_mean"] == pytest.approx(20.0 / 4.5)
    assert branch2["pressure_drop_mean"] == pytest.approx(32.5)
    assert branch2["flow_mean"] == pytest.approx(2.25)
    assert branch2["rank"] == pytest.approx(1.0)
    assert ranked.iloc[0]["branch_id"] == pytest.approx(2.0)
    assert metadata["selected_frames"][0]["path"].endswith("result_0002_centerline.vtp")
    assert len(metadata["selected_frames"]) == 2
    assert not (tmp_path / "out" / "intermediate_centerlines").exists()
    assert Path(result["resistance_map"]).exists()


def test_compute_pulmonary_resistance_map_subsamples_last_cycle_frames(monkeypatch, tmp_path: Path):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    frame_paths = []
    for idx in range(6):
        frame = tmp_path / f"result_{idx + 1:04d}.vtu"
        frame.write_text("dummy", encoding="utf-8")
        frame_paths.append(frame)

    manifest_lines = ["path,time_s"]
    for idx, frame in enumerate(frame_paths):
        manifest_lines.append(f"{frame.name},{0.2 * idx:.1f}")
    manifest = tmp_path / "frames.csv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    seen_inputs: list[str] = []

    def fake_run(cmd, capture_output, text, check):
        seen_inputs.append(Path(cmd[1]).name)
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[100.0, 90.0, 100.0, 80.0],
            velocity=[4.0, 4.0, 3.0, 3.0],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=1.0,
        max_frames=3,
    )

    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    assert metadata["available_frame_count"] == 5
    assert metadata["selected_frame_count"] == 3
    assert metadata["max_frames"] == 3
    assert len(metadata["selected_frames"]) == 3
    assert len(seen_inputs) == 3
