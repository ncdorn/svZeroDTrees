from __future__ import annotations

import json
import threading
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.post_processing import compute_pulmonary_resistance_map
from svzerodtrees.post_processing.resistance_map import (
    _compute_pulmonary_resistance_map_for_selected_frames,
)


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


def _write_mapped_centerline_with_named_arrays(
    path: Path,
    *,
    pressure_name: str,
    velocity_name: str,
    pressure,
    velocity,
) -> None:
    _write_centerline(path)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())

    for name, values in {
        pressure_name: pressure,
        velocity_name: velocity,
    }.items():
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
    for name in ("result_0001.vtu", "result_0002.vtu", "result_0003.vtu", "result_0004.vtu"):
        frame = tmp_path / name
        frame.write_text("dummy", encoding="utf-8")
        frame_paths.append(frame)

    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        "\n".join(
            [
                "path,time_s",
                f"{frame_paths[0].name},0.0",
                f"{frame_paths[1].name},0.4",
                f"{frame_paths[2].name},0.8",
                f"{frame_paths[3].name},1.2",
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
        cycle_duration_s=0.8,
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
    assert metadata["selected_frames"][1]["path"].endswith("result_0003_centerline.vtp")
    assert len(metadata["selected_frames"]) == 2
    assert metadata["selection_window_start_s"] == pytest.approx(0.4)
    assert metadata["selection_window_end_s"] == pytest.approx(1.2)
    assert metadata["selection_policy"] == "all_frames_last_full_cycle"
    assert metadata["available_frame_count"] == 2
    assert metadata["selected_frame_count"] == 2
    assert metadata["max_frames"] is None
    assert metadata["workers_requested"] is None
    assert metadata["workers_used"] == 1
    assert not (tmp_path / "out" / "intermediate_centerlines").exists()
    assert Path(result["resistance_map"]).exists()


def test_compute_pulmonary_resistance_map_for_selected_frames_writes_systolic_outputs(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    selected_frame = tmp_path / "result_0002.vtu"
    selected_frame.write_text("dummy", encoding="utf-8")
    selected_frames = pd.DataFrame(
        [
            {
                "timestep_id": 2,
                "path": str(selected_frame),
                "time_s": 0.4,
            }
        ]
    )

    def fake_run(cmd, capture_output, text, check):
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[110.0, 90.0, 110.0, 75.0],
            velocity=[5.0, 5.0, 2.5, 2.5],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    result = _compute_pulmonary_resistance_map_for_selected_frames(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        selected_frames=selected_frames,
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=0.8,
        available_frame_count=2,
        selection_window_start_s=0.4,
        selection_window_end_s=1.2,
        selection_tolerance_s=1e-9,
        selection_policy="max_mpa_pressure_last_cycle",
        metric_suffix="systolic",
        metadata_extra={
            "selection_mode": "max_mpa_pressure_last_cycle",
            "selected_timestep_id": 2,
            "selected_time_s": 0.4,
            "selected_pressure_mmhg": 25.0,
            "selected_frame_path": str(selected_frame),
            "tie_break_policy": "earliest_timestep_id",
        },
    )

    summary = pd.read_csv(result["summary_csv"])
    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(result["resistance_map"])
    reader.Update()
    poly = reader.GetOutput()

    branch1 = summary.loc[summary["branch_id"] == 1].iloc[0]
    assert branch1["pressure_drop_systolic"] == pytest.approx(20.0)
    assert branch1["flow_systolic"] == pytest.approx(5.0)
    assert branch1["resistance_systolic"] == pytest.approx(4.0)
    assert metadata["selection_policy"] == "max_mpa_pressure_last_cycle"
    assert metadata["selection_mode"] == "max_mpa_pressure_last_cycle"
    assert metadata["selected_timestep_id"] == 2
    assert metadata["selected_frames"][0]["timestep_id"] == 2
    assert metadata["selected_frames"][0]["source_frame_path"] == str(selected_frame)
    assert poly.GetPointData().GetArray("branch_resistance_systolic") is not None
    assert Path(result["resistance_map"]).name == "resistance_map_systolic.vtp"


def test_compute_pulmonary_resistance_map_selects_all_last_cycle_frames_in_exact_mode(
    monkeypatch, tmp_path: Path
):
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
    )

    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    assert seen_inputs == [
        "result_0001.vtu",
        "result_0002.vtu",
        "result_0003.vtu",
        "result_0004.vtu",
        "result_0005.vtu",
    ]
    assert metadata["selection_window_start_s"] == pytest.approx(0.0)
    assert metadata["selection_window_end_s"] == pytest.approx(1.0)
    assert metadata["selection_policy"] == "all_frames_last_full_cycle"
    assert metadata["available_frame_count"] == 5
    assert metadata["selected_frame_count"] == 5
    assert metadata["max_frames"] is None
    assert metadata["workers_requested"] is None
    assert metadata["workers_used"] == 1
    assert len(metadata["selected_frames"]) == 5


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
    assert metadata["selection_window_start_s"] == pytest.approx(0.0)
    assert metadata["selection_window_end_s"] == pytest.approx(1.0)
    assert metadata["selection_policy"] == "all_frames_last_full_cycle"
    assert metadata["max_frames"] == 3
    assert metadata["workers_requested"] is None
    assert metadata["workers_used"] == 1
    assert len(metadata["selected_frames"]) == 3
    assert len(seen_inputs) == 3


def test_compute_pulmonary_resistance_map_accepts_capitalized_hemodynamic_arrays(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    frame = tmp_path / "result_0001.vtu"
    frame.write_text("dummy", encoding="utf-8")
    excluded_frame = tmp_path / "result_0002.vtu"
    excluded_frame.write_text("dummy", encoding="utf-8")
    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        "path,time_s\nresult_0001.vtu,0.0\nresult_0002.vtu,0.5\n",
        encoding="utf-8",
    )

    def fake_run(cmd, capture_output, text, check):
        _write_mapped_centerline_with_named_arrays(
            Path(cmd[3]),
            pressure_name="Pressure",
            velocity_name="Velocity",
            pressure=[100.0, 80.0, 100.0, 70.0],
            velocity=[4.0, 4.0, 2.0, 2.0],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=1.0,
        max_frames=1,
    )

    summary = pd.read_csv(result["summary_csv"])
    assert not summary.empty
    assert Path(result["resistance_map"]).exists()


def test_compute_pulmonary_resistance_map_parallel_workers_preserve_deterministic_outputs(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    frame_paths = []
    for idx in range(4):
        frame = tmp_path / f"result_{idx + 1:04d}.vtu"
        frame.write_text("dummy", encoding="utf-8")
        frame_paths.append(frame)

    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        "\n".join(
            [
                "path,time_s",
                "result_0001.vtu,0.0",
                "result_0002.vtu,0.2",
                "result_0003.vtu,0.4",
                "result_0004.vtu,0.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    start_gate = threading.Event()
    active_lock = threading.Lock()
    active_runs = 0
    max_active_runs = 0

    def fake_run(cmd, capture_output, text, check):
        nonlocal active_runs, max_active_runs
        with active_lock:
            active_runs += 1
            max_active_runs = max(max_active_runs, active_runs)
            if active_runs >= 2:
                start_gate.set()
        start_gate.wait(timeout=1.0)
        time.sleep(0.01)
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[100.0, 80.0, 100.0, 70.0],
            velocity=[4.0, 4.0, 2.0, 2.0],
        )
        with active_lock:
            active_runs -= 1
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=0.6,
        workers=2,
    )

    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    assert max_active_runs >= 2
    assert metadata["workers_requested"] == 2
    assert metadata["workers_used"] == 2
    assert [Path(entry["path"]).name for entry in metadata["selected_frames"]] == [
        "0000_result_0001_centerline.vtp",
        "0001_result_0002_centerline.vtp",
        "0002_result_0003_centerline.vtp",
    ]


def test_compute_pulmonary_resistance_map_auto_workers_use_slurm_cpus(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)
    frame = tmp_path / "result_0001.vtu"
    frame.write_text("dummy", encoding="utf-8")
    manifest = tmp_path / "frames.csv"
    manifest.write_text("path,time_s\nresult_0001.vtu,0.0\nresult_0002.vtu,1.0\n", encoding="utf-8")
    excluded_frame = tmp_path / "result_0002.vtu"
    excluded_frame.write_text("dummy", encoding="utf-8")

    def fake_run(cmd, capture_output, text, check):
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[100.0, 80.0, 100.0, 70.0],
            velocity=[4.0, 4.0, 2.0, 2.0],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=1.0,
        max_frames=1,
        workers="auto",
    )

    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    assert metadata["workers_requested"] == "auto"
    assert metadata["workers_used"] == 4


def test_compute_pulmonary_resistance_map_auto_workers_fall_back_to_local_cpu_count(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)
    frame = tmp_path / "result_0001.vtu"
    frame.write_text("dummy", encoding="utf-8")
    excluded_frame = tmp_path / "result_0002.vtu"
    excluded_frame.write_text("dummy", encoding="utf-8")
    manifest = tmp_path / "frames.csv"
    manifest.write_text("path,time_s\nresult_0001.vtu,0.0\nresult_0002.vtu,1.0\n", encoding="utf-8")

    def fake_run(cmd, capture_output, text, check):
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[100.0, 80.0, 100.0, 70.0],
            velocity=[4.0, 4.0, 2.0, 2.0],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.os.cpu_count", lambda: 3)

    result = compute_pulmonary_resistance_map(
        svslicer_path=str(svslicer),
        centerline=str(centerline),
        frames_csv=str(manifest),
        output_dir=str(tmp_path / "out"),
        cycle_duration_s=1.0,
        max_frames=1,
        workers="auto",
    )

    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))

    assert metadata["workers_requested"] == "auto"
    assert metadata["workers_used"] == 3


def test_compute_pulmonary_resistance_map_worker_failure_raises_cleanly(
    monkeypatch, tmp_path: Path
):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)
    frame_paths = []
    for idx in range(3):
        frame = tmp_path / f"result_{idx + 1:04d}.vtu"
        frame.write_text("dummy", encoding="utf-8")
        frame_paths.append(frame)
    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        "path,time_s\nresult_0001.vtu,0.0\nresult_0002.vtu,0.5\nresult_0003.vtu,1.0\n",
        encoding="utf-8",
    )

    def fake_run(cmd, capture_output, text, check):
        if Path(cmd[1]).name == "result_0002.vtu":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        _write_mapped_centerline(
            Path(cmd[3]),
            pressure=[100.0, 80.0, 100.0, 70.0],
            velocity=[4.0, 4.0, 2.0, 2.0],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="svSlicer failed"):
        compute_pulmonary_resistance_map(
            svslicer_path=str(svslicer),
            centerline=str(centerline),
            frames_csv=str(manifest),
            output_dir=str(tmp_path / "out"),
            cycle_duration_s=1.0,
            workers=2,
        )

    assert not (tmp_path / "out" / "resistance_map_metadata.json").exists()
