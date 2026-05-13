import pytest
import subprocess
from pathlib import Path

import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees import cli


class RecordingWorkflow:
    seen = []

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        cls.seen.append(cfg)
        return cls(cfg)

    def run(self):
        return {"status": "ok"}


def test_cli_schema_renders_config_template(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "schema"])

    assert cli.main() == 0

    rendered = capsys.readouterr().out
    assert "workflow: pipeline" in rendered
    assert "paths:" in rendered
    assert "bcs:" in rendered


def test_cli_dispatches_real_config_to_pipeline_workflow(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
pipeline:
  run_steady: false
  optimize_bcs: false
  run_threed: false
  adapt: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": RecordingWorkflow})
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)])

    assert cli.main() == 0
    assert RecordingWorkflow.seen[0].workflow == "pipeline"
    assert RecordingWorkflow.seen[0].paths.root == str(tmp_path)


def test_cli_dispatches_real_config_to_construct_trees_workflow(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: construct_trees
paths:
  root: {tmp_path}
  zerod_config: model.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: rcr
  rcr_params: [1.0, 2.0, 3.0, 4.0]
trees:
  d_min: 0.01
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"construct_trees": RecordingWorkflow})
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["svzerodtrees", "construct-trees", str(cfg_path)],
    )

    assert cli.main() == 0
    assert RecordingWorkflow.seen[0].workflow == "construct_trees"


def test_cli_rejects_subcommand_workflow_mismatch(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "tune-bcs", str(cfg_path)])

    with pytest.raises(ValueError, match="does not match subcommand"):
        cli.main()


def test_run_from_config_helper_dispatches_correct_workflow(monkeypatch, tmp_path):
    class DummyWorkflow:
        seen = []

        @classmethod
        def from_config(cls, cfg):
            cls.seen.append(cfg)
            return cls()

        def run(self):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": DummyWorkflow})

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"version: 1\nworkflow: pipeline\npaths:\n  root: {tmp_path}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)])
    assert cli.main() == 0
    assert DummyWorkflow.seen[0].workflow == "pipeline"


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

    for name, values in {"BranchId": [1.0, 1.0, 2.0, 2.0], "Path": [0.0, 1.0, 0.0, 1.0]}.items():
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


def test_cli_postprocess_resistance_map_smoke(monkeypatch, tmp_path):
    svslicer = tmp_path / "svslicer"
    svslicer.write_text("#!/bin/sh\n", encoding="utf-8")
    centerline = tmp_path / "centerlines.vtp"
    _write_centerline(centerline)

    frame1 = tmp_path / "result_0001.vtu"
    frame2 = tmp_path / "result_0002.vtu"
    frame1.write_text("dummy", encoding="utf-8")
    frame2.write_text("dummy", encoding="utf-8")

    manifest = tmp_path / "frames.csv"
    manifest.write_text(
        f"path,time_s\n{frame1.name},0.2\n{frame2.name},0.9\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  analyses:
    - kind: pulmonary_resistance_map
      output: out
      options:
        svslicer_path: {svslicer}
        centerline: {centerline}
        frames_csv: {manifest}
        cycle_duration_s: 1.0
""",
        encoding="utf-8",
    )

    datasets = {
        "result_0001": {"pressure": [100.0, 90.0, 100.0, 80.0], "velocity": [4.0, 4.0, 3.0, 3.0]},
        "result_0002": {"pressure": [110.0, 90.0, 110.0, 70.0], "velocity": [5.0, 5.0, 2.0, 2.0]},
    }

    def fake_run(cmd, capture_output, text, check):
        payload = datasets[Path(cmd[1]).stem]
        _write_mapped_centerline(Path(cmd[3]), pressure=payload["pressure"], velocity=payload["velocity"])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("svzerodtrees.post_processing.resistance_map.subprocess.run", fake_run)
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "postprocess", str(cfg_path)])

    assert cli.main() == 0
    assert (tmp_path / "out" / "branch_resistance_summary.csv").exists()
    assert (tmp_path / "out" / "ranked_stent_candidates.csv").exists()
    assert (tmp_path / "out" / "resistance_map_mean.vtp").exists()
