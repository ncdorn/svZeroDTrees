import json
import pytest
import subprocess
import sys
from pathlib import Path

import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees import cli
from svzerodtrees.api import TuneBCsWorkflow


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
    assert "flow_observation_type: flow" in rendered
    assert "confirmation_absolute_tolerance: 1e-8" in rendered
    assert "pressure_bound_multiplier: 10.0" in rendered
    assert "cycle_stability_tolerance: 1e-3" in rendered
    assert "replay_minimum_cycles: 3" in rendered
    assert "replay_maximum_cycles: 10" in rendered
    assert "enforcement: strict_network  # strict_network | target_focused" in rendered
    assert "targets:" in rendered
    assert "normalized_rms_tolerance: 0.05" in rendered


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


def test_cli_dispatches_documented_full_pa_pipeline_config(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []

    cfg_path = tmp_path / "full-pa.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
  zerod_config: seed.json
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
  inflow: inflow.csv
bcs:
  type: impedance
  impedance:
    tuning_model: full_pa
    outlet_mapping_mode: auto
    tune_space:
      free:
        - name: lpa.alpha
          init: 0.9
          lb: 0.7
          ub: 0.99
      fixed: []
      tied: []
pipeline:
  run_steady: false
  optimize_bcs: true
  run_threed: false
  adapt: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": RecordingWorkflow})
    monkeypatch.setattr(
        cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)]
    )

    assert cli.main() == 0
    config = RecordingWorkflow.seen[0]
    assert config.bcs.type == "impedance"
    assert config.bcs.impedance.tuning_model == "full_pa"
    assert config.bcs.impedance.outlet_mapping_mode == "auto"
    assert config.bcs.impedance.use_mean is False
    assert config.bcs.impedance.diameter_scale == pytest.approx(1.0)


def test_cli_learned_full_pa_seed_generation_smoke(monkeypatch, tmp_path):
    """Exercise the public config/workflow boundary without external tools."""

    source_config = tmp_path / "source_0d_config.json"
    source_config.write_text('{"vessels": [], "boundary_conditions": []}\n', encoding="utf-8")
    centerline = tmp_path / "centerline.vtp"
    centerline.write_text("synthetic centerline input\n", encoding="utf-8")
    solver = tmp_path / "svzerodsolver"
    solver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    solver.chmod(0o755)
    clinical_targets = tmp_path / "clinical_targets.csv"
    clinical_targets.write_text("target,value\n", encoding="utf-8")
    (tmp_path / "mesh-surfaces").mkdir()

    learned_executable = tmp_path / "fake-learned-zerod"
    learned_executable.write_text(
        """#!%s
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', required=True)
parser.add_argument('--zerod-json', required=True)
parser.add_argument('--centerline-vtp', required=True)
parser.add_argument('--svzerod', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--output-filename', required=True)
args = parser.parse_args()
payload = {
    'boundary_conditions': [
        {'bc_name': 'INFLOW', 'bc_type': 'FLOW'},
        {'bc_name': 'RESISTANCE_1', 'bc_type': 'RESISTANCE', 'bc_values': {'R': 11.0}},
        {'bc_name': 'RESISTANCE_2', 'bc_type': 'RESISTANCE', 'bc_values': {'R': 12.0}},
        {'bc_name': 'RESISTANCE_3', 'bc_type': 'RESISTANCE', 'bc_values': {'R': 13.0}},
    ],
    'vessels': [
        {'vessel_name': 'branch1', 'boundary_conditions': {'outlet': 'RESISTANCE_1'}},
        {'vessel_name': 'branch2', 'boundary_conditions': {'outlet': 'RESISTANCE_2'}},
        {'vessel_name': 'branch3', 'boundary_conditions': {'outlet': 'RESISTANCE_3'}},
    ],
}
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
(output / args.output_filename).write_text(json.dumps(payload, sort_keys=True) + '\\n', encoding='utf-8')
""" % sys.executable,
        encoding="utf-8",
    )
    learned_executable.chmod(0o755)

    cfg_path = tmp_path / "learned.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: tune_bcs
paths:
  root: {tmp_path}
  clinical_targets: {clinical_targets}
  mesh_surfaces: {tmp_path / 'mesh-surfaces'}
seed_generation:
  method: learned_zerod
  anatomy: pulmonary
  input_zerod_config: {source_config}
  centerline: {centerline}
  svzerodsolver: {solver}
  output_dir: generated
  learned_zerod_executable: {learned_executable}
  output_filename: learned_full_pa_seed.json
bcs:
  type: impedance
  is_pulmonary: true
  impedance:
    tuning_model: full_pa
    outlet_mapping_mode: serialized_cap_order
""",
        encoding="utf-8",
    )

    calls = {}

    def fake_tuning(**kwargs):
        calls.update(kwargs)
        return {"tuned_zerod_config": str(tmp_path / "tuned.json")}

    reported = {}

    class ReportingTuneBCsWorkflow(TuneBCsWorkflow):
        def run(self):
            result = super().run()
            reported.update(result)
            return result

    monkeypatch.setattr("svzerodtrees.api.run_impedance_tuning_for_iteration", fake_tuning)
    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"tune_bcs": ReportingTuneBCsWorkflow})
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "tune-bcs", str(cfg_path)])

    assert cli.main() == 0

    seed_path = tmp_path / "generated" / "learned_full_pa_seed.json"
    metadata_path = tmp_path / "generated" / "learned_seed_metadata.json"
    assert reported["learned_seed"] == str(seed_path)
    assert reported["learned_seed_metadata"] == str(metadata_path)
    assert calls["seed_config"] == str(seed_path)
    assert json.loads(seed_path.read_text(encoding="utf-8"))["boundary_conditions"][-1]["bc_name"] == "RESISTANCE_3"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["method"] == "learned_zerod"
    assert metadata["output_paths"]["seed"] == str(seed_path)
    assert metadata["output_paths"]["metadata"] == str(metadata_path)
    assert metadata["command"]["argv"][0] == str(learned_executable)


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


def test_cli_dispatches_real_config_to_calibration_workflow(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {tmp_path}
  zerod_config: model.json
  output_config: calibrated.json
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: mapped.vtp
    centerline: centerline.vtp
    flow_observation_type: flow
  parameters:
    vessels:
      default: [R_poiseuille]
    junctions:
      default: [R_poiseuille]
  observation_qc:
    enforcement: target_focused
  targets:
    mpa_pressure:
      vessel: branch0_seg0
      interface: external_upstream
      weight: 1.0
      normalized_rms_tolerance: 0.05
    rpa_flow_split:
      rpa_vessel: branch1_seg0
      lpa_vessel: branch2_seg0
      interface: external_downstream
      weight: 1.0
      absolute_tolerance: 0.02
    require_improvement_over_baseline: true
  solver:
    replay_minimum_cycles: 3
    replay_maximum_cycles: 8
    required_consecutive_stable_pairs: 2
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"calibrate_0d_from_3d": RecordingWorkflow})
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["svzerodtrees", "calibrate-0d-from-3d", str(cfg_path)],
    )

    assert cli.main() == 0
    config = RecordingWorkflow.seen[0]
    assert config.workflow == "calibrate_0d_from_3d"
    assert config.calibration.observation_qc.enforcement == "target_focused"
    assert config.calibration.targets.mpa_pressure.vessel == "branch0_seg0"
    assert config.calibration.targets.mpa_pressure.interface == "external_upstream"
    assert config.calibration.targets.rpa_flow_split.rpa_vessel == "branch1_seg0"
    assert config.calibration.targets.rpa_flow_split.lpa_vessel == "branch2_seg0"
    assert config.calibration.solver.replay_minimum_cycles == 3
    assert config.calibration.solver.replay_maximum_cycles == 8
    assert config.calibration.solver.required_consecutive_stable_pairs == 2


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


def test_cli_adapt_benchmark_smoke(monkeypatch, tmp_path):
    preop = tmp_path / "preop.json"
    postop = tmp_path / "postop.json"
    tree_params = tmp_path / "optimized_params.csv"
    clinical = tmp_path / "clinical_targets.csv"
    for path in (preop, postop, tree_params, clinical):
        path.write_text("{}", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: adapt_benchmark
paths:
  root: {tmp_path}
adapt_benchmark:
  study_id: smoke-study
  output_dir: benchmark
  tree_params_csv: {tree_params}
  clinical_targets_csv: {clinical}
  scenarios:
    - name: baseline
      preop_rri_config: {preop}
      postop_rri_config: {postop}
""",
        encoding="utf-8",
    )

    def fake_run_benchmark(spec):
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_summary.csv").write_text("scenario,model\nbaseline,M1\n", encoding="utf-8")
        (out / "benchmark_summary.json").write_text('{"study_id":"smoke-study"}', encoding="utf-8")
        return {
            "study_id": spec.study_id,
            "summary_csv": str(out / "benchmark_summary.csv"),
            "summary_json": str(out / "benchmark_summary.json"),
            "rows": [],
        }

    monkeypatch.setattr("svzerodtrees.api.run_adaptation_benchmark_study", fake_run_benchmark)
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "adapt-benchmark", str(cfg_path)])

    assert cli.main() == 0
    assert (tmp_path / "benchmark" / "benchmark_summary.csv").exists()
    assert (tmp_path / "benchmark" / "benchmark_summary.json").exists()
