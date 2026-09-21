from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svzerodtrees.api import run_from_config_file
from svzerodtrees.calibration import workflow as calibration_workflow
from svzerodtrees.post_processing.centerline_timeseries import (
    publish_centerline_timeseries,
)


def _write_fixture_calibration_config(
    path: Path,
    *,
    output_path: Path,
    baseline_path: Path,
    mapped_path: Path,
    metadata_path: Path,
    target_focused: bool = False,
) -> None:
    target_config = ""
    if target_focused:
        target_config = """
  observation_qc:
    enforcement: target_focused
  targets:
    mpa_pressure:
      vessel: branch0_seg0
      interface: external_upstream
    rpa_flow_split:
      rpa_vessel: branch1_seg0
      lpa_vessel: branch2_seg0
      interface: external_downstream
"""
    path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {path.parent}
  zerod_config: {baseline_path}
  output_config: {output_path}
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: {mapped_path}
    metadata_json: {metadata_path}
    centerline: {mapped_path}
    pressure_array: pressure
    flow_array: flow
    flow_observation_type: flow
  parameters:
    vessels:
      default: [R_poiseuille]
    junctions:
      default: []
  solver:
    confirmation_absolute_tolerance: 1.0e-8
    confirmation_relative_tolerance: 1.0e-6
    # Replay pressures are solver-native dyn/cm^2 while the mapped fixture
    # observations are expressed in mmHg.
    pressure_bound_multiplier: 2000.0
    flow_bound_multiplier: 10.0
    cycle_stability_tolerance: 1.0e-3
{target_config}
""",
        encoding="utf-8",
    )


def _write_postprocess_suite_calibration_config(
    path: Path,
    *,
    output_path: Path,
    zerod_path: Path,
    descriptor_path: Path,
) -> None:
    path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {path.parent}
  zerod_config: {zerod_path}
  output_config: {output_path}
calibration:
  data_source:
    mode: postprocess_suite
    postprocess_metadata_json: {descriptor_path}
  parameters:
    vessels:
      default: [R_poiseuille]
    junctions:
      default: []
  solver:
    confirmation_absolute_tolerance: 1.0e-8
    confirmation_relative_tolerance: 1.0e-6
    pressure_bound_multiplier: 2000.0
    flow_bound_multiplier: 10.0
    cycle_stability_tolerance: 1.0e-3
""",
        encoding="utf-8",
    )


def test_target_focused_workflow_dispatches_with_valid_target_contract(
    monkeypatch, tmp_path
):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"
    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_fixture_calibration_config(
        config_path,
        output_path=output_path,
        baseline_path=baseline_path,
        mapped_path=mapped_path,
        metadata_path=metadata_path,
        target_focused=True,
    )

    calls = []

    def fake_calibrate(payload):
        calls.append(json.loads(json.dumps(payload)))
        return json.loads(json.dumps(payload))

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        fake_calibrate,
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        lambda _payload: _fixture_replay_rows(),
    )
    assemble = calibration_workflow.assemble_calibration_payload

    def advisory_global_failure(**kwargs):
        assembly = assemble(**kwargs)
        assembly.observation_qc["checks"]["pressure_drop_direction"] = False
        assembly.observation_qc["severity"]["pressure_drop_direction"] = "advisory"
        assembly.observation_qc["advisory_checks"]["pressure_drop_direction"] = False
        assembly.observation_qc["failed_advisory_checks"] = [
            "pressure_drop_direction"
        ]
        return assembly

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.assemble_calibration_payload",
        advisory_global_failure,
    )

    result = run_from_config_file(str(config_path))

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert result["target_quality"]["status"] == "pass"
    assert result["run_id"]
    assert result["calibration_targets_report"] == str(
        tmp_path / "calibration_targets.json"
    )
    report_paths = [
        result["observation_qc_report"],
        result["calibration_confirmation_report"],
        result["calibration_replay_report"],
        result["calibration_targets_report"],
        result["calibration_summary_report"],
    ]
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    assert all(report["run_id"] == result["run_id"] for report in reports)
    assert all(report["digests"] == result["digests"] for report in reports)
    assert result["observation_qc"]["enforcement"] == "target_focused"
    assert result["observation_qc"]["status"] == "pass"
    assert result["observation_qc"]["failed_advisory_checks"] == [
        "pressure_drop_direction"
    ]
    assert all(
        result["observation_qc"]["checks"][name]
        for name in (
            "root_waveform_agreement",
            "target_topology",
            "target_sampling_resolution",
            "target_split_denominator",
        )
    )


def test_target_gate_failure_does_not_publish_current_run_success(
    monkeypatch, tmp_path
):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"
    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_fixture_calibration_config(
        config_path,
        output_path=output_path,
        baseline_path=baseline_path,
        mapped_path=mapped_path,
        metadata_path=metadata_path,
        target_focused=True,
    )

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        lambda payload: json.loads(json.dumps(payload)),
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        lambda _payload: _fixture_replay_rows(pressure_offset=10.0),
    )

    with pytest.raises(ValueError, match="pulmonary target gates failed"):
        run_from_config_file(str(config_path))

    assert not output_path.exists()
    assert not (tmp_path / "calibration_summary.json").exists()
    target_report = json.loads(
        (tmp_path / "calibration_targets.json").read_text(encoding="utf-8")
    )
    assert target_report["status"] == "fail"
    assert target_report["candidate"]["gate_results"]["mpa_pressure"] is False
    assert target_report["run_id"]


def test_baseline_policy_failure_does_not_publish_candidate(
    monkeypatch, tmp_path
):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"
    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_fixture_calibration_config(
        config_path,
        output_path=output_path,
        baseline_path=baseline_path,
        mapped_path=mapped_path,
        metadata_path=metadata_path,
        target_focused=True,
    )

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        lambda payload: json.loads(json.dumps(payload)),
    )
    replay_calls = 0

    def baseline_then_worse(_payload):
        nonlocal replay_calls
        replay_calls += 1
        return _fixture_replay_rows(pressure_offset=0.01 if replay_calls == 2 else 0.0)

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        baseline_then_worse,
    )

    with pytest.raises(ValueError, match="pulmonary target gates failed"):
        run_from_config_file(str(config_path))

    assert replay_calls == 2
    assert not output_path.exists()
    assert not (tmp_path / "calibration_summary.json").exists()
    target_report = json.loads(
        (tmp_path / "calibration_targets.json").read_text(encoding="utf-8")
    )
    assert target_report["status"] == "fail"
    assert target_report["candidate"]["status"] == "pass"
    assert target_report["baseline_policy"]["status"] == "fail"


def _fixture_replay_rows(
    *, unstable: bool = False, pressure_offset: float = 0.0
) -> list[dict[str, float | str]]:
    # Simulated pysvzerod pressures are solver-native CGS (dyn/cm^2), while
    # the mapped 3D fixture metadata remains in mmHg.
    pressure_scale = 1333.2236842105263
    values = {
        "branch0_seg0": (10.0, 100.0, 90.0),
        "branch1_seg0": (6.0, 90.0, 80.0),
        "branch2_seg0": (4.0, 90.0, 70.0),
    }
    rows: list[dict[str, float | str]] = []
    for name, (flow, pressure_in, pressure_out) in values.items():
        for index in range(10):
            phase_index = index % 3
            phase_scale = (0.0, 0.1, 0.0)[phase_index]
            if unstable and index >= 4:
                flow = 12.0
                pressure_in = 2000.0
            rows.append(
                {
                    "name": name,
                    "time": float(index),
                    "flow_in": flow * (1.0 + phase_scale),
                    "flow_out": flow * (1.0 + phase_scale),
                    "pressure_in": pressure_scale
                    * (pressure_in + pressure_offset + (1.0 if phase_index == 1 else 0.0)),
                    "pressure_out": pressure_scale
                    * (pressure_out + pressure_offset + (1.0 if phase_index == 1 else 0.0)),
                }
            )
    return rows


def _write_polydata(
    path: Path,
    *,
    branch_ids,
    paths,
    pressure=None,
    flow=None,
    area=None,
    extra_arrays: dict[str, list[float]] | None = None,
) -> None:
    groups = []
    group_start = 0
    for index in range(1, len(branch_ids) + 1):
        if index == len(branch_ids) or branch_ids[index] != branch_ids[group_start]:
            groups.append((group_start, index))
            group_start = index

    def expand(values):
        if values is None:
            return None
        expanded = []
        for start, end in groups:
            group = list(values[start:end])
            expanded.extend(group)
            if len(group) == 2:
                expanded.insert(-1, (group[0] + group[1]) / 2.0)
        return expanded

    expanded_branch_ids = expand(branch_ids)
    expanded_paths = expand(paths)
    expanded_area = expand(area) if area is not None else [1.0] * len(expanded_branch_ids)
    points = vtk.vtkPoints()
    for index in range(len(expanded_branch_ids)):
        points.InsertNextPoint(float(index), 0.0, 0.0)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    lines = vtk.vtkCellArray()
    point_offset = 0
    for start, end in groups:
        count = end - start
        expanded_count = 3 if count == 2 else count
        for index in range(expanded_count - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_offset + index)
            line.GetPointIds().SetId(1, point_offset + index + 1)
            lines.InsertNextCell(line)
        point_offset += expanded_count
    poly.SetLines(lines)

    arrays = {
        "BranchId": expanded_branch_ids,
        "Path": expanded_paths,
        "CenterlineSectionArea": expanded_area,
    }
    if pressure is not None:
        arrays["pressure"] = expand(pressure)
    if flow is not None:
        arrays["velocity"] = expand(flow)
    if extra_arrays:
        arrays.update({name: expand(values) for name, values in extra_arrays.items()})

    for name, values in arrays.items():
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def _write_timeseries_metadata(path: Path) -> None:
    timestamps = [0.0, 0.5]
    path.write_text(
        json.dumps(
            {
                "kind": "centerline_timeseries_last_cycle",
                "frame_count": 2,
                "point_count": 9,
                "frame_indices": [0, 1],
                "timestamps_s": timestamps,
                "cycle_duration_s": 1.0,
                "data_contract": {
                    "pressure": {"quantity": "pressure", "units": "mmHg"},
                    "flow": {"quantity": "volumetric_flow", "units": "cm^3/s"},
                },
                "processed_frames": [
                    {
                        "frame_index": index,
                        "time_s": timestamp,
                        "point_arrays": [f"pressure_{index}", f"velocity_{index}"],
                    }
                    for index, timestamp in enumerate(timestamps)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_calibrate_0d_from_3d_workflow_writes_output(monkeypatch, tmp_path):
    zerod_path = tmp_path / "zerod.json"
    output_path = tmp_path / "calibrated.json"
    centerline_path = tmp_path / "centerline.vtp"
    mapped_path = tmp_path / "mapped.vtp"

    zerod_payload = {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [10.0, 10.0], "t": [0.0, 1.0]}},
            {"bc_name": "OUT1", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
            {"bc_name": "OUT2", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "junction_values": {"R_poiseuille": [0.1, 0.2], "L": [0.01, 0.02]},
            }
        ],
        "simulation_parameters": {
            "density": 1.06,
            "viscosity": 0.04,
            "number_of_cardiac_cycles": 1,
            "number_of_time_pts_per_cardiac_cycle": 4,
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    zerod_path.write_text(json.dumps(zerod_payload), encoding="utf-8")

    _write_polydata(
        centerline_path,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    _write_polydata(
        mapped_path,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        extra_arrays={
            "pressure_0": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "velocity_0": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
            "pressure_1": [101.0, 91.0, 91.0, 81.0, 91.0, 71.0],
            "velocity_1": [11.0, 11.0, 7.0, 7.0, 5.0, 5.0],
        },
    )
    metadata_path = tmp_path / "mapped_metadata.json"
    _write_timeseries_metadata(metadata_path)

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {tmp_path}
  zerod_config: {zerod_path.name}
  output_config: {output_path.name}
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: {mapped_path.name}
    metadata_json: {metadata_path.name}
    centerline: {centerline_path.name}
    flow_array: velocity
    flow_observation_type: flow
  parameters:
    vessels:
      default: [R_poiseuille, C]
      overrides:
        branch1_seg0: [R_poiseuille]
    junctions:
      default: [R_poiseuille, L]
  solver:
    maximum_iterations: 9
""",
        encoding="utf-8",
    )

    captured = {}
    calls = []

    def fake_calibrate(payload):
        calls.append(json.loads(json.dumps(payload)))
        captured["payload"] = payload
        result = json.loads(json.dumps(payload))
        result["vessels"][0]["zero_d_element_values"]["R_poiseuille"] = -4200.0
        return result

    def fake_simulate(payload):
        assert payload["simulation_parameters"]["number_of_cardiac_cycles"] >= 2
        assert payload["simulation_parameters"]["output_all_cycles"] is True
        rows = []
        for vessel in payload["vessels"]:
            for time in range(7):
                rows.append(
                    {
                        "name": vessel["vessel_name"],
                        "time": float(time),
                        "flow_in": 1.0,
                        "flow_out": 1.0,
                        "pressure_in": 100.0,
                        "pressure_out": 90.0,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr("svzerodtrees.calibration.workflow.calibrate_pysvzerod", fake_calibrate)
    monkeypatch.setattr("svzerodtrees.calibration.workflow.simulate_pysvzerod", fake_simulate)

    result = run_from_config_file(str(cfg_path))

    assert result["status"] == "ok"
    assert result["output_config"] == str(output_path)
    assert result["observation_count"] == 2
    assert result["variable_count"] == 12
    assert Path(result["observation_qc_report"]).exists()
    assert json.loads(Path(result["observation_qc_report"]).read_text())["status"] == "pass"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == -4200.0
    assert all("calibrate" not in vessel for vessel in written["vessels"])
    assert len(calls) == 2
    assert calls[1]["y"] == calls[0]["y"]
    assert calls[1]["dy"] == calls[0]["dy"]
    assert calls[1]["calibration_parameters"] == calls[0]["calibration_parameters"]
    assert calls[1]["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == -4200.0
    confirmation = json.loads(
        (tmp_path / "calibration_confirmation.json").read_text(encoding="utf-8")
    )
    assert [item["pass"] for item in confirmation["invocations"]] == [
        "first",
        "confirmation",
    ]
    assert "branch0_seg0.R_poiseuille" in confirmation["negative_parameter_paths"]
    assert "branch0_seg0.R_poiseuille" in confirmation["large_ratio_paths"]
    replay = json.loads((tmp_path / "calibration_replay.json").read_text(encoding="utf-8"))
    assert replay["status"] == "pass"
    assert json.loads((tmp_path / "calibration_summary.json").read_text(encoding="utf-8"))["status"] == "ok"
    assert captured["payload"]["calibration_parameters"]["maximum_iterations"] == 9
    assert captured["payload"]["y"]["flow:INFLOW:branch0_seg0"] == [10.0, 11.0]
    assert captured["payload"]["vessels"][1]["calibrate"] == ["R_poiseuille"]
    assert captured["payload"]["junctions"][0]["calibrate"] == ["R_poiseuille", "L"]


def test_calibrate_0d_from_3d_workflow_rejects_nonfinite_output(monkeypatch, tmp_path):
    zerod_path = tmp_path / "zerod.json"
    output_path = tmp_path / "calibrated.json"
    centerline_path = tmp_path / "centerline.vtp"
    mapped_path = tmp_path / "mapped.vtp"

    zerod_payload = {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [10.0, 10.0], "t": [0.0, 1.0]}},
            {"bc_name": "OUT1", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
            {"bc_name": "OUT2", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "junction_values": {"R_poiseuille": [0.1, 0.2], "L": [0.01, 0.02]},
            }
        ],
        "simulation_parameters": {
            "density": 1.06,
            "viscosity": 0.04,
            "number_of_cardiac_cycles": 1,
            "number_of_time_pts_per_cardiac_cycle": 1,
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    zerod_path.write_text(json.dumps(zerod_payload), encoding="utf-8")

    _write_polydata(
        centerline_path,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    _write_polydata(
        mapped_path,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        extra_arrays={
            "pressure_0": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "velocity_0": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
            "pressure_1": [101.0, 91.0, 91.0, 81.0, 91.0, 71.0],
            "velocity_1": [11.0, 11.0, 7.0, 7.0, 5.0, 5.0],
        },
    )
    metadata_path = tmp_path / "mapped_metadata.json"
    _write_timeseries_metadata(metadata_path)

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {tmp_path}
  zerod_config: {zerod_path.name}
  output_config: {output_path.name}
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: {mapped_path.name}
    metadata_json: {metadata_path.name}
    centerline: {centerline_path.name}
    flow_array: velocity
    flow_observation_type: flow
  parameters:
    vessels:
      default: [R_poiseuille, C]
    junctions:
      default: [R_poiseuille, L]
""",
        encoding="utf-8",
    )

    def fake_calibrate(payload):
        result = json.loads(json.dumps(payload))
        result["vessels"][0]["zero_d_element_values"]["R_poiseuille"] = float("nan")
        return result

    monkeypatch.setattr("svzerodtrees.calibration.workflow.calibrate_pysvzerod", fake_calibrate)

    with pytest.raises(ValueError, match="non-finite values after solver calibration"):
        run_from_config_file(str(cfg_path))

    assert not output_path.exists()


def test_fixture_workflow_accepts_stable_negative_resistance(monkeypatch, tmp_path):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"
    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_fixture_calibration_config(
        config_path,
        output_path=output_path,
        baseline_path=baseline_path,
        mapped_path=mapped_path,
        metadata_path=metadata_path,
    )

    calls: list[dict] = []

    def stable_negative_calibrate(payload):
        calls.append(json.loads(json.dumps(payload)))
        result = json.loads(json.dumps(payload))
        result["vessels"][0]["zero_d_element_values"]["R_poiseuille"] = -4.2
        return result

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        stable_negative_calibrate,
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        lambda _payload: _fixture_replay_rows(),
    )

    result = run_from_config_file(str(config_path))

    assert result["status"] == "ok"
    assert result["observation_count"] == 3
    assert result["variable_count"] == 12
    assert len(calls) == 2
    assert calls[0]["y"] == calls[1]["y"]
    assert calls[0]["dy"] == calls[1]["dy"]
    assert calls[0]["calibration_parameters"] == calls[1]["calibration_parameters"]
    assert calls[1]["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == -4.2

    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert published["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == -4.2
    assert published["junctions"][0]["junction_type"] == "NORMAL_JUNCTION"
    assert all("calibrate" not in vessel for vessel in published["vessels"])
    assert all("calibrate" not in junction for junction in published["junctions"])
    assert result["calibration_confirmation"]["negative_parameter_paths"] == [
        "branch0_seg0.R_poiseuille"
    ]
    assert any(
        item["path"] == "branch0_seg0.R_poiseuille"
        and item["value"] == -4.2
        for item in result["calibration_confirmation"]["warnings"][
            "negative_parameters"
        ]
    )
    assert result["run_id"]
    assert result["replay_stability"]["status"] == "pass"
    assert json.loads(
        (tmp_path / "calibration_observation_qc.json").read_text(encoding="utf-8")
    ) == result["observation_qc"]
    assert json.loads(
        (tmp_path / "calibration_summary.json").read_text(encoding="utf-8")
    )["status"] == "ok"


def test_fixture_workflow_rejects_finite_but_unstable_replay(monkeypatch, tmp_path):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"
    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_fixture_calibration_config(
        config_path,
        output_path=output_path,
        baseline_path=baseline_path,
        mapped_path=mapped_path,
        metadata_path=metadata_path,
    )

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        lambda payload: json.loads(json.dumps(payload)),
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        lambda _payload: _fixture_replay_rows(unstable=True),
    )

    with pytest.raises(ValueError, match="replay stability checks failed"):
        run_from_config_file(str(config_path))

    assert not output_path.exists()
    replay = json.loads(
        (tmp_path / "calibration_replay.json").read_text(encoding="utf-8")
    )
    assert replay["status"] == "fail"
    assert not replay["checks"]["cycle_stability"]


def test_real_centerline_producer_descriptor_calibrates_and_gates_lineage(
    monkeypatch, tmp_path
):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    source_path = fixture_dir / "mapped_timeseries.vtp"
    postprocess_dir = tmp_path / "postprocess"
    descriptor_path = postprocess_dir / "postprocess_suite_metadata.json"
    postprocess_dir.mkdir()

    # Seed the producer with the same completed suite envelope that the
    # pulmonary suite writes, then let the real centerline producer add its
    # artifact and exact tuned-config lineage.
    completed_steps = {
        name: {"status": "completed"}
        for name in (
            "pressure",
            "flow_split",
            "frames",
            "resistance_map",
            "centerline_timeseries",
            "resistance_map_systolic",
        )
    }
    descriptor_path.write_text(
        json.dumps(
            {
                "kind": "pulmonary_threed_suite",
                "schema_version": "1.0",
                "status": "completed",
                "steps": completed_steps,
            }
        ),
        encoding="utf-8",
    )

    def write_source_frame(frame_index: int) -> Path:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(source_path))
        reader.Update()
        poly = vtk.vtkPolyData()
        poly.DeepCopy(reader.GetOutput())
        for source_name, output_name in (
            (f"pressure_{frame_index}", "pressure"),
            (f"flow_{frame_index}", "velocity"),
        ):
            source_array = poly.GetPointData().GetArray(source_name)
            assert source_array is not None
            output_array = numpy_to_vtk(vtk_to_numpy(source_array), deep=True)
            output_array.SetName(output_name)
            poly.GetPointData().AddArray(output_array)
        output_path = postprocess_dir / f"mapped_frame_{frame_index}.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(poly)
        assert writer.Write() == 1
        return output_path

    source_frames = [
        {"path": str(write_source_frame(index)), "time_s": index / 3.0}
        for index in range(3)
    ]
    producer_result = publish_centerline_timeseries(
        selected_frames=source_frames,
        output_dir=postprocess_dir,
        cycle_duration_s=1.0,
        reference_centerline=source_path,
        suite_metadata_path=descriptor_path,
        tuned_zerod_config_path=baseline_path,
    )
    published_descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert published_descriptor["status"] == "completed"
    assert published_descriptor["lineage"]["tuned_zerod_config_sha256"]
    assert producer_result["descriptor"] == str(descriptor_path)

    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.yml"
    _write_postprocess_suite_calibration_config(
        config_path,
        output_path=output_path,
        zerod_path=baseline_path,
        descriptor_path=descriptor_path,
    )
    calls = []

    def fake_calibrate(payload):
        calls.append(json.loads(json.dumps(payload)))
        return json.loads(json.dumps(payload))

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        fake_calibrate,
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod",
        lambda _payload: _fixture_replay_rows(),
    )

    result = run_from_config_file(str(config_path))

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert result["data_source_provenance"]["mode"] == "postprocess_suite"
    assert result["data_source_provenance"]["lineage"]["status"] == "matched"
    assert result["data_source_provenance"]["descriptor_sha256"]

    # A byte-different tuned model is rejected during descriptor resolution,
    # before the existing calibrator can be dispatched.
    different_zerod_path = tmp_path / "different_zerod.json"
    different_zerod_path.write_bytes(baseline_path.read_bytes() + b"\n")
    different_config_path = tmp_path / "different_calibrate.yml"
    _write_postprocess_suite_calibration_config(
        different_config_path,
        output_path=tmp_path / "different_calibrated.json",
        zerod_path=different_zerod_path,
        descriptor_path=descriptor_path,
    )
    with pytest.raises(ValueError, match="lineage"):
        run_from_config_file(str(different_config_path))
    assert len(calls) == 2

    # A failed suite envelope may still retain the valid centerline artifact
    # for diagnostics, but it is not a calibration source.
    failed_descriptor_path = tmp_path / "failed_suite_metadata.json"
    failed_descriptor_path.write_text(
        json.dumps(
            {
                "kind": "pulmonary_threed_suite",
                "schema_version": "1.0",
                "status": "failed",
                "steps": {
                    **completed_steps,
                    "resistance_map_systolic": {"status": "failed"},
                },
            }
        ),
        encoding="utf-8",
    )
    publish_centerline_timeseries(
        selected_frames=source_frames,
        output_dir=postprocess_dir,
        cycle_duration_s=1.0,
        reference_centerline=source_path,
        suite_metadata_path=failed_descriptor_path,
        tuned_zerod_config_path=baseline_path,
    )
    failed_config_path = tmp_path / "failed_calibrate.yml"
    _write_postprocess_suite_calibration_config(
        failed_config_path,
        output_path=tmp_path / "failed_calibrated.json",
        zerod_path=baseline_path,
        descriptor_path=failed_descriptor_path,
    )
    with pytest.raises(ValueError, match="completed terminal status"):
        run_from_config_file(str(failed_config_path))
    assert len(calls) == 2
