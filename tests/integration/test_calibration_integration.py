from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.api import run_from_config_file


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
