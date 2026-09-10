from __future__ import annotations

import json
from pathlib import Path

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
    points = vtk.vtkPoints()
    for point in (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, -1.0, 0.0),
    ):
        points.InsertNextPoint(*point)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    lines = vtk.vtkCellArray()
    for start in (0, 2, 4):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, start)
        line.GetPointIds().SetId(1, start + 1)
        lines.InsertNextCell(line)
    poly.SetLines(lines)

    arrays = {
        "BranchId": branch_ids,
        "Path": paths,
        "CenterlineSectionArea": area if area is not None else [1.0] * len(branch_ids),
    }
    if pressure is not None:
        arrays["pressure"] = pressure
    if flow is not None:
        arrays["velocity"] = flow
    if extra_arrays:
        arrays.update(extra_arrays)

    for name, values in arrays.items():
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


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
    centerline: {centerline_path.name}
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

    def fake_calibrate(payload):
        captured["payload"] = payload
        result = json.loads(json.dumps(payload))
        result["vessels"][0]["zero_d_element_values"]["R_poiseuille"] = 42.0
        return result

    monkeypatch.setattr("svzerodtrees.calibration.workflow.calibrate_pysvzerod", fake_calibrate)

    result = run_from_config_file(str(cfg_path))

    assert result["status"] == "ok"
    assert result["output_config"] == str(output_path)
    assert result["observation_count"] == 2
    assert result["variable_count"] == 12
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == 42.0
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
    centerline: {centerline_path.name}
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
