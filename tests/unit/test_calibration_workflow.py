from __future__ import annotations

import json
from pathlib import Path

import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.calibration.workflow import assemble_calibration_payload
from svzerodtrees.config import (
    CalibrationConfig,
    CalibrationDataSourceConfig,
    CalibrationParametersConfig,
    CalibrationParameterSelectionConfig,
    CalibrationSolverConfig,
)


def _write_polydata(path: Path, *, branch_ids, paths, pressure=None, flow=None) -> None:
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
    }
    if pressure is not None:
        arrays["pressure"] = pressure
    if flow is not None:
        arrays["velocity"] = flow

    for name, values in arrays.items():
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def _write_zerod_config(path: Path) -> None:
    payload = {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [10.0], "t": [0.0]}},
            {"bc_name": "OUT1", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
            {"bc_name": "OUT2", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "junction_values": {
                    "R_poiseuille": [0.1, 0.2],
                    "L": [0.01, 0.02],
                    "stenosis_coefficient": [0.0, 0.0],
                },
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
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _calibration_config(tmp_path: Path) -> CalibrationConfig:
    return CalibrationConfig(
        data_source=CalibrationDataSourceConfig(
            mode="mapped_centerline",
            mapped_centerline_result=str(tmp_path / "mapped.vtp"),
            centerline=str(tmp_path / "centerline.vtp"),
        ),
        parameters=CalibrationParametersConfig(
            vessels=CalibrationParameterSelectionConfig(
                default=["R_poiseuille", "C"],
                overrides={"branch2_seg0": ["R_poiseuille"]},
            ),
            junctions=CalibrationParameterSelectionConfig(
                default=["R_poiseuille", "L"],
            ),
        ),
        solver=CalibrationSolverConfig(
            initial_damping_factor=2.0,
            maximum_iterations=7,
            tolerance_gradient=1e-5,
            tolerance_increment=1e-8,
        ),
    )


def test_assemble_calibration_payload_from_mapped_centerline(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"

    _write_polydata(
        centerline,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        pressure=[100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
        flow=[10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
    )
    _write_zerod_config(zerod)

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=_calibration_config(tmp_path),
    )

    assert assembly.observation_count == 1
    assert assembly.variable_count == 12
    assert assembly.solver_payload["y"]["flow:INFLOW:branch0_seg0"] == [10.0]
    assert assembly.solver_payload["y"]["pressure:branch0_seg0:J0"] == [90.0]
    assert assembly.solver_payload["y"]["flow:J0:branch1_seg0"] == [6.0]
    assert assembly.solver_payload["y"]["pressure:branch2_seg0:OUT2"] == [70.0]
    assert assembly.solver_payload["dy"]["pressure:J0:branch2_seg0"] == [0.0]
    assert assembly.solver_payload["vessels"][0]["calibrate"] == ["R_poiseuille", "C"]
    assert assembly.solver_payload["vessels"][2]["calibrate"] == ["R_poiseuille"]
    assert assembly.solver_payload["junctions"][0]["calibrate"] == ["R_poiseuille", "L"]
    assert assembly.solver_payload["calibration_parameters"] == {
        "initial_damping_factor": 2.0,
        "maximum_iterations": 7,
        "tolerance_gradient": 1e-05,
        "tolerance_increment": 1e-08,
    }


def test_assemble_calibration_payload_rejects_multiple_segments_per_branch(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"

    _write_polydata(
        centerline,
        branch_ids=[0, 0, 0, 0, 0, 0],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 0, 0, 0, 0],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        pressure=[100.0, 90.0, 100.0, 90.0, 100.0, 90.0],
        flow=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    )

    payload = {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [1.0], "t": [0.0]}},
            {"bc_name": "OUT", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
        ],
        "junctions": [],
        "simulation_parameters": {},
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 1.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch0_seg1",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 1.0},
                "boundary_conditions": {"outlet": "OUT"},
            },
        ],
    }
    zerod.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="one 0D vessel per centerline branch"):
        assemble_calibration_payload(
            zerod_config_path=str(zerod),
            calibration=CalibrationConfig(
                data_source=CalibrationDataSourceConfig(
                    mode="mapped_centerline",
                    mapped_centerline_result=str(mapped),
                    centerline=str(centerline),
                ),
                parameters=CalibrationParametersConfig(),
            ),
        )
