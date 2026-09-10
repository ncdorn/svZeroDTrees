from __future__ import annotations

import json
from pathlib import Path

import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.calibration.workflow import (
    CalibrationAssembly,
    _evaluate_replay,
    _normalize_calibrated_config,
    _normalize_calibration_input,
    assemble_calibration_payload,
    calibrate_0d_from_mapped_centerline,
)
from svzerodtrees.config import (
    CalibrationConfig,
    CalibrationDataSourceConfig,
    CalibrationInputNormalizationConfig,
    CalibrationParametersConfig,
    CalibrationParameterSelectionConfig,
    CalibrationSolverConfig,
)


def _write_polydata(
    path: Path,
    *,
    branch_ids,
    paths,
    pressure=None,
    flow=None,
    area=None,
    extra_arrays: dict[str, list[float]] | None = None,
    expand_two_point_branches: bool = True,
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
            if expand_two_point_branches and len(group) == 2:
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
        expanded_count = 3 if expand_two_point_branches and count == 2 else count
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


def _write_zerod_config(path: Path) -> None:
    payload = {
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
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_timeseries_metadata(
    path: Path,
    *,
    frame_count: int,
    point_count: int = 9,
    flow_array: str = "velocity",
    cycle_duration_s: float = 1.0,
) -> None:
    timestamps = [cycle_duration_s * index / frame_count for index in range(frame_count)]
    processed_frames = [
        {
            "frame_index": index,
            "time_s": timestamp,
            "point_arrays": [f"pressure_{index}", f"{flow_array}_{index}"],
        }
        for index, timestamp in enumerate(timestamps)
    ]
    path.write_text(
        json.dumps(
            {
                "kind": "centerline_timeseries_last_cycle",
                "frame_count": frame_count,
                "point_count": point_count,
                "frame_indices": list(range(frame_count)),
                "timestamps_s": timestamps,
                "cycle_duration_s": cycle_duration_s,
                "data_contract": {
                    "pressure": {"quantity": "pressure", "units": "mmHg"},
                    "flow": {"quantity": "volumetric_flow", "units": "cm^3/s"},
                },
                "processed_frames": processed_frames,
            }
        ),
        encoding="utf-8",
    )


def _calibration_config(tmp_path: Path) -> CalibrationConfig:
    return CalibrationConfig(
        data_source=CalibrationDataSourceConfig(
            mode="mapped_centerline",
            mapped_centerline_result=str(tmp_path / "mapped.vtp"),
            metadata_json=str(tmp_path / "mapped_metadata.json"),
            centerline=str(tmp_path / "centerline.vtp"),
            flow_array="velocity",
            flow_observation_type="flow",
            area_array="CenterlineSectionArea",
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
        input_normalization=CalibrationInputNormalizationConfig(),
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
    assert assembly.solver_payload["y"]["pressure:branch2_seg0:OUT2"] == [80.0]
    assert assembly.solver_payload["dy"]["pressure:J0:branch2_seg0"] == [0.0]
    assert assembly.solver_payload["vessels"][0]["calibrate"] == ["R_poiseuille", "C"]
    assert assembly.solver_payload["vessels"][2]["calibrate"] == ["R_poiseuille"]
    assert assembly.solver_payload["junctions"][0]["calibrate"] == ["R_poiseuille", "L"]
    assert assembly.observation_qc["status"] == "pass"
    assert assembly.observation_qc["checks"] == {
        "junction_mass_balance": True,
        "path_coverage": True,
        "pressure_drop_direction": True,
        "root_waveform_agreement": True,
        "sampling_resolution": True,
        "vessel_flow_continuity": True,
    }
    assert assembly.interface_sampling["branch0_seg0:upstream"] == {
        "interface_kind": "external_upstream",
        "quality_status": "qualified_interior",
        "requested_path": 0.0,
        "selected_path": 0.5,
        "inset_distance": 0.5,
        "usable_sample_count": 3,
        "paired_observation": True,
        "excluded_from_calibration": False,
    }
    assert assembly.interface_sampling["branch0_seg0:downstream"]["selected_path"] == 1.0
    assert assembly.solver_payload["calibration_parameters"] == {
        "initial_damping_factor": 2.0,
        "maximum_iterations": 7,
        "tolerance_gradient": 1e-05,
        "tolerance_increment": 1e-08,
    }


def test_committed_calibration_fixture_covers_public_observation_contract(
    fixtures_dir, tmp_path
):
    fixture_dir = fixtures_dir / "calibration"
    baseline_path = fixture_dir / "finite_rigid_baseline.json"
    mapped_path = fixture_dir / "mapped_timeseries.vtp"
    metadata_path = fixture_dir / "mapped_timeseries_metadata.json"

    calibration = CalibrationConfig(
        data_source=CalibrationDataSourceConfig(
            mode="mapped_centerline",
            mapped_centerline_result=str(mapped_path),
            metadata_json=str(metadata_path),
            centerline=str(mapped_path),
            flow_array="flow",
            flow_observation_type="flow",
        ),
        parameters=CalibrationParametersConfig(
            vessels=CalibrationParameterSelectionConfig(default=["R_poiseuille"]),
            junctions=CalibrationParameterSelectionConfig(default=[]),
        ),
    )

    assembly = assemble_calibration_payload(
        zerod_config_path=str(baseline_path),
        calibration=calibration,
    )

    assert assembly.observation_count == 3
    assert assembly.variable_count == 12
    # The fixture advertises an area of 7 for each point.  Direct integrated
    # flow must still reach the solver unchanged.
    assert assembly.solver_payload["y"]["flow:INFLOW:branch0_seg0"] == [
        10.0,
        11.0,
        10.0,
    ]
    assert assembly.solver_payload["y"]["flow:J0:branch1_seg0"] == [
        6.0,
        6.6,
        6.0,
    ]
    assert assembly.interface_sampling["branch0_seg0:upstream"] == {
        "interface_kind": "external_upstream",
        "quality_status": "qualified_interior",
        "requested_path": 0.0,
        "selected_path": 0.3333333333333333,
        "inset_distance": 0.3333333333333333,
        "usable_sample_count": 4,
        "paired_observation": True,
        "excluded_from_calibration": False,
    }
    assert assembly.observation_qc["status"] == "pass"

    normalized, normalization = _normalize_calibrated_config(
        assembly.solver_payload
    )
    assert normalized["junctions"][0]["junction_type"] == "NORMAL_JUNCTION"
    assert normalization["junction_type_changes"] == [
        {
            "from": "internal_junction",
            "junction_name": "J0",
            "to": "NORMAL_JUNCTION",
        }
    ]

    # The fixture also carries an unmapped two-point branch. Add that branch
    # to a temporary copy of the finite baseline to prove explicit exclusion
    # is the only supported way to assemble an under-resolved vessel.
    short_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    short_baseline["boundary_conditions"].extend(
        [
            {
                "bc_name": "SHORT_IN",
                "bc_type": "FLOW",
                "bc_values": {"Q": [2.0], "t": [0.0]},
            },
            {
                "bc_name": "SHORT_OUT",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 100.0},
            },
        ]
    )
    short_baseline["vessels"].append(
        {
            "vessel_id": 3,
            "vessel_name": "branch3_seg0",
            "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {"R_poiseuille": 40.0, "C": 0.0, "L": 0.04},
            "boundary_conditions": {"inlet": "SHORT_IN", "outlet": "SHORT_OUT"},
        }
    )
    short_baseline_path = tmp_path / "short_baseline.json"
    short_baseline_path.write_text(json.dumps(short_baseline), encoding="utf-8")
    short_calibration = CalibrationConfig(
        data_source=calibration.data_source,
        parameters=CalibrationParametersConfig(
            vessels=CalibrationParameterSelectionConfig(
                default=["R_poiseuille"], overrides={"branch3_seg0": []}
            ),
            junctions=CalibrationParameterSelectionConfig(default=[]),
        ),
    )

    short_assembly = assemble_calibration_payload(
        zerod_config_path=str(short_baseline_path),
        calibration=short_calibration,
    )

    assert short_assembly.excluded_blocks == {
        "branch3_seg0": "empty_vessel_parameter_override"
    }
    assert short_assembly.interface_sampling["branch3_seg0:upstream"][
        "quality_status"
    ] == "excluded_underresolved"
    assert short_assembly.observation_qc["status"] == "pass"


def test_endpoint_qualification_rejects_underresolved_branch(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"
    _write_polydata(
        centerline,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        expand_two_point_branches=False,
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        pressure=[100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
        flow=[10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
        expand_two_point_branches=False,
    )
    _write_zerod_config(zerod)

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=_calibration_config(tmp_path),
    )

    assert assembly.observation_qc["status"] == "fail"
    assert not assembly.observation_qc["checks"]["sampling_resolution"]
    assert assembly.interface_sampling["branch0_seg0:upstream"]["quality_status"] == (
        "underresolved"
    )


def test_explicit_empty_parameter_override_records_underresolved_exclusion(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"
    for path in (centerline, mapped):
        _write_polydata(
            path,
            branch_ids=[0, 0, 1, 1, 2, 2],
            paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            pressure=([100.0, 90.0, 90.0, 80.0, 90.0, 70.0] if path == mapped else None),
            flow=([10.0, 10.0, 6.0, 6.0, 4.0, 4.0] if path == mapped else None),
            expand_two_point_branches=False,
        )
    _write_zerod_config(zerod)

    calibration = _calibration_config(tmp_path)
    calibration.parameters.vessels.overrides = {
        "branch0_seg0": [],
        "branch1_seg0": [],
        "branch2_seg0": [],
    }
    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=calibration,
    )

    assert assembly.excluded_blocks == {
        "branch0_seg0": "empty_vessel_parameter_override",
        "branch1_seg0": "empty_vessel_parameter_override",
        "branch2_seg0": "empty_vessel_parameter_override",
    }
    assert assembly.interface_sampling["branch0_seg0:upstream"]["quality_status"] == (
        "excluded_underresolved"
    )
    assert assembly.interface_sampling["branch0_seg0:upstream"]["excluded_from_calibration"]


def test_failed_observation_qc_writes_report_without_dispatching_solver(monkeypatch, tmp_path):
    output_path = tmp_path / "calibrated.json"
    assembly = CalibrationAssembly(
        solver_payload={},
        observation_count=1,
        variable_count=0,
        input_normalization={},
        observation_qc={
            "status": "fail",
            "checks": {"junction_mass_balance": False},
            "metrics": {},
            "thresholds": {},
            "selected_interfaces": {},
            "exclusions": {},
        },
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.assemble_calibration_payload",
        lambda **_kwargs: assembly,
    )
    solver_called = False

    def fail_if_called(_payload):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver dispatch must be gated by observation QC")

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="observation QC failed before solver dispatch"):
        calibrate_0d_from_mapped_centerline(
            zerod_config_path="unused.json",
            output_config_path=str(output_path),
            calibration=None,
        )

    assert not solver_called
    assert not output_path.exists()
    report = json.loads(
        (tmp_path / "calibration_observation_qc.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "fail"


def test_confirmation_failure_does_not_publish_solver_config(monkeypatch, tmp_path):
    output_path = tmp_path / "calibrated.json"
    assembly = CalibrationAssembly(
        solver_payload={
            "vessels": [
                {
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_values": {"R_poiseuille": 1.0, "C": 2.0},
                    "calibrate": ["R_poiseuille"],
                }
            ],
            "junctions": [],
            "y": {"flow:IN:branch0_seg0": [1.0]},
            "dy": {"flow:IN:branch0_seg0": [0.0]},
            "calibration_parameters": {"maximum_iterations": 1},
        },
        observation_count=1,
        variable_count=1,
        input_normalization={},
        observation_qc={
            "status": "pass",
            "checks": {},
            "metrics": {},
            "thresholds": {},
            "selected_interfaces": {},
            "exclusions": {},
        },
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.assemble_calibration_payload",
        lambda **_kwargs: assembly,
    )
    calls = []

    def fake_calibrate(payload):
        calls.append(json.loads(json.dumps(payload)))
        result = json.loads(json.dumps(payload))
        if len(calls) == 2:
            result["vessels"][0]["zero_d_element_values"]["R_poiseuille"] = 2.0
        return result

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod",
        fake_calibrate,
    )

    with pytest.raises(ValueError, match="did not reach a parameter fixed point"):
        calibrate_0d_from_mapped_centerline(
            zerod_config_path="unused.json",
            output_config_path=str(output_path),
            calibration=_calibration_config(tmp_path),
        )

    assert len(calls) == 2
    assert calls[1]["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == 1.0
    assert not output_path.exists()
    assert not (tmp_path / "calibration_confirmation.json").exists()


def test_normalizes_calibration_output_for_simulation():
    calibrated = {
        "y": {},
        "dy": {},
        "calibration_parameters": {"maximum_iterations": 1},
        "calibration_diagnostics": {"status": "ok"},
        "vessels": [
            {
                "vessel_name": "branch0_seg0",
                "calibrate": ["R_poiseuille"],
                "zero_d_element_values": {"R_poiseuille": -8.0},
            }
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "internal_junction",
                "inlet_vessels": [0],
                "outlet_vessels": [1],
                "junction_values": {"R_poiseuille": 1.0},
                "calibrate": [],
            },
            {
                "junction_name": "J1",
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [2],
                "outlet_vessels": [3, 4],
                "junction_values": {
                    "R_poiseuille": [1.0, 2.0],
                    "L": [0.0, 0.0],
                    "stenosis_coefficient": [0.0, 0.0],
                },
                "calibrate": [],
            },
        ],
    }

    normalized, report = _normalize_calibrated_config(calibrated)

    assert calibrated["junctions"][0]["junction_type"] == "internal_junction"
    assert normalized["junctions"][0]["junction_type"] == "NORMAL_JUNCTION"
    assert "junction_values" not in normalized["junctions"][0]
    assert normalized["junctions"][1]["junction_values"] == calibrated["junctions"][1][
        "junction_values"
    ]
    assert all("calibrate" not in block for block in normalized["vessels"])
    assert all("calibrate" not in block for block in normalized["junctions"])
    assert set(normalized) == {"vessels", "junctions"}
    assert report["junction_type_changes"] == [
        {
            "from": "internal_junction",
            "junction_name": "J0",
            "to": "NORMAL_JUNCTION",
        }
    ]
    assert report["removed_junction_values"] == ["junctions.J0.junction_values"]


def test_replay_evaluation_accepts_stable_negative_resistance_result():
    payload = {
        "y": {
            "flow:INFLOW:branch0_seg0": [1.0, 1.0],
            "pressure:INFLOW:branch0_seg0": [100.0, 100.0],
        }
    }
    rows = []
    for name, value in (
        ("branch0_seg0", 1.0),
        ("branch1_seg0", 0.5),
    ):
        for time in range(7):
            rows.append(
                {
                    "name": name,
                    "time": float(time),
                    "flow_in": value,
                    "flow_out": value,
                    "pressure_in": 100.0,
                    "pressure_out": 90.0,
                }
            )

    summary = _evaluate_replay(
        result=rows,
        payload=payload,
        replay_settings={
            "pressure_bound_multiplier": 2.0,
            "flow_bound_multiplier": 2.0,
            "cycle_stability_tolerance": 0.0,
        },
        validation_settings={"number_of_time_pts_per_cardiac_cycle": 4},
    )

    assert summary["status"] == "pass"
    assert summary["checks"] == {
        "finite_pressure_and_flow": True,
        "bounded_pressure": True,
        "bounded_flow": True,
        "cycle_stability": True,
    }


def test_replay_evaluation_rejects_unstable_or_unbounded_result():
    payload = {
        "y": {
            "flow:INFLOW:branch0_seg0": [1.0, 1.0],
            "pressure:INFLOW:branch0_seg0": [100.0, 100.0],
        }
    }
    rows = []
    for time in range(7):
        value = 1.0 if time < 4 else 1.5
        rows.append(
            {
                "name": "branch0_seg0",
                "time": float(time),
                "flow_in": value,
                "flow_out": value,
                "pressure_in": 100.0 if time < 4 else 1000.0,
                "pressure_out": 90.0,
            }
        )

    summary = _evaluate_replay(
        result=rows,
        payload=payload,
        replay_settings={
            "pressure_bound_multiplier": 2.0,
            "flow_bound_multiplier": 2.0,
            "cycle_stability_tolerance": 0.01,
        },
        validation_settings={"number_of_time_pts_per_cardiac_cycle": 4},
    )

    assert summary["status"] == "fail"
    assert not summary["checks"]["bounded_pressure"]
    assert not summary["checks"]["cycle_stability"]


def test_failed_replay_does_not_publish_solver_config(monkeypatch, tmp_path):
    output_path = tmp_path / "calibrated.json"
    assembly = CalibrationAssembly(
        solver_payload={
            "boundary_conditions": [
                {
                    "bc_name": "IN",
                    "bc_type": "FLOW",
                    "bc_values": {"Q": [1.0, 1.0], "t": [0.0, 1.0]},
                }
            ],
            "simulation_parameters": {
                "number_of_cardiac_cycles": 1,
                "number_of_time_pts_per_cardiac_cycle": 4,
            },
            "vessels": [
                {
                    "vessel_id": 0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {"R_poiseuille": -1.0},
                    "boundary_conditions": {"inlet": "IN"},
                    "calibrate": ["R_poiseuille"],
                }
            ],
            "junctions": [],
            "y": {
                "flow:IN:branch0_seg0": [1.0, 1.0],
                "pressure:IN:branch0_seg0": [100.0, 100.0],
            },
            "dy": {},
            "calibration_parameters": {},
        },
        observation_count=2,
        variable_count=2,
        input_normalization={},
        observation_qc={
            "status": "pass",
            "checks": {},
            "metrics": {},
            "thresholds": {},
            "selected_interfaces": {},
            "exclusions": {},
        },
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.assemble_calibration_payload",
        lambda **_kwargs: assembly,
    )

    def fake_calibrate(payload):
        return json.loads(json.dumps(payload))

    def unstable_simulate(_payload):
        return [
            {
                "name": "branch0_seg0",
                "time": float(index),
                "flow_in": 1.0,
                "flow_out": 1.0,
                "pressure_in": 100.0 if index < 4 else 2000.0,
                "pressure_out": 90.0,
            }
            for index in range(7)
        ]

    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.calibrate_pysvzerod", fake_calibrate
    )
    monkeypatch.setattr(
        "svzerodtrees.calibration.workflow.simulate_pysvzerod", unstable_simulate
    )

    with pytest.raises(ValueError, match="replay stability checks failed"):
        calibrate_0d_from_mapped_centerline(
            zerod_config_path="unused.json",
            output_config_path=str(output_path),
            calibration=_calibration_config(tmp_path),
        )

    assert not output_path.exists()
    replay = json.loads(
        (tmp_path / "calibration_replay.json").read_text(encoding="utf-8")
    )
    assert replay["status"] == "fail"


def test_normalizes_only_positive_infinite_vessel_compliance(tmp_path):
    zerod = tmp_path / "zerod.json"
    _write_zerod_config(zerod)
    source = json.loads(zerod.read_text(encoding="utf-8"))
    source["vessels"][0]["zero_d_element_values"]["C"] = float("inf")
    source_before = json.loads(json.dumps(source))

    normalized, report = _normalize_calibration_input(
        source,
        infinite_vessel_compliance="zero",
    )

    assert source == source_before
    assert normalized["vessels"][0]["zero_d_element_values"]["C"] == 0.0
    assert report == {
        "infinite_vessel_compliance": "zero",
        "changed_count": 1,
        "changed_paths": ["vessels[0].zero_d_element_values.C"],
    }


def test_rigid_baseline_normalization_matches_finite_reference():
    root = Path(__file__).parents[2]
    artifact_dir = root / "tmp/tst-stan-5-iter03-centerline-timeseries"
    source = json.loads((artifact_dir / "baseline_0d.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (artifact_dir / "baseline_0d_c0.json").read_text(encoding="utf-8")
    )

    normalized, report = _normalize_calibration_input(
        source,
        infinite_vessel_compliance="zero",
    )

    assert normalized == reference
    assert report["changed_count"] == 38
    assert len(report["changed_paths"]) == 38


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_rejects_unsupported_nonfinite_compliance(value):
    source = {"vessels": [{"zero_d_element_values": {"C": value}}]}

    with pytest.raises(ValueError, match="only positive infinity"):
        _normalize_calibration_input(
            source,
            infinite_vessel_compliance="zero",
        )


def test_rejects_positive_infinity_outside_vessel_compliance():
    source = {
        "vessels": [{"zero_d_element_values": {"C": 0.1}}],
        "simulation_parameters": {"density": float("inf")},
    }

    with pytest.raises(ValueError, match="simulation_parameters.density"):
        _normalize_calibration_input(
            source,
            infinite_vessel_compliance="zero",
        )


def test_assemble_calibration_payload_from_mapped_centerline_timeseries(tmp_path):
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
        extra_arrays={
            "pressure_0": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "velocity_0": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
            "pressure_1": [101.0, 91.0, 91.0, 81.0, 91.0, 71.0],
            "velocity_1": [11.0, 11.0, 7.0, 7.0, 5.0, 5.0],
        },
    )
    _write_timeseries_metadata(tmp_path / "mapped_metadata.json", frame_count=2)
    _write_zerod_config(zerod)

    calibration = _calibration_config(tmp_path)
    calibration.data_source.pressure_array = "pressure"
    calibration.data_source.flow_array = "velocity"

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=calibration,
    )

    assert assembly.observation_count == 2
    assert assembly.variable_count == 12
    assert assembly.solver_payload["y"]["flow:INFLOW:branch0_seg0"] == [10.0, 11.0]
    assert assembly.solver_payload["y"]["pressure:branch0_seg0:J0"] == [90.0, 91.0]
    assert assembly.solver_payload["y"]["flow:J0:branch1_seg0"] == [6.0, 7.0]
    assert assembly.solver_payload["y"]["pressure:branch2_seg0:OUT2"] == [80.0, 81.0]
    assert assembly.solver_payload["dy"]["pressure:J0:branch2_seg0"] == [0.0, 0.0]


def test_assemble_calibration_payload_converts_velocity_to_flow_with_area(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"

    _write_polydata(
        centerline,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        area=[2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        pressure=[100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
        flow=[10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
        area=[2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        extra_arrays={
            "centerline_velocity": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0]
        },
    )
    _write_zerod_config(zerod)

    calibration = _calibration_config(tmp_path)
    calibration.data_source.flow_array = "centerline_velocity"
    calibration.data_source.flow_observation_type = "velocity"
    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=calibration,
    )

    assert assembly.solver_payload["y"]["flow:INFLOW:branch0_seg0"] == [20.0]
    assert assembly.solver_payload["y"]["flow:J0:branch1_seg0"] == [18.0]
    assert assembly.solver_payload["y"]["flow:J0:branch2_seg0"] == [16.0]
    assert assembly.observation_qc["status"] == "fail"
    assert not assembly.observation_qc["checks"]["junction_mass_balance"]


def test_assemble_calibration_payload_supports_direct_flow_observations(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"

    _write_polydata(
        centerline,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        area=[2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 1, 1, 2, 2],
        paths=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        pressure=[100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
        area=[2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        extra_arrays={"flow": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0]},
    )
    _write_zerod_config(zerod)

    calibration = _calibration_config(tmp_path)
    calibration.data_source.flow_array = "flow"
    calibration.data_source.flow_observation_type = "flow"
    calibration.data_source.area_array = None

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=calibration,
    )

    assert assembly.solver_payload["y"]["flow:INFLOW:branch0_seg0"] == [10.0]
    assert assembly.solver_payload["y"]["flow:J0:branch1_seg0"] == [6.0]
    assert assembly.solver_payload["y"]["flow:J0:branch2_seg0"] == [4.0]


def test_numbered_timeseries_requires_matching_metadata(tmp_path):
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
        extra_arrays={
            "pressure_0": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "flow_0": [10.0, 10.0, 6.0, 6.0, 4.0, 4.0],
        },
    )
    _write_zerod_config(zerod)

    calibration = _calibration_config(tmp_path)
    calibration.data_source.flow_array = "flow"
    calibration.data_source.metadata_json = None
    with pytest.raises(ValueError, match="require data_source.metadata_json"):
        assemble_calibration_payload(zerod_config_path=str(zerod), calibration=calibration)


def test_mean_resistance_map_is_not_a_timeseries_source(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "resistance_map_mean.vtp"
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
    calibration = _calibration_config(tmp_path)
    calibration.data_source.mapped_centerline_result = str(mapped)
    with pytest.raises(ValueError, match="not an ordered timeseries"):
        assemble_calibration_payload(zerod_config_path=str(zerod), calibration=calibration)


def test_assemble_calibration_payload_derives_periodic_dy_for_timeseries(tmp_path):
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
        extra_arrays={
            "pressure_0": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "pressure_1": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "pressure_2": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "pressure_3": [100.0, 90.0, 90.0, 80.0, 90.0, 70.0],
            "flow_0": [0.0, 0.0, 6.0, 6.0, 4.0, 4.0],
            "flow_1": [1.0, 1.0, 6.0, 6.0, 4.0, 4.0],
            "flow_2": [0.0, 0.0, 6.0, 6.0, 4.0, 4.0],
            "flow_3": [-1.0, -1.0, 6.0, 6.0, 4.0, 4.0],
        },
    )
    _write_timeseries_metadata(
        tmp_path / "mapped_metadata.json", frame_count=4, flow_array="flow"
    )

    payload = {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                    "bc_values": {"Q": [0.0, 1.0, 0.0, -1.0, 0.0], "t": [0.0, 0.5, 1.0, 1.5, 2.0]},
            },
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
            "number_of_time_pts_per_cardiac_cycle": 4,
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    zerod.write_text(json.dumps(payload), encoding="utf-8")

    calibration = _calibration_config(tmp_path)
    calibration.data_source.flow_array = "flow"
    calibration.data_source.flow_observation_type = "flow"
    calibration.data_source.area_array = None

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=calibration,
    )

    assert assembly.solver_payload["dy"]["flow:INFLOW:branch0_seg0"] == [4.0, 0.0, -4.0, 0.0]
    assert assembly.solver_payload["dy"]["pressure:INFLOW:branch0_seg0"] == [0.0, 0.0, 0.0, 0.0]


def test_assemble_calibration_payload_supports_multiple_segments_per_branch(tmp_path):
    centerline = tmp_path / "centerline.vtp"
    mapped = tmp_path / "mapped.vtp"
    zerod = tmp_path / "zerod.json"

    _write_polydata(
        centerline,
        branch_ids=[0, 0, 0, 0, 0, 0],
        paths=[0.0, 0.1635, 0.3, 0.5439, 0.8, 0.9809],
    )
    _write_polydata(
        mapped,
        branch_ids=[0, 0, 0, 0, 0, 0],
        paths=[0.0, 0.1635, 0.3, 0.5439, 0.8, 0.9809],
        pressure=[100.0, 98.0, 96.0, 94.0, 92.0, 90.0],
        flow=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    )

    payload = {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": [1.0], "t": [0.0]}},
            {"bc_name": "OUT", "bc_type": "RESISTANCE", "bc_values": {"R": 1.0}},
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [0],
                "outlet_vessels": [1],
            },
            {
                "junction_name": "J1",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [1],
                "outlet_vessels": [2],
            },
        ],
        "simulation_parameters": {},
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "branch0_seg0",
                "vessel_length": 0.1635,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 1.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch0_seg1",
                "vessel_length": 0.3804,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 1.0},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch0_seg2",
                "vessel_length": 0.4370,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 1.0},
                "boundary_conditions": {"outlet": "OUT"},
            },
        ],
    }
    zerod.write_text(json.dumps(payload), encoding="utf-8")

    assembly = assemble_calibration_payload(
        zerod_config_path=str(zerod),
        calibration=CalibrationConfig(
            data_source=CalibrationDataSourceConfig(
                mode="mapped_centerline",
                mapped_centerline_result=str(mapped),
                centerline=str(centerline),
                flow_array="velocity",
                flow_observation_type="flow",
            ),
            parameters=CalibrationParametersConfig(),
        ),
    )

    assert assembly.observation_count == 1
    assert assembly.solver_payload["y"]["pressure:INFLOW:branch0_seg0"] == [98.0]
    assert assembly.solver_payload["y"]["pressure:branch0_seg0:J0"] == [98.0]
    assert assembly.solver_payload["y"]["pressure:J0:branch0_seg1"] == [98.0]
    assert assembly.solver_payload["y"]["pressure:branch0_seg1:J1"] == [94.0]
    assert assembly.solver_payload["y"]["pressure:J1:branch0_seg2"] == [94.0]
    assert assembly.solver_payload["y"]["pressure:branch0_seg2:OUT"] == [92.0]
    assert assembly.interface_sampling["branch0_seg1:upstream"] == {
        "interface_kind": "internal",
        "quality_status": "interpolated_internal",
        "requested_path": 0.1635,
        "selected_path": 0.1635,
        "inset_distance": 0.0,
        "usable_sample_count": 6,
        "paired_observation": True,
        "excluded_from_calibration": False,
    }


def test_assemble_calibration_payload_rejects_unavailable_junction_parameters(tmp_path):
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

    payload = {
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
                "zero_d_element_values": {"R_poiseuille": 10.0, "C": 0.1, "L": 0.01, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"inlet": "INFLOW"},
            },
            {
                "vessel_id": 1,
                "vessel_name": "branch1_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 20.0, "C": 0.2, "L": 0.02, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT1"},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 30.0, "C": 0.3, "L": 0.03, "stenosis_coefficient": 0.0},
                "boundary_conditions": {"outlet": "OUT2"},
            },
        ],
    }
    zerod.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selects unavailable parameters for J0"):
        assemble_calibration_payload(
            zerod_config_path=str(zerod),
            calibration=CalibrationConfig(
            data_source=CalibrationDataSourceConfig(
                mode="mapped_centerline",
                mapped_centerline_result=str(mapped),
                centerline=str(centerline),
                flow_array="velocity",
                flow_observation_type="flow",
            ),
                parameters=CalibrationParametersConfig(
                    junctions=CalibrationParameterSelectionConfig(default=["R_poiseuille"]),
                ),
            ),
        )


def test_assemble_calibration_payload_rejects_nonfinite_input_parameters(tmp_path):
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

    payload = json.loads(zerod.read_text(encoding="utf-8"))
    payload["vessels"][0]["zero_d_element_values"]["C"] = float("inf")
    zerod.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(ValueError, match="contains non-finite numeric values incompatible with stage-1 calibration"):
        assemble_calibration_payload(
            zerod_config_path=str(zerod),
            calibration=_calibration_config(tmp_path),
        )
