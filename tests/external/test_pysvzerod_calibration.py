"""Opt-in synthetic regression against the installed upstream pysvzerod.

The case in this module is deliberately generated at runtime.  It has no
patient data and obtains its mapped observations from a direct simulation of
the same small solver configuration used by the calibration workflow.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest


pytestmark = pytest.mark.external


def _require_real_solver() -> dict[str, Any]:
    """Require the opt-in solver and skip only when the optional solver is absent."""
    if os.environ.get("SVZERODTREES_RUN_REAL_SOLVER") != "1":
        pytest.skip(
            "opt-in actual-solver regression; set "
            "SVZERODTREES_RUN_REAL_SOLVER=1 to run"
        )

    if importlib.util.find_spec("pysvzerod") is None:
        pytest.skip(
            "optional pysvzerod is absent; install the pinned external solver"
        )

    from svzerodtrees._pysvzerod import require_pysvzerod_api

    try:
        provenance = require_pysvzerod_api()
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        pytest.fail(
            "SVZERODTREES_RUN_REAL_SOLVER=1 requested an incompatible or "
            f"unloadable pysvzerod: {exc}"
        )

    if not isinstance(provenance, dict):
        pytest.fail("pysvzerod provenance must be a mapping")
    if not provenance.get("module_path"):
        pytest.fail("pysvzerod provenance did not record module_path")
    if not provenance.get("module_sha256"):
        pytest.fail("pysvzerod provenance did not record module_sha256")
    metadata = provenance.get("file_metadata")
    if not isinstance(metadata, dict) or metadata.get("readable") is not True:
        pytest.fail("pysvzerod provenance did not confirm a readable solver artifact")
    return provenance


def _synthetic_solver_config() -> dict[str, Any]:
    """Return a small pulsatile bifurcation with one inactive vessel block."""
    return {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {
                    "Q": [10.0, 11.0, 10.0, 10.0],
                    "t": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                },
            },
            {
                "bc_name": "OUT1",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 100.0, "Pd": 0.0},
            },
            {
                "bc_name": "OUT2",
                "bc_type": "RESISTANCE",
                "bc_values": {"R": 100.0, "Pd": 0.0},
            },
        ],
        "junctions": [
            {
                "junction_name": "J0",
                "junction_type": "internal_junction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
            }
        ],
        "simulation_parameters": {
            "density": 1.06,
            "viscosity": 0.04,
            "number_of_cardiac_cycles": 1,
            "number_of_time_pts_per_cardiac_cycle": 4,
            "output_all_cycles": False,
            "output_mean_only": False,
        },
        "vessels": [
            _vessel("branch0_seg0", 0, 10.0, inlet="INFLOW"),
            _vessel("branch1_seg0", 1, 20.0, outlet="OUT1"),
            _vessel("branch2_seg0", 2, 30.0, outlet="OUT2"),
        ],
    }


def _vessel(
    name: str,
    vessel_id: int,
    resistance: float,
    *,
    inlet: str | None = None,
    outlet: str | None = None,
) -> dict[str, Any]:
    boundary_conditions: dict[str, str] = {}
    if inlet is not None:
        boundary_conditions["inlet"] = inlet
    if outlet is not None:
        boundary_conditions["outlet"] = outlet
    return {
        "vessel_id": vessel_id,
        "vessel_name": name,
        "vessel_length": 1.0,
        "zero_d_element_type": "BloodVessel",
        "zero_d_element_values": {
            "R_poiseuille": resistance,
            "C": 0.0,
            "L": 0.0,
            "stenosis_coefficient": 0.0,
        },
        "boundary_conditions": boundary_conditions,
    }


def _result_rows(result: Any) -> list[dict[str, Any]]:
    """Convert the standard pysvzerod table to JSON-like rows for the fixture."""
    if hasattr(result, "to_dict"):
        rows = result.to_dict(orient="records")
    elif isinstance(result, list):
        rows = result
    elif isinstance(result, dict):
        keys = list(result)
        lengths = {len(result[key]) for key in keys}
        if len(lengths) != 1:
            raise AssertionError("solver result columns have inconsistent lengths")
        rows = [
            {key: result[key][index] for key in keys}
            for index in range(next(iter(lengths)))
        ]
    else:
        raise AssertionError(f"unsupported pysvzerod result type: {type(result)!r}")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise AssertionError("pysvzerod returned no row-oriented result")
    return rows


def _series(
    rows: Iterable[dict[str, Any]], vessel: str, field: str, *, frame_count: int = 3
) -> np.ndarray:
    selected = sorted(
        (
            (float(row["time"]), float(row[field]))
            for row in rows
            if str(row.get("name")) == vessel
        ),
        key=lambda item: item[0],
    )
    if len(selected) < frame_count:
        raise AssertionError(
            f"solver result did not provide {frame_count} samples for {vessel}:{field}"
        )
    times = np.asarray([item[0] for item in selected], dtype=float)
    values = np.asarray([item[1] for item in selected], dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise AssertionError(f"solver result contains non-finite {vessel}:{field}")
    # The one-cycle solver output includes the shared endpoint at t=1.0.  The
    # mapped artifact intentionally records the first three periodic phases.
    return values[:frame_count]


def _write_synthetic_timeseries(
    path: Path,
    *,
    values: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write a centerline and mapped centerline with explicit branch metadata."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    branch_ids: list[float] = []
    paths: list[float] = []
    point_rows: list[list[np.ndarray]] = []
    for branch_id in range(3):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(3)
        for point_index, position in enumerate((0.0, 0.5, 1.0)):
            index = points.InsertNextPoint(position, float(branch_id), 0.0)
            line.GetPointIds().SetId(point_index, index)
            branch_ids.append(float(branch_id))
            paths.append(position)
        lines.InsertNextCell(line)

        branch = values[f"branch{branch_id}_seg0"]
        pressure_in = branch["pressure_in"]
        pressure_out = branch["pressure_out"]
        flow_in = branch["flow_in"]
        flow_out = branch["flow_out"]
        if branch_id == 0:
            pressure = np.column_stack((pressure_in, pressure_in, pressure_out))
            flow = np.column_stack((flow_in, flow_in, flow_out))
        else:
            pressure = np.column_stack((pressure_in, pressure_out, pressure_out))
            flow = np.column_stack((flow_in, flow_out, flow_out))
        point_rows.append([pressure, flow])

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    def add_array(name: str, array: np.ndarray) -> None:
        vtk_array = numpy_to_vtk(np.asarray(array, dtype=np.float64), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)

    add_array("BranchId", np.asarray(branch_ids))
    add_array("Path", np.asarray(paths))
    for frame_index in range(3):
        add_array(
            f"pressure_{frame_index}",
            np.concatenate(
                [point_rows[branch_id][0][:, frame_index] for branch_id in range(3)]
            ),
        )
        add_array(
            f"flow_{frame_index}",
            np.concatenate(
                [point_rows[branch_id][1][:, frame_index] for branch_id in range(3)]
            ),
        )

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise AssertionError(f"failed to write synthetic mapped centerline: {path}")


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "centerline_timeseries_last_cycle",
                "frame_count": 3,
                "point_count": 9,
                "frame_indices": [0, 1, 2],
                "timestamps_s": [0.0, 1.0 / 3.0, 2.0 / 3.0],
                "cycle_duration_s": 1.0,
                "data_contract": {
                    # The synthetic mapped values intentionally retain the
                    # native scalar values returned by the solver.  The
                    # workflow's replay target contract uses this explicit
                    # identity unit for its synthetic comparison.
                    "pressure": {"quantity": "pressure", "units": "mmHg"},
                    "flow": {
                        "quantity": "volumetric_flow",
                        "units": "cm^3/s",
                    },
                },
                "processed_frames": [
                    {
                        "frame_index": index,
                        "time_s": index / 3.0,
                        "point_arrays": [f"pressure_{index}", f"flow_{index}"],
                    }
                    for index in range(3)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_calibration_config(
    path: Path,
    *,
    baseline: Path,
    mapped: Path,
    metadata: Path,
    output: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "workflow": "calibrate_0d_from_3d",
                "paths": {
                    "root": str(path.parent),
                    "zerod_config": str(baseline),
                    "output_config": str(output),
                },
                "calibration": {
                    "data_source": {
                        "mode": "mapped_centerline",
                        "mapped_centerline_result": str(mapped),
                        "metadata_json": str(metadata),
                        "centerline": str(mapped),
                        "pressure_array": "pressure",
                        "flow_array": "flow",
                        "flow_observation_type": "flow",
                    },
                    "parameters": {
                        "vessels": {
                            "default": ["R_poiseuille"],
                            # This block is deliberately inactive so the
                            # post-calibration contract is observable.
                            "overrides": {"branch2_seg0": []},
                        },
                        "junctions": {"default": []},
                    },
                    "solver": {
                        "confirmation_absolute_tolerance": 1.0e-8,
                        "confirmation_relative_tolerance": 1.0e-6,
                        "pressure_bound_multiplier": 100.0,
                        "flow_bound_multiplier": 100.0,
                        "cycle_stability_tolerance": 1.0e-3,
                        "replay_minimum_cycles": 3,
                        "replay_maximum_cycles": 4,
                        "required_consecutive_stable_pairs": 1,
                    },
                    "observation_qc": {"enforcement": "target_focused"},
                    "targets": {
                        "mpa_pressure": {
                            "vessel": "branch0_seg0",
                            "interface": "external_upstream",
                            "weight": 1.0,
                            "normalized_rms_tolerance": 0.05,
                        },
                        "rpa_flow_split": {
                            "rpa_vessel": "branch1_seg0",
                            "lpa_vessel": "branch2_seg0",
                            "interface": "external_downstream",
                            "weight": 1.0,
                            "absolute_tolerance": 0.02,
                        },
                        "require_improvement_over_baseline": False,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _assert_finite_solver_rows(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    assert rows
    for row in rows:
        for key in ("time", "flow_in", "flow_out", "pressure_in", "pressure_out"):
            assert key in row
            assert np.isfinite(float(row[key]))


def _assert_last_two_cycles_stable(
    rows: Iterable[dict[str, Any]], *, points_per_cycle: int
) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        vessel = str(row["name"])
        for field in ("flow_in", "flow_out", "pressure_in", "pressure_out"):
            grouped.setdefault((vessel, field), []).append(
                (float(row["time"]), float(row[field]))
            )
    for key, series in grouped.items():
        series.sort(key=lambda item: item[0])
        values = np.asarray([item[1] for item in series], dtype=float)
        assert values.size >= 2 * points_per_cycle, key
        np.testing.assert_allclose(
            values[-points_per_cycle:],
            values[-2 * points_per_cycle : -points_per_cycle],
            rtol=1.0e-5,
            atol=1.0e-8,
            err_msg=f"negative-resistance series did not settle: {key}",
        )


def test_actual_solver_synthetic_calibration_and_negative_resistance(tmp_path: Path):
    """Exercise two-pass calibration, target replay, and stable negative R."""
    provenance = _require_real_solver()

    from svzerodtrees._pysvzerod import simulate_pysvzerod
    from svzerodtrees.api import run_from_config_file

    baseline = _synthetic_solver_config()
    baseline_path = tmp_path / "synthetic_baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    baseline_rows = _result_rows(simulate_pysvzerod(copy.deepcopy(baseline)))
    _assert_finite_solver_rows(baseline_rows)
    values: dict[str, dict[str, np.ndarray]] = {}
    for vessel in ("branch0_seg0", "branch1_seg0", "branch2_seg0"):
        values[vessel] = {
            "pressure_in": _series(baseline_rows, vessel, "pressure_in"),
            "pressure_out": _series(baseline_rows, vessel, "pressure_out"),
            "flow_in": _series(baseline_rows, vessel, "flow_in"),
            "flow_out": _series(baseline_rows, vessel, "flow_out"),
        }

    mapped_path = tmp_path / "synthetic_centerline_timeseries.vtp"
    metadata_path = tmp_path / "synthetic_centerline_timeseries_metadata.json"
    _write_synthetic_timeseries(mapped_path, values=values)
    _write_metadata(metadata_path)

    output_path = tmp_path / "calibrated.json"
    config_path = tmp_path / "calibrate.json"
    _write_calibration_config(
        config_path,
        baseline=baseline_path,
        mapped=mapped_path,
        metadata=metadata_path,
        output=output_path,
    )

    result = run_from_config_file(str(config_path))
    assert result["status"] == "ok"
    assert output_path.exists()

    published = json.loads(output_path.read_text(encoding="utf-8"))
    published_rows = _result_rows(simulate_pysvzerod(copy.deepcopy(published)))
    _assert_finite_solver_rows(published_rows)

    confirmation = result["calibration_confirmation"]
    assert confirmation["converged"] is True
    assert len(confirmation["invocations"]) == 2
    assert confirmation["inactive_parameters_preserved"] is True
    assert all(
        invocation["inactive_parameter_count"] > 0
        for invocation in confirmation["invocations"]
    )
    assert result["solver_provenance"]["module_sha256"] == provenance["module_sha256"]
    assert all(
        invocation["solver_provenance"]["module_sha256"]
        == provenance["module_sha256"]
        for invocation in confirmation["invocations"]
    )

    baseline_by_name = {item["vessel_name"]: item for item in baseline["vessels"]}
    published_by_name = {item["vessel_name"]: item for item in published["vessels"]}
    inactive_baseline = baseline_by_name["branch2_seg0"]["zero_d_element_values"]
    inactive_published = published_by_name["branch2_seg0"]["zero_d_element_values"]
    assert inactive_published == inactive_baseline

    assert result["replay_stability"]["status"] == "pass"
    assert result["replay_stability"].get("accepted_final_cycle")
    target_quality = result["target_quality"]
    assert target_quality["status"] == "pass"
    assert target_quality["candidate"]["status"] == "pass"
    assert target_quality["candidate"]["gate_results"] == {
        "mpa_pressure": True,
        "rpa_flow_split": True,
    }
    assert target_quality["candidate"]["composite_score"] >= 0.0

    diagnostics = {
        "solver_provenance": provenance,
        "calibration_passes": len(confirmation["invocations"]),
        "inactive_parameter_values": inactive_published,
        "target_quality": target_quality["candidate"],
    }
    diagnostics_path = tmp_path / "solver_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    recorded = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert recorded["solver_provenance"]["module_sha256"] == provenance["module_sha256"]

    negative = copy.deepcopy(published)
    negative["vessels"][1]["zero_d_element_values"]["R_poiseuille"] = -1.0
    negative["simulation_parameters"].update(
        {
            "number_of_cardiac_cycles": 4,
            "output_all_cycles": True,
            "output_mean_only": False,
            "output_interval": 1,
        }
    )
    negative_rows = _result_rows(simulate_pysvzerod(negative))
    _assert_finite_solver_rows(negative_rows)
    _assert_last_two_cycles_stable(negative_rows, points_per_cycle=4)
    assert any(
        vessel["zero_d_element_values"]["R_poiseuille"] < 0.0
        for vessel in negative["vessels"]
    )
