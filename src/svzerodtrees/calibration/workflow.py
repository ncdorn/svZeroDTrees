from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .._pysvzerod import (
    calibrate_pysvzerod,
    clear_calibration_provenance,
    last_calibration_provenance,
)
from ..config import CalibrationConfig

_VESSEL_NAME_RE = re.compile(r"^branch(?P<branch_id>\d+)_seg(?P<seg_id>\d+)$")
_PATH_TOLERANCE_ABS = 1e-3


@dataclass
class BranchSeries:
    paths: np.ndarray
    pressure_series: list[np.ndarray]
    flow_series: list[np.ndarray]


@dataclass
class VesselTopology:
    branch_id: int
    seg_id: int
    start_path: float
    end_path: float


@dataclass
class CalibrationAssembly:
    solver_payload: Dict[str, Any]
    observation_count: int
    variable_count: int
    input_normalization: Dict[str, Any]


def _read_polydata(path: str | Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())
    if poly.GetNumberOfPoints() == 0:
        raise ValueError(f"polydata file contains no points: {path}")
    return poly


def _point_array(poly: vtk.vtkPolyData, name: str) -> np.ndarray:
    array = poly.GetPointData().GetArray(name)
    if array is None:
        raise ValueError(f"required point-data array '{name}' was not found")
    values = np.asarray(vtk_to_numpy(array))
    if values.ndim != 1:
        raise ValueError(f"point-data array '{name}' must be scalar for stage-1 calibration")
    return values.astype(np.float64, copy=False)


def _point_array_or_none(poly: vtk.vtkPolyData, name: str) -> np.ndarray | None:
    array = poly.GetPointData().GetArray(name)
    if array is None:
        return None
    values = np.asarray(vtk_to_numpy(array))
    if values.ndim != 1:
        raise ValueError(f"point-data array '{name}' must be scalar for stage-1 calibration")
    return values.astype(np.float64, copy=False)


def _series_arrays(poly: vtk.vtkPolyData, base_name: str) -> list[tuple[int, np.ndarray]]:
    exact = _point_array_or_none(poly, base_name)
    if exact is not None:
        return [(0, exact)]

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    point_data = poly.GetPointData()
    matches: list[tuple[int, np.ndarray]] = []
    for idx in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(idx)
        if array is None:
            continue
        name = array.GetName()
        if name is None:
            continue
        match = pattern.match(name)
        if match is None:
            continue
        values = np.asarray(vtk_to_numpy(array))
        if values.ndim != 1:
            raise ValueError(f"point-data array '{name}' must be scalar for stage-1 calibration")
        matches.append((int(match.group(1)), values.astype(np.float64, copy=False)))

    if not matches:
        raise ValueError(
            f"required point-data array '{base_name}' or numbered series '{base_name}_<index>' was not found"
        )

    matches.sort(key=lambda item: item[0])
    observed = [index for index, _values in matches]
    expected = list(range(matches[-1][0] + 1))
    if observed != expected:
        raise ValueError(
            f"point-data series '{base_name}_<index>' must be contiguous from 0; found indices {observed}"
        )
    return matches


def _require_finite_series(values: Iterable[float], *, label: str) -> list[float]:
    resolved = [float(value) for value in values]
    if any(not np.isfinite(value) for value in resolved):
        raise ValueError(f"{label} contains non-finite values")
    return resolved


def _branch_series_from_mapped_centerline(
    *,
    centerline_path: str,
    mapped_centerline_path: str,
    pressure_array: str,
    flow_array: str,
    flow_observation_type: str,
    area_array: str | None,
    branch_id_array: str,
    path_array: str,
) -> tuple[Dict[int, BranchSeries], int]:
    centerline_poly = _read_polydata(centerline_path)
    mapped_poly = _read_polydata(mapped_centerline_path)

    if centerline_poly.GetNumberOfPoints() != mapped_poly.GetNumberOfPoints():
        raise ValueError(
            "mapped centerline result must have the same number of points as the provided centerline in stage 1"
        )

    branch_ids = _point_array_or_none(mapped_poly, branch_id_array)
    if branch_ids is None:
        branch_ids = _point_array(centerline_poly, branch_id_array)
    paths = _point_array_or_none(mapped_poly, path_array)
    if paths is None:
        paths = _point_array(centerline_poly, path_array)

    pressure_series = dict(_series_arrays(mapped_poly, pressure_array))
    flow_series = dict(_series_arrays(mapped_poly, flow_array))
    if set(pressure_series) != set(flow_series):
        raise ValueError(
            "mapped centerline result pressure and flow series must use matching observation indices"
        )
    if flow_observation_type == "velocity":
        if area_array is None:
            raise ValueError("area_array is required when flow observations are derived from velocity")
        areas = _point_array_or_none(mapped_poly, area_array)
        if areas is None:
            areas = _point_array(centerline_poly, area_array)
        if not np.isfinite(areas).all():
            raise ValueError(f"area samples for '{area_array}' contain non-finite values")
        flow_series = {
            observation_index: np.asarray(values * areas, dtype=np.float64)
            for observation_index, values in flow_series.items()
        }
    observation_indices = sorted(pressure_series)

    branch_series_by_id: Dict[int, BranchSeries] = {}
    unique_branch_ids = np.unique(branch_ids[np.isfinite(branch_ids)])
    for raw_branch_id in unique_branch_ids:
        branch_id = int(round(float(raw_branch_id)))
        if branch_id < 0:
            continue

        indices = np.flatnonzero(np.isclose(branch_ids, raw_branch_id))
        if indices.size == 0:
            continue

        branch_paths = np.asarray(paths[indices], dtype=np.float64)
        order = np.argsort(branch_paths, kind="mergesort")
        sorted_indices = indices[order]
        sorted_paths = branch_paths[order]
        unique_paths, inverse = np.unique(sorted_paths, return_inverse=True)
        if unique_paths.size == 0:
            raise ValueError(f"branch {branch_id} has no mapped path samples")

        reduced_pressure_series: list[np.ndarray] = []
        reduced_flow_series: list[np.ndarray] = []
        for observation_index in observation_indices:
            pressure_values = np.asarray(pressure_series[observation_index][sorted_indices], dtype=np.float64)
            flow_values = np.asarray(flow_series[observation_index][sorted_indices], dtype=np.float64)
            if pressure_values.size == 0 or flow_values.size == 0:
                raise ValueError(f"branch {branch_id} has no mapped observation samples")
            if not np.isfinite(pressure_values).all():
                raise ValueError(
                    f"pressure samples for branch {branch_id} observation {observation_index} contain non-finite values"
                )
            if not np.isfinite(flow_values).all():
                raise ValueError(
                    f"flow samples for branch {branch_id} observation {observation_index} contain non-finite values"
                )

            pressure_sum = np.zeros(unique_paths.size, dtype=np.float64)
            flow_sum = np.zeros(unique_paths.size, dtype=np.float64)
            counts = np.zeros(unique_paths.size, dtype=np.float64)
            np.add.at(pressure_sum, inverse, pressure_values)
            np.add.at(flow_sum, inverse, flow_values)
            np.add.at(counts, inverse, 1.0)
            reduced_pressure_series.append(pressure_sum / counts)
            reduced_flow_series.append(flow_sum / counts)

        branch_series_by_id[branch_id] = BranchSeries(
            paths=unique_paths,
            pressure_series=reduced_pressure_series,
            flow_series=reduced_flow_series,
        )

    if not branch_series_by_id:
        raise ValueError("no branch observations could be assembled from the mapped centerline result")
    return branch_series_by_id, len(observation_indices)


def _branch_and_segment_for_vessel(vessel_name: str) -> tuple[int, int]:
    match = _VESSEL_NAME_RE.match(vessel_name)
    if match is None:
        raise ValueError(
            "stage-1 calibration requires vessel names of the form 'branch<id>_seg<id>'; "
            f"received '{vessel_name}'"
        )
    return int(match.group("branch_id")), int(match.group("seg_id"))


def _network_topology(config: Dict[str, Any]) -> tuple[Dict[str, VesselTopology], Dict[str, str], Dict[str, str]]:
    vessel_id_to_name: Dict[int, str] = {}
    upstream_names: Dict[str, str] = {}
    downstream_names: Dict[str, str] = {}
    branch_vessels: Dict[int, list[tuple[int, str, Dict[str, Any]]]] = {}

    for vessel in config.get("vessels", []) or []:
        vessel_name = str(vessel["vessel_name"])
        vessel_id = int(vessel["vessel_id"])
        branch_id, seg_id = _branch_and_segment_for_vessel(vessel_name)

        vessel_id_to_name[vessel_id] = vessel_name
        branch_vessels.setdefault(branch_id, []).append((seg_id, vessel_name, vessel))

        boundary_conditions = vessel.get("boundary_conditions") or {}
        if boundary_conditions.get("inlet"):
            upstream_names[vessel_name] = str(boundary_conditions["inlet"])
        if boundary_conditions.get("outlet"):
            downstream_names[vessel_name] = str(boundary_conditions["outlet"])

    vessel_topology: Dict[str, VesselTopology] = {}
    for branch_id, branch_entries in branch_vessels.items():
        branch_entries.sort(key=lambda item: item[0])
        seg_ids = [seg_id for seg_id, _name, _vessel in branch_entries]
        expected = list(range(len(branch_entries)))
        if seg_ids != expected:
            raise ValueError(
                f"branch {branch_id} vessel segments must be contiguous from seg0; found {seg_ids}"
            )

        cumulative_path = 0.0
        for seg_id, vessel_name, vessel in branch_entries:
            if "vessel_length" not in vessel:
                raise ValueError(
                    f"stage-1 calibration requires vessel_length for multi-segment mapping ({vessel_name})"
                )
            vessel_length = float(vessel["vessel_length"])
            if not np.isfinite(vessel_length) or vessel_length <= 0.0:
                raise ValueError(f"{vessel_name} vessel_length must be positive and finite")
            start_path = cumulative_path
            cumulative_path += vessel_length
            vessel_topology[vessel_name] = VesselTopology(
                branch_id=branch_id,
                seg_id=seg_id,
                start_path=start_path,
                end_path=cumulative_path,
            )

    for junction in config.get("junctions", []) or []:
        junction_name = str(junction["junction_name"])
        for vessel_id in junction.get("inlet_vessels", []) or []:
            vessel_name = vessel_id_to_name[int(vessel_id)]
            downstream_names[vessel_name] = junction_name
        for vessel_id in junction.get("outlet_vessels", []) or []:
            vessel_name = vessel_id_to_name[int(vessel_id)]
            upstream_names[vessel_name] = junction_name

    missing_upstream = sorted(name for name in vessel_topology if name not in upstream_names)
    missing_downstream = sorted(name for name in vessel_topology if name not in downstream_names)
    if missing_upstream or missing_downstream:
        details = []
        if missing_upstream:
            details.append(f"missing upstream connection for {missing_upstream}")
        if missing_downstream:
            details.append(f"missing downstream connection for {missing_downstream}")
        raise ValueError("; ".join(details))

    return vessel_topology, upstream_names, downstream_names


def _sample_series_at_position(
    branch_series: BranchSeries,
    position: float,
    *,
    label: str,
) -> tuple[list[float], list[float]]:
    if position < 0.0:
        raise ValueError(f"{label} cannot be negative")

    min_path = float(branch_series.paths[0])
    max_path = float(branch_series.paths[-1])
    tolerance = max(_PATH_TOLERANCE_ABS, 1e-6 * max(1.0, abs(max_path)))
    if position < min_path - tolerance or position > max_path + tolerance:
        raise ValueError(
            f"{label}={position} lies outside mapped branch path range [{min_path}, {max_path}]"
        )

    clamped_position = min(max(position, min_path), max_path)
    pressure_values = [
        float(series[0]) if branch_series.paths.size == 1 else float(np.interp(clamped_position, branch_series.paths, series))
        for series in branch_series.pressure_series
    ]
    flow_values = [
        float(series[0]) if branch_series.paths.size == 1 else float(np.interp(clamped_position, branch_series.paths, series))
        for series in branch_series.flow_series
    ]
    return (
        _require_finite_series(pressure_values, label=f"{label} pressure samples"),
        _require_finite_series(flow_values, label=f"{label} flow samples"),
    )


def _selected_parameters(
    block_names: Iterable[str],
    *,
    default: list[str],
    overrides: Dict[str, list[str]],
    context: str,
) -> Dict[str, list[str]]:
    valid_names = {str(name) for name in block_names}
    unknown = sorted(set(overrides) - valid_names)
    if unknown:
        raise ValueError(f"{context}.overrides references unknown blocks: {unknown}")
    return {name: list(overrides.get(name, default)) for name in valid_names}


def _validate_selected_block_parameters(
    blocks: Iterable[Dict[str, Any]],
    *,
    name_key: str,
    values_key: str,
    selected_parameters: Dict[str, list[str]],
    context: str,
) -> None:
    for block in blocks:
        block_name = str(block[name_key])
        available = set((block.get(values_key) or {}).keys())
        selected = selected_parameters.get(block_name, [])
        unknown = sorted(name for name in selected if name not in available)
        if unknown:
            raise ValueError(
                f"{context} selects unavailable parameters for {block_name}: {unknown}; "
                f"available parameters: {sorted(available)}"
            )


def _observation_dt_from_inflow(
    solver_config: Dict[str, Any],
    *,
    observation_count: int,
) -> float | None:
    if observation_count <= 1:
        return None

    inflow_bc = None
    for boundary_condition in solver_config.get("boundary_conditions", []) or []:
        if str(boundary_condition.get("bc_name", "")) == "INFLOW":
            inflow_bc = boundary_condition
            break
    if inflow_bc is None:
        raise ValueError(
            "timeseries calibration requires an INFLOW boundary condition with bc_values.t to derive dy observations"
        )

    bc_values = inflow_bc.get("bc_values") or {}
    time_values = bc_values.get("t")
    if not isinstance(time_values, list) or len(time_values) < 2:
        raise ValueError(
            "timeseries calibration requires INFLOW bc_values.t with at least two time points to derive dy observations"
        )

    times = np.asarray(time_values, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("INFLOW bc_values.t contains non-finite values")

    cycle_period = float(times[-1] - times[0])
    if cycle_period <= 0.0:
        raise ValueError("INFLOW bc_values.t must span a positive cycle period")
    return cycle_period / float(observation_count)


def _periodic_derivative(values: list[float], dt: float | None) -> list[float]:
    if dt is None or len(values) <= 1:
        return [0.0] * len(values)
    if dt <= 0.0:
        raise ValueError("observation dt must be positive")

    samples = np.asarray(values, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("cannot derive dy from non-finite observation values")
    if samples.size == 2:
        return [0.0, 0.0]

    derivative = (np.roll(samples, -1) - np.roll(samples, 1)) / (2.0 * dt)
    return [float(value) for value in derivative]


def _collect_nonfinite_paths(value: Any, *, path: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            paths.extend(_collect_nonfinite_paths(nested, path=child_path))
        return paths
    if isinstance(value, (list, tuple)):
        paths: list[str] = []
        for index, nested in enumerate(value):
            child_path = f"{path}[{index}]"
            paths.extend(_collect_nonfinite_paths(nested, path=child_path))
        return paths
    if isinstance(value, np.ndarray):
        return _collect_nonfinite_paths(value.tolist(), path=path)
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return [path or "<root>"]
    return []


def _normalize_calibration_input(
    solver_config: Dict[str, Any],
    *,
    infinite_vessel_compliance: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the one supported non-finite input normalization policy."""
    if infinite_vessel_compliance not in {"error", "zero"}:
        raise ValueError(
            "calibration.input_normalization.infinite_vessel_compliance "
            "must be one of error|zero"
        )

    normalized = copy.deepcopy(solver_config)
    changed_paths: list[str] = []
    for vessel_index, vessel in enumerate(normalized.get("vessels", []) or []):
        values = vessel.get("zero_d_element_values") or {}
        compliance = values.get("C")
        if isinstance(compliance, (float, np.floating)) and np.isposinf(compliance):
            path = (
                f"vessels[{vessel_index}].zero_d_element_values.C"
            )
            if infinite_vessel_compliance == "zero":
                values["C"] = 0.0
                changed_paths.append(path)

    nonfinite_paths = _collect_nonfinite_paths(normalized)
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:5])
        remainder = len(nonfinite_paths) - min(len(nonfinite_paths), 5)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "zerod_config contains non-finite numeric values incompatible with "
            "stage-1 calibration; only positive "
            "infinity at vessels[*].zero_d_element_values.C may be normalized "
            f"with input_normalization.infinite_vessel_compliance: zero. "
            f"Invalid paths: {preview}{suffix}"
        )

    return normalized, {
        "infinite_vessel_compliance": infinite_vessel_compliance,
        "changed_count": len(changed_paths),
        "changed_paths": changed_paths,
    }


def assemble_calibration_payload(
    *,
    zerod_config_path: str,
    calibration: CalibrationConfig,
) -> CalibrationAssembly:
    with open(zerod_config_path, "r", encoding="utf-8") as stream:
        solver_config = json.load(stream)
    solver_config, input_normalization = _normalize_calibration_input(
        solver_config,
        infinite_vessel_compliance=(
            calibration.input_normalization.infinite_vessel_compliance
        ),
    )

    branch_series_by_id, observation_count = _branch_series_from_mapped_centerline(
        centerline_path=calibration.data_source.centerline or "",
        mapped_centerline_path=calibration.data_source.mapped_centerline_result or "",
        pressure_array=calibration.data_source.pressure_array,
        flow_array=calibration.data_source.flow_array,
        flow_observation_type=calibration.data_source.flow_observation_type,
        area_array=calibration.data_source.area_array,
        branch_id_array=calibration.data_source.branch_id_array,
        path_array=calibration.data_source.path_array,
    )
    vessel_topology, upstream_names, downstream_names = _network_topology(solver_config)
    observation_dt = _observation_dt_from_inflow(
        solver_config,
        observation_count=observation_count,
    )

    y: Dict[str, list[float]] = {}
    for vessel_name, topology in vessel_topology.items():
        if topology.branch_id not in branch_series_by_id:
            raise ValueError(
                f"mapped centerline result does not contain observations for branch {topology.branch_id} ({vessel_name})"
            )

        branch_series = branch_series_by_id[topology.branch_id]
        upstream_name = upstream_names[vessel_name]
        downstream_name = downstream_names[vessel_name]
        proximal_pressure, proximal_flow = _sample_series_at_position(
            branch_series,
            topology.start_path,
            label=f"{vessel_name} upstream interface",
        )
        distal_pressure, distal_flow = _sample_series_at_position(
            branch_series,
            topology.end_path,
            label=f"{vessel_name} downstream interface",
        )

        variables = {
            f"flow:{upstream_name}:{vessel_name}": proximal_flow,
            f"pressure:{upstream_name}:{vessel_name}": proximal_pressure,
            f"flow:{vessel_name}:{downstream_name}": distal_flow,
            f"pressure:{vessel_name}:{downstream_name}": distal_pressure,
        }
        for variable_name, values in variables.items():
            if len(values) != observation_count:
                raise ValueError(f"observation series length mismatch for '{variable_name}'")
            y[variable_name] = [float(value) for value in values]

    dy = {
        variable_name: _periodic_derivative(values, observation_dt)
        for variable_name, values in y.items()
    }

    vessel_parameters = _selected_parameters(
        (str(vessel["vessel_name"]) for vessel in solver_config.get("vessels", []) or []),
        default=calibration.parameters.vessels.default,
        overrides=calibration.parameters.vessels.overrides,
        context="calibration.parameters.vessels",
    )
    junction_parameters = _selected_parameters(
        (str(junction["junction_name"]) for junction in solver_config.get("junctions", []) or []),
        default=calibration.parameters.junctions.default,
        overrides=calibration.parameters.junctions.overrides,
        context="calibration.parameters.junctions",
    )
    _validate_selected_block_parameters(
        solver_config.get("vessels", []) or [],
        name_key="vessel_name",
        values_key="zero_d_element_values",
        selected_parameters=vessel_parameters,
        context="calibration.parameters.vessels",
    )
    _validate_selected_block_parameters(
        solver_config.get("junctions", []) or [],
        name_key="junction_name",
        values_key="junction_values",
        selected_parameters=junction_parameters,
        context="calibration.parameters.junctions",
    )

    payload = json.loads(json.dumps(solver_config))
    payload["y"] = y
    payload["dy"] = dy
    payload["calibration_parameters"] = {
        "initial_damping_factor": float(calibration.solver.initial_damping_factor),
        "maximum_iterations": int(calibration.solver.maximum_iterations),
        "tolerance_gradient": float(calibration.solver.tolerance_gradient),
        "tolerance_increment": float(calibration.solver.tolerance_increment),
    }

    for vessel in payload.get("vessels", []) or []:
        vessel_name = str(vessel["vessel_name"])
        vessel["calibrate"] = list(vessel_parameters.get(vessel_name, []))
    for junction in payload.get("junctions", []) or []:
        junction_name = str(junction["junction_name"])
        junction["calibrate"] = list(junction_parameters.get(junction_name, []))

    return CalibrationAssembly(
        solver_payload=payload,
        observation_count=observation_count,
        variable_count=len(y),
        input_normalization=input_normalization,
    )


def calibrate_0d_from_mapped_centerline(
    *,
    zerod_config_path: str,
    output_config_path: str,
    calibration: CalibrationConfig,
) -> Dict[str, Any]:
    assembly = assemble_calibration_payload(
        zerod_config_path=zerod_config_path,
        calibration=calibration,
    )
    clear_calibration_provenance()
    calibrated = calibrate_pysvzerod(assembly.solver_payload)

    nonfinite_paths = _collect_nonfinite_paths(calibrated)
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:5])
        remainder = len(nonfinite_paths) - min(len(nonfinite_paths), 5)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "calibrated svZeroD config contains non-finite values after solver calibration: "
            f"{preview}{suffix}"
        )

    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(calibrated, stream, indent=4, allow_nan=False)

    return {
        "status": "ok",
        "output_config": str(output_path),
        "observation_count": assembly.observation_count,
        "variable_count": assembly.variable_count,
        "input_normalization": assembly.input_normalization,
        "solver_provenance": last_calibration_provenance(),
    }
