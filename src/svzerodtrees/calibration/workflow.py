from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
_MIN_USABLE_INTERFACE_SAMPLES = 3
_FLOW_EPS = 1e-8


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
    interface_sampling: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    excluded_blocks: Dict[str, str] = field(default_factory=dict)
    observation_qc: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationTiming:
    timestamps_s: np.ndarray | None
    cycle_duration_s: float | None


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


def _load_timeseries_metadata(
    metadata_path: str | Path,
    *,
    mapped_poly: vtk.vtkPolyData,
    pressure_series: Dict[int, np.ndarray],
    flow_series: Dict[int, np.ndarray],
    pressure_array: str,
    flow_array: str,
    flow_observation_type: str,
) -> ObservationTiming:
    path = Path(metadata_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"timeseries metadata sidecar was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"timeseries metadata sidecar is not valid JSON: {path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("timeseries metadata sidecar must contain a JSON object")
    if metadata.get("kind") != "centerline_timeseries_last_cycle":
        raise ValueError(
            "timeseries metadata kind must be 'centerline_timeseries_last_cycle'; "
            "resistance_map_mean.vtp is not a timeseries source"
        )

    point_count = metadata.get("point_count")
    if point_count != mapped_poly.GetNumberOfPoints():
        raise ValueError(
            "timeseries metadata point_count does not match mapped centerline: "
            f"metadata={point_count}, VTP={mapped_poly.GetNumberOfPoints()}"
        )

    observed_indices = sorted(pressure_series)
    if observed_indices != sorted(flow_series):
        raise ValueError("timeseries pressure and flow arrays must use matching frame indices")
    frame_count = metadata.get("frame_count", metadata.get("selected_frame_count"))
    if frame_count != len(observed_indices):
        raise ValueError(
            "timeseries metadata frame_count does not match numbered arrays: "
            f"metadata={frame_count}, arrays={len(observed_indices)}"
        )
    expected_indices = list(range(len(observed_indices)))
    if observed_indices != expected_indices:
        raise ValueError(
            "timeseries numbered arrays must be contiguous from frame 0; "
            f"found {observed_indices}"
        )

    frame_indices = metadata.get("frame_indices")
    timestamps = metadata.get("timestamps_s")
    if frame_indices != expected_indices:
        raise ValueError(
            "timeseries metadata frame_indices must match numbered arrays "
            f"{expected_indices}"
        )
    if not isinstance(timestamps, list) or len(timestamps) != len(expected_indices):
        raise ValueError(
            "timeseries metadata timestamps_s must contain one timestamp per numbered frame"
        )
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(timestamps_array).all():
        raise ValueError("timeseries metadata timestamps_s contains non-finite values")

    expected_pressure_names = [f"{pressure_array}_{index}" for index in expected_indices]
    expected_flow_names = [f"{flow_array}_{index}" for index in expected_indices]
    frames = metadata.get("processed_frames")
    if not isinstance(frames, list) or len(frames) != len(expected_indices):
        raise ValueError(
            "timeseries metadata processed_frames must contain one record per numbered frame"
        )
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("timeseries metadata processed_frames entries must be objects")
        if frame.get("frame_index") != index or frame.get("time_s") != timestamps[index]:
            raise ValueError(
                "timeseries metadata processed_frames must preserve frame ordering and timestamps"
            )
        if frame.get("point_arrays") != [expected_pressure_names[index], expected_flow_names[index]]:
            raise ValueError(
                "timeseries metadata point_arrays does not match the numbered VTP arrays"
            )

    data_contract = metadata.get("data_contract")
    if not isinstance(data_contract, dict):
        raise ValueError("timeseries metadata data_contract is required")
    pressure_contract = data_contract.get("pressure")
    flow_contract = data_contract.get("flow")
    if not isinstance(pressure_contract, dict) or not pressure_contract.get("units"):
        raise ValueError("timeseries metadata must declare pressure units")
    if not isinstance(flow_contract, dict):
        raise ValueError("timeseries metadata must declare flow semantics and units")
    if flow_observation_type == "flow":
        if (
            flow_contract.get("quantity") != "volumetric_flow"
            or flow_contract.get("units") != "cm^3/s"
        ):
            raise ValueError(
                "timeseries metadata must declare flow quantity=volumetric_flow and units=cm^3/s"
            )
    elif flow_contract.get("quantity") != "velocity" or not flow_contract.get("units"):
        raise ValueError("velocity timeseries metadata must declare quantity=velocity and units")

    for series_name, series in (
        ("pressure", pressure_series),
        ("flow", flow_series),
    ):
        for index, values in series.items():
            if values.size != mapped_poly.GetNumberOfPoints():
                raise ValueError(
                    f"{series_name}_{index} point count does not match mapped centerline"
                )

    if len(timestamps_array) <= 1:
        return ObservationTiming(timestamps_array, None)
    cycle_duration = metadata.get("cycle_duration_s")
    if cycle_duration is None:
        raise ValueError(
            "timeseries metadata cycle_duration_s is required for periodic derivatives"
        )
    cycle_duration = float(cycle_duration)
    if not np.isfinite(cycle_duration) or cycle_duration <= 0.0:
        raise ValueError("timeseries metadata cycle_duration_s must be positive and finite")
    if np.any(np.diff(timestamps_array) <= 0.0):
        raise ValueError("timeseries metadata timestamps_s must be strictly increasing")
    intervals = np.diff(timestamps_array)
    wrap_interval = cycle_duration - float(timestamps_array[-1] - timestamps_array[0])
    if wrap_interval <= 0.0:
        raise ValueError(
            "timeseries metadata timestamps_s and cycle_duration_s do not define a positive periodic interval"
        )
    all_intervals = np.append(intervals, wrap_interval)
    if not np.allclose(all_intervals, all_intervals[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            "nonuniform timeseries timing is unsupported; recorded timestamps must be uniformly spaced"
        )
    return ObservationTiming(timestamps_array, cycle_duration)


def _require_finite_series(values: Iterable[float], *, label: str) -> list[float]:
    resolved = [float(value) for value in values]
    if any(not np.isfinite(value) for value in resolved):
        raise ValueError(f"{label} contains non-finite values")
    return resolved


def _branch_series_from_mapped_centerline(
    *,
    centerline_path: str,
    mapped_centerline_path: str,
    metadata_json: str | None,
    pressure_array: str,
    flow_array: str,
    flow_observation_type: str,
    area_array: str | None,
    branch_id_array: str,
    path_array: str,
) -> tuple[Dict[int, BranchSeries], int, ObservationTiming]:
    if flow_observation_type not in {"flow", "velocity"}:
        raise ValueError(
            "calibration.data_source.flow_observation_type must be one of flow|velocity"
        )
    centerline_poly = _read_polydata(centerline_path)
    mapped_poly = _read_polydata(mapped_centerline_path)

    mapped_name = Path(mapped_centerline_path).name
    if mapped_name in {"resistance_map_mean.vtp", "resistance_map_systolic.vtp"}:
        raise ValueError(
            f"{mapped_name} contains retained mean/systolic fields, not an ordered "
            "timeseries; use centerline_timeseries_last_cycle.vtp"
        )

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
    point_data = mapped_poly.GetPointData()
    numbered_series = any(
        (point_data.GetArrayName(index) or "").startswith(f"{pressure_array}_")
        or (point_data.GetArrayName(index) or "").startswith(f"{flow_array}_")
        for index in range(point_data.GetNumberOfArrays())
    )
    if numbered_series and flow_observation_type == "velocity" and flow_array.lower() == "velocity":
        raise ValueError(
            "velocity_0..N is the legacy svSlicer integrated-flow spelling; "
            "configure it with flow_observation_type: flow"
        )
    if numbered_series:
        if metadata_json is None:
            raise ValueError(
                "numbered pressure/flow arrays require data_source.metadata_json "
                "for frame ordering, units, and timestamps"
            )
        observation_timing = _load_timeseries_metadata(
            metadata_json,
            mapped_poly=mapped_poly,
            pressure_series=pressure_series,
            flow_series=flow_series,
            pressure_array=pressure_array,
            flow_array=flow_array,
            flow_observation_type=flow_observation_type,
        )
    else:
        observation_timing = ObservationTiming(None, None)
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
        if not np.isfinite(branch_paths).all():
            raise ValueError(f"path samples for branch {branch_id} contain non-finite values")
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
    return branch_series_by_id, len(observation_indices), observation_timing


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


def _sample_interface(
    branch_series: BranchSeries,
    requested_path: float,
    *,
    label: str,
    interface_kind: str,
    excluded: bool = False,
) -> tuple[list[float], list[float], Dict[str, Any]]:
    """Sample one paired interface and return its deterministic sampling record.

    External boundary planes are represented by the adjacent interior mapped
    cross-section.  Junction interfaces remain interpolation queries at the
    topology-derived path.  An explicitly excluded under-resolved block may
    retain an endpoint query so the solver observation payload stays complete,
    but that sample is marked as unqualified and cannot be used for calibration.
    """
    paths = np.asarray(branch_series.paths, dtype=np.float64)
    usable_sample_count = int(paths.size)
    if usable_sample_count == 0:
        raise ValueError(f"{label} has no usable mapped branch samples")

    requested_path = float(requested_path)
    if not np.isfinite(requested_path):
        raise ValueError(f"{label} requested path must be finite")

    min_path = float(paths[0])
    max_path = float(paths[-1])
    tolerance = max(_PATH_TOLERANCE_ABS, 1e-6 * max(1.0, abs(max_path)))
    if requested_path < min_path - tolerance or requested_path > max_path + tolerance:
        raise ValueError(
            f"{label}={requested_path} lies outside mapped branch path range "
            f"[{min_path}, {max_path}]"
        )

    if interface_kind not in {"external_upstream", "external_downstream", "internal"}:
        raise ValueError(f"unsupported interface kind '{interface_kind}' for {label}")

    if interface_kind == "internal":
        selected_path = min(max(requested_path, min_path), max_path)
        quality_status = "interpolated_internal"
        inset_distance = 0.0
    elif usable_sample_count < _MIN_USABLE_INTERFACE_SAMPLES:
        # Keep the observation shape complete so the pre-optimization QC gate
        # can emit a deterministic failure report. An explicitly excluded block
        # is still marked separately from an unqualified selected block.
        selected_path = min(max(requested_path, min_path), max_path)
        quality_status = "excluded_underresolved" if excluded else "underresolved"
        inset_distance = 0.0
    else:
        if interface_kind == "external_upstream":
            selected_path = float(paths[1])
        else:
            selected_path = float(paths[-2])
        quality_status = "qualified_interior"
        inset_distance = abs(selected_path - requested_path)

    pressure_values, flow_values = _sample_series_at_position(
        branch_series,
        selected_path,
        label=label,
    )
    sampling = {
        "interface_kind": interface_kind,
        "quality_status": quality_status,
        "requested_path": requested_path,
        "selected_path": float(selected_path),
        "inset_distance": float(inset_distance),
        "usable_sample_count": usable_sample_count,
        "paired_observation": True,
        "excluded_from_calibration": bool(excluded),
    }
    return pressure_values, flow_values, sampling


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


def _periodic_derivative(
    values: list[float],
    timing: ObservationTiming,
) -> list[float]:
    if timing.timestamps_s is None or len(values) <= 1:
        return [0.0] * len(values)

    timestamps = timing.timestamps_s
    intervals = np.diff(timestamps)
    wrap_interval = float(timing.cycle_duration_s) - float(timestamps[-1] - timestamps[0])
    dt = float(intervals[0]) if intervals.size else wrap_interval
    if dt <= 0.0 or wrap_interval <= 0.0:
        raise ValueError("recorded timeseries timestamps must define positive intervals")

    samples = np.asarray(values, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("cannot derive dy from non-finite observation values")
    if samples.size == 2:
        return [0.0, 0.0]

    derivative = (np.roll(samples, -1) - np.roll(samples, 1)) / (2.0 * dt)
    return [float(value) for value in derivative]


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _relative_rms_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(_rms(left), _rms(right), _FLOW_EPS)
    return _rms(np.asarray(left) - np.asarray(right)) / scale


def _flow_boundary_condition(config: Dict[str, Any]) -> Dict[str, Any] | None:
    for boundary_condition in config.get("boundary_conditions", []) or []:
        if str(boundary_condition.get("bc_type", "")).upper() == "FLOW":
            return boundary_condition
    return None


def _reference_inflow_series(
    config: Dict[str, Any],
    *,
    observation_count: int,
    timing: ObservationTiming,
) -> tuple[np.ndarray | None, str | None]:
    boundary_condition = _flow_boundary_condition(config)
    if boundary_condition is None:
        return None, "no FLOW boundary condition was found"

    values = np.asarray(
        (boundary_condition.get("bc_values") or {}).get("Q", []),
        dtype=np.float64,
    )
    if values.size == 0 or not np.isfinite(values).all():
        return None, "FLOW boundary condition Q values are missing or non-finite"
    if observation_count == 1:
        return np.asarray([float(values[0])]), None
    if timing.timestamps_s is None:
        return None, "recorded timestamps are required for root waveform QC"

    times = np.asarray(
        (boundary_condition.get("bc_values") or {}).get("t", []),
        dtype=np.float64,
    )
    if times.size != values.size or times.size < 2 or not np.isfinite(times).all():
        return None, "FLOW boundary condition Q and t values must be finite and paired"
    if np.any(np.diff(times) <= 0.0):
        return None, "FLOW boundary condition t values must be strictly increasing"
    period = float(times[-1] - times[0])
    if period <= 0.0:
        return None, "FLOW boundary condition t values must span a positive period"

    phases = np.mod(np.asarray(timing.timestamps_s) - timing.timestamps_s[0], period)
    reference = np.interp(phases, times - times[0], values)
    return np.asarray(reference, dtype=np.float64), None


def _build_observation_qc(
    *,
    solver_config: Dict[str, Any],
    y: Dict[str, list[float]],
    observation_count: int,
    observation_timing: ObservationTiming,
    vessel_topology: Dict[str, VesselTopology],
    upstream_names: Dict[str, str],
    downstream_names: Dict[str, str],
    branch_series_by_id: Dict[int, BranchSeries],
    interface_sampling: Dict[str, Dict[str, Any]],
    excluded_blocks: Dict[str, str],
    qc_config: Any,
) -> Dict[str, Any]:
    vessel_continuity: Dict[str, float] = {}
    pressure_directions: Dict[str, Dict[str, Any]] = {}
    path_coverage: Dict[str, float] = {}

    for vessel_name, topology in sorted(vessel_topology.items()):
        proximal_flow = np.asarray(
            y[f"flow:{upstream_names[vessel_name]}:{vessel_name}"],
            dtype=np.float64,
        )
        distal_flow = np.asarray(
            y[f"flow:{vessel_name}:{downstream_names[vessel_name]}"],
            dtype=np.float64,
        )
        vessel_continuity[vessel_name] = _relative_rms_difference(
            proximal_flow,
            distal_flow,
        )

        proximal_pressure = np.asarray(
            y[f"pressure:{upstream_names[vessel_name]}:{vessel_name}"],
            dtype=np.float64,
        )
        distal_pressure = np.asarray(
            y[f"pressure:{vessel_name}:{downstream_names[vessel_name]}"],
            dtype=np.float64,
        )
        mean_flow = float(np.mean(proximal_flow))
        mean_pressure_drop = float(np.mean(proximal_pressure - distal_pressure))
        if abs(mean_flow) <= _FLOW_EPS:
            direction = "indeterminate"
            direction_passed = True
        else:
            direction = "positive" if mean_pressure_drop * mean_flow > 0.0 else "wrong"
            direction_passed = direction == "positive"
        pressure_directions[vessel_name] = {
            "mean_flow": mean_flow,
            "mean_pressure_drop": mean_pressure_drop,
            "direction": direction,
            "passed": direction_passed,
        }

        branch_series = branch_series_by_id[topology.branch_id]
        requested_length = topology.end_path - topology.start_path
        overlap_start = max(topology.start_path, float(branch_series.paths[0]))
        overlap_end = min(topology.end_path, float(branch_series.paths[-1]))
        covered_length = max(0.0, overlap_end - overlap_start)
        path_coverage[vessel_name] = (
            covered_length / requested_length if requested_length > 0.0 else 0.0
        )

    junction_balance: Dict[str, float | None] = {}
    junction_structure_failures: list[str] = []
    for junction in sorted(
        solver_config.get("junctions", []) or [],
        key=lambda item: str(item["junction_name"]),
    ):
        junction_name = str(junction["junction_name"])
        inlet_series = [
            np.asarray(y[f"flow:{vessel_name}:{junction_name}"], dtype=np.float64)
            for vessel_name in sorted(vessel_topology)
            if downstream_names[vessel_name] == junction_name
        ]
        outlet_series = [
            np.asarray(y[f"flow:{junction_name}:{vessel_name}"], dtype=np.float64)
            for vessel_name in sorted(vessel_topology)
            if upstream_names[vessel_name] == junction_name
        ]
        if not inlet_series or not outlet_series:
            junction_balance[junction_name] = None
            junction_structure_failures.append(junction_name)
            continue
        inlet_total = np.sum(np.vstack(inlet_series), axis=0)
        outlet_total = np.sum(np.vstack(outlet_series), axis=0)
        junction_balance[junction_name] = _relative_rms_difference(
            inlet_total,
            outlet_total,
        )

    root_reference, root_error = _reference_inflow_series(
        solver_config,
        observation_count=observation_count,
        timing=observation_timing,
    )
    root_observed = None
    if root_reference is not None:
        flow_bc_name = str(_flow_boundary_condition(solver_config)["bc_name"])
        root_vessels = [
            vessel_name
            for vessel_name in sorted(vessel_topology)
            if upstream_names[vessel_name] == flow_bc_name
        ]
        if len(root_vessels) != 1:
            root_error = "FLOW boundary condition must connect to exactly one vessel"
        else:
            root_observed = np.asarray(
                y[f"flow:{flow_bc_name}:{root_vessels[0]}"],
                dtype=np.float64,
            )

    if root_observed is None or root_reference is None:
        root_waveform = {
            "relative_rms_error": None,
            "amplitude_ratio": None,
            "error": root_error,
        }
    else:
        reference_amplitude = float(np.ptp(root_reference))
        observed_amplitude = float(np.ptp(root_observed))
        root_waveform = {
            "relative_rms_error": _relative_rms_difference(root_observed, root_reference),
            "amplitude_ratio": (
                observed_amplitude / reference_amplitude
                if reference_amplitude > _FLOW_EPS
                else None
            ),
            "error": None,
        }

    selected_interfaces = {
        key: interface_sampling[key]
        for key in sorted(interface_sampling)
    }
    selected_sample_counts = [
        int(sample["usable_sample_count"])
        for sample in selected_interfaces.values()
        if not sample["excluded_from_calibration"]
    ]
    minimum_sample_count = (
        min(selected_sample_counts) if selected_sample_counts else None
    )
    resolution_failures = {
        key: int(sample["usable_sample_count"])
        for key, sample in selected_interfaces.items()
        if not sample["excluded_from_calibration"]
        and int(sample["usable_sample_count"]) < int(qc_config.minimum_usable_samples)
    }

    metrics = {
        "vessel_flow_continuity": {
            "per_vessel_relative_rms_error": vessel_continuity,
            "maximum_relative_rms_error": max(vessel_continuity.values(), default=0.0),
        },
        "junction_mass_balance": {
            "per_junction_relative_rms_error": junction_balance,
            "maximum_relative_rms_error": max(
                (value for value in junction_balance.values() if value is not None),
                default=0.0,
            ),
            "structure_failures": junction_structure_failures,
        },
        "root_waveform_agreement": root_waveform,
        "pressure_drop_direction": {
            "per_vessel": pressure_directions,
            "passing_fraction": float(
                sum(item["passed"] for item in pressure_directions.values())
                / max(len(pressure_directions), 1)
            ),
        },
        "path_coverage": {
            "per_vessel_fraction": path_coverage,
            "minimum_fraction": min(path_coverage.values(), default=0.0),
        },
        "sampling_resolution": {
            "per_interface_usable_sample_count": {
                key: int(sample["usable_sample_count"])
                for key, sample in selected_interfaces.items()
            },
            "minimum_usable_sample_count": minimum_sample_count,
            "failures": resolution_failures,
        },
    }
    thresholds = {
        "vessel_flow_continuity_tolerance": float(
            qc_config.vessel_flow_continuity_tolerance
        ),
        "junction_mass_balance_tolerance": float(
            qc_config.junction_mass_balance_tolerance
        ),
        "root_waveform_rms_tolerance": float(qc_config.root_waveform_rms_tolerance),
        "minimum_pressure_drop_fraction": float(
            qc_config.minimum_pressure_drop_fraction
        ),
        "minimum_path_coverage": float(qc_config.minimum_path_coverage),
        "minimum_usable_samples": int(qc_config.minimum_usable_samples),
    }
    checks = {
        "vessel_flow_continuity": (
            metrics["vessel_flow_continuity"]["maximum_relative_rms_error"]
            <= thresholds["vessel_flow_continuity_tolerance"]
        ),
        "junction_mass_balance": (
            not junction_structure_failures
            and metrics["junction_mass_balance"]["maximum_relative_rms_error"]
            <= thresholds["junction_mass_balance_tolerance"]
        ),
        "root_waveform_agreement": (
            root_waveform["relative_rms_error"] is not None
            and root_waveform["relative_rms_error"]
            <= thresholds["root_waveform_rms_tolerance"]
        ),
        "pressure_drop_direction": (
            metrics["pressure_drop_direction"]["passing_fraction"]
            >= thresholds["minimum_pressure_drop_fraction"]
        ),
        "path_coverage": (
            metrics["path_coverage"]["minimum_fraction"]
            >= thresholds["minimum_path_coverage"]
        ),
        "sampling_resolution": not resolution_failures,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "selected_interfaces": selected_interfaces,
        "exclusions": {
            key: excluded_blocks[key] for key in sorted(excluded_blocks)
        },
    }


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

    branch_series_by_id, observation_count, observation_timing = _branch_series_from_mapped_centerline(
        centerline_path=calibration.data_source.centerline or "",
        mapped_centerline_path=calibration.data_source.mapped_centerline_result or "",
        metadata_json=calibration.data_source.metadata_json,
        pressure_array=calibration.data_source.pressure_array,
        flow_array=calibration.data_source.flow_array,
        flow_observation_type=calibration.data_source.flow_observation_type,
        area_array=calibration.data_source.area_array,
        branch_id_array=calibration.data_source.branch_id_array,
        path_array=calibration.data_source.path_array,
    )
    vessel_topology, upstream_names, downstream_names = _network_topology(solver_config)
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

    junction_names = {
        str(junction["junction_name"])
        for junction in solver_config.get("junctions", []) or []
    }
    interface_sampling: Dict[str, Dict[str, Any]] = {}
    excluded_blocks: Dict[str, str] = {}
    y: Dict[str, list[float]] = {}
    for vessel_name, topology in vessel_topology.items():
        if topology.branch_id not in branch_series_by_id:
            raise ValueError(
                f"mapped centerline result does not contain observations for branch {topology.branch_id} ({vessel_name})"
            )

        branch_series = branch_series_by_id[topology.branch_id]
        upstream_name = upstream_names[vessel_name]
        downstream_name = downstream_names[vessel_name]
        excluded = (
            vessel_name in calibration.parameters.vessels.overrides
            and not vessel_parameters.get(vessel_name, [])
        )
        if excluded:
            excluded_blocks[vessel_name] = "empty_vessel_parameter_override"

        upstream_kind = (
            "internal" if upstream_name in junction_names else "external_upstream"
        )
        downstream_kind = (
            "internal" if downstream_name in junction_names else "external_downstream"
        )
        proximal_pressure, proximal_flow, proximal_sampling = _sample_interface(
            branch_series,
            topology.start_path,
            label=f"{vessel_name} upstream interface",
            interface_kind=upstream_kind,
            excluded=excluded,
        )
        distal_pressure, distal_flow, distal_sampling = _sample_interface(
            branch_series,
            topology.end_path,
            label=f"{vessel_name} downstream interface",
            interface_kind=downstream_kind,
            excluded=excluded,
        )
        interface_sampling[f"{vessel_name}:upstream"] = proximal_sampling
        interface_sampling[f"{vessel_name}:downstream"] = distal_sampling

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
        variable_name: _periodic_derivative(values, observation_timing)
        for variable_name, values in y.items()
    }

    observation_qc = _build_observation_qc(
        solver_config=solver_config,
        y=y,
        observation_count=observation_count,
        observation_timing=observation_timing,
        vessel_topology=vessel_topology,
        upstream_names=upstream_names,
        downstream_names=downstream_names,
        branch_series_by_id=branch_series_by_id,
        interface_sampling=interface_sampling,
        excluded_blocks=excluded_blocks,
        qc_config=calibration.observation_qc,
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
        interface_sampling=interface_sampling,
        excluded_blocks=excluded_blocks,
        observation_qc=observation_qc,
    )


def _parameter_blocks(
    config: Dict[str, Any],
) -> Iterable[tuple[str, str, Dict[str, Any], set[str]]]:
    """Yield block parameter mappings with their calibration selection."""
    for block_kind, values_key, name_key in (
        ("vessel", "zero_d_element_values", "vessel_name"),
        ("junction", "junction_values", "junction_name"),
    ):
        for block in config.get(f"{block_kind}s", []) or []:
            if not isinstance(block, dict):
                continue
            name = str(block.get(name_key, "<unnamed>"))
            values = block.get(values_key) or {}
            if not isinstance(values, dict):
                raise ValueError(f"{block_kind} {name} parameter values must be a mapping")
            selected = {str(value) for value in block.get("calibrate", []) or []}
            yield block_kind, name, values, selected


def _flatten_parameter_value(value: Any, *, label: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for index, nested in enumerate(value):
            flattened.extend(
                _flatten_parameter_value(nested, label=f"{label}[{index}]")
            )
        return flattened
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration parameter {label} must be numeric") from exc
    if not np.isfinite(resolved):
        raise ValueError(f"calibration parameter {label} must be finite")
    return [resolved]


def _validate_calibration_result(
    *,
    input_payload: Dict[str, Any],
    calibrated: Any,
    parameter_ratio_warning_threshold: float,
    pass_name: str,
) -> Dict[str, Any]:
    """Validate one black-box calibrator result and collect non-fatal warnings."""
    if not isinstance(calibrated, dict):
        raise ValueError("solver calibration returned a non-object result")

    nonfinite_paths = _collect_nonfinite_paths(calibrated)
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:5])
        remainder = len(nonfinite_paths) - min(len(nonfinite_paths), 5)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "calibrated svZeroD config contains non-finite values after solver "
            f"calibration: {preview}{suffix}"
        )

    if (
        not np.isfinite(parameter_ratio_warning_threshold)
        or parameter_ratio_warning_threshold < 1.0
    ):
        raise ValueError(
            "calibration.solver.parameter_ratio_warning_threshold must be finite and at least 1"
        )

    input_blocks = {
        (kind, name): (values, selected)
        for kind, name, values, selected in _parameter_blocks(input_payload)
    }
    output_blocks = {
        (kind, name): (values, selected)
        for kind, name, values, selected in _parameter_blocks(calibrated)
    }
    if set(input_blocks) != set(output_blocks):
        raise ValueError("solver calibration changed the set of vessel or junction blocks")

    selected_parameter_count = 0
    inactive_parameter_count = 0
    inactive_violations: list[str] = []
    negative_parameters: list[Dict[str, Any]] = []
    large_parameter_ratios: list[Dict[str, Any]] = []
    for block_key in sorted(input_blocks):
        input_values, selected = input_blocks[block_key]
        output_values, _ = output_blocks[block_key]
        if not isinstance(output_values, dict):
            raise ValueError(f"solver calibration returned invalid parameter values for {block_key[1]}")
        missing_selected = sorted(set(selected) - set(input_values))
        if missing_selected:
            raise ValueError(
                f"solver calibration input is missing selected parameters for "
                f"{block_key[1]}: {missing_selected}"
            )
        for parameter, input_value in input_values.items():
            if parameter not in output_values:
                raise ValueError(
                    f"solver calibration removed parameter {block_key[1]}.{parameter}"
                )
            input_flat = _flatten_parameter_value(
                input_value, label=f"{block_key[1]}.{parameter}"
            )
            output_flat = _flatten_parameter_value(
                output_values[parameter], label=f"{block_key[1]}.{parameter}"
            )
            if len(input_flat) != len(output_flat):
                raise ValueError(
                    f"solver calibration changed scalar/list shape for "
                    f"{block_key[1]}.{parameter}"
                )
            for index, (initial, final) in enumerate(zip(input_flat, output_flat)):
                label = f"{block_key[1]}.{parameter}"
                if len(input_flat) > 1:
                    label += f"[{index}]"
                if parameter not in selected:
                    inactive_parameter_count += 1
                    if final != initial:
                        inactive_violations.append(label)
                    continue
                selected_parameter_count += 1
                if final < 0.0:
                    negative_parameters.append(
                        {
                            "path": label,
                            "parameter": parameter,
                            "pass": pass_name,
                            "value": float(final),
                        }
                    )
                if initial == 0.0 and final == 0.0:
                    ratio = 1.0
                elif initial == 0.0 or final == 0.0:
                    ratio = np.inf
                else:
                    ratio = max(abs(final / initial), abs(initial / final))
                if ratio > parameter_ratio_warning_threshold:
                    large_parameter_ratios.append(
                        {
                            "path": label,
                            "parameter": parameter,
                            "pass": pass_name,
                            "initial_value": float(initial),
                            "value": float(final),
                            "ratio": None if np.isinf(ratio) else float(ratio),
                            "threshold": float(parameter_ratio_warning_threshold),
                        }
                    )

    if inactive_violations:
        raise ValueError(
            "solver calibration changed inactive parameters: "
            + ", ".join(sorted(inactive_violations))
        )

    negative_parameters.sort(key=lambda item: (item["path"], item["pass"]))
    large_parameter_ratios.sort(key=lambda item: (item["path"], item["pass"]))
    return {
        "pass": pass_name,
        "status": "ok",
        "selected_parameter_count": selected_parameter_count,
        "inactive_parameter_count": inactive_parameter_count,
        "negative_parameter_paths": sorted(
            {item["path"] for item in negative_parameters}
        ),
        "large_ratio_paths": sorted(
            {item["path"] for item in large_parameter_ratios}
        ),
        "negative_parameters": negative_parameters,
        "large_parameter_ratios": large_parameter_ratios,
    }


def _replace_parameter_values(
    payload: Dict[str, Any],
    calibrated: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy only solver block values into a fresh confirmation payload."""
    replacement = copy.deepcopy(payload)
    for block_kind, values_key, name_key in (
        ("vessel", "zero_d_element_values", "vessel_name"),
        ("junction", "junction_values", "junction_name"),
    ):
        calibrated_values = {
            (block_kind, str(block[name_key])): copy.deepcopy(block.get(values_key) or {})
            for block in calibrated.get(f"{block_kind}s", []) or []
            if isinstance(block, dict) and name_key in block
        }
        for block in replacement.get(f"{block_kind}s", []) or []:
            if not isinstance(block, dict) or name_key not in block:
                continue
            key = (block_kind, str(block[name_key]))
            if key not in calibrated_values:
                raise ValueError(
                    f"solver calibration did not return {block_kind} block {key[1]}"
                )
            block[values_key] = calibrated_values[key]
    return replacement


def _confirmation_parameter_deltas(
    *,
    initial_payload: Dict[str, Any],
    first_calibrated: Dict[str, Any],
    confirmation_calibrated: Dict[str, Any],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> list[Dict[str, Any]]:
    """Compare selected scalar/list values from the two black-box passes."""
    initial_blocks = {
        (kind, name): (values, selected)
        for kind, name, values, selected in _parameter_blocks(initial_payload)
    }
    first_blocks = {
        (kind, name): values
        for kind, name, values, _selected in _parameter_blocks(first_calibrated)
    }
    confirmation_blocks = {
        (kind, name): values
        for kind, name, values, _selected in _parameter_blocks(confirmation_calibrated)
    }
    if set(initial_blocks) != set(first_blocks) or set(initial_blocks) != set(
        confirmation_blocks
    ):
        raise ValueError("solver calibration changed the set of vessel or junction blocks")

    deltas: list[Dict[str, Any]] = []
    for block_key in sorted(initial_blocks):
        _initial_values, selected = initial_blocks[block_key]
        first_values = first_blocks[block_key]
        confirmation_values = confirmation_blocks[block_key]
        for parameter in sorted(selected):
            if parameter not in first_values or parameter not in confirmation_values:
                raise ValueError(
                    f"solver calibration confirmation is missing selected parameter "
                    f"{block_key[1]}.{parameter}"
                )
            first_flat = _flatten_parameter_value(
                first_values[parameter], label=f"{block_key[1]}.{parameter}"
            )
            confirmation_flat = _flatten_parameter_value(
                confirmation_values[parameter], label=f"{block_key[1]}.{parameter}"
            )
            if len(first_flat) != len(confirmation_flat):
                raise ValueError(
                    f"solver calibration confirmation changed scalar/list shape for "
                    f"{block_key[1]}.{parameter}"
                )
            for index, (first, confirmation) in enumerate(
                zip(first_flat, confirmation_flat)
            ):
                label = f"{block_key[1]}.{parameter}"
                if len(first_flat) > 1:
                    label += f"[{index}]"
                absolute_delta = abs(confirmation - first)
                scale = max(abs(first), abs(confirmation))
                relative_delta = absolute_delta / scale if scale > 0.0 else 0.0
                limit = absolute_tolerance + relative_tolerance * scale
                deltas.append(
                    {
                        "path": label,
                        "first_value": float(first),
                        "confirmation_value": float(confirmation),
                        "absolute_delta": float(absolute_delta),
                        "relative_delta": float(relative_delta),
                        "limit": float(limit),
                        "passed": bool(absolute_delta <= limit),
                    }
                )
    return deltas


def _validate_confirmation_tolerances(calibration: CalibrationConfig) -> tuple[float, float]:
    absolute_tolerance = float(calibration.solver.confirmation_absolute_tolerance)
    relative_tolerance = float(calibration.solver.confirmation_relative_tolerance)
    if (
        not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
        or not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
    ):
        raise ValueError(
            "calibration.solver confirmation_absolute_tolerance and "
            "confirmation_relative_tolerance must be finite and non-negative"
        )
    return absolute_tolerance, relative_tolerance


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
    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path = output_path.parent / "calibration_observation_qc.json"
    qc_path.write_text(
        json.dumps(assembly.observation_qc, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if assembly.observation_qc.get("status") != "pass":
        failed_checks = [
            name
            for name, passed in assembly.observation_qc.get("checks", {}).items()
            if not passed
        ]
        raise ValueError(
            "calibration observation QC failed before solver dispatch: "
            f"{', '.join(failed_checks)}; report: {qc_path}"
        )

    absolute_tolerance, relative_tolerance = _validate_confirmation_tolerances(calibration)
    clear_calibration_provenance()
    try:
        first_calibrated = calibrate_pysvzerod(assembly.solver_payload)
    except Exception as exc:
        raise ValueError(
            "solver calibration first pass failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    first_pass = _validate_calibration_result(
        input_payload=assembly.solver_payload,
        calibrated=first_calibrated,
        parameter_ratio_warning_threshold=(
            calibration.solver.parameter_ratio_warning_threshold
        ),
        pass_name="first",
    )
    first_provenance = last_calibration_provenance()

    confirmation_payload = _replace_parameter_values(
        assembly.solver_payload,
        first_calibrated,
    )
    try:
        confirmation_calibrated = calibrate_pysvzerod(confirmation_payload)
    except Exception as exc:
        raise ValueError(
            "solver calibration confirmation pass failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    confirmation_pass = _validate_calibration_result(
        input_payload=confirmation_payload,
        calibrated=confirmation_calibrated,
        parameter_ratio_warning_threshold=(
            calibration.solver.parameter_ratio_warning_threshold
        ),
        pass_name="confirmation",
    )
    confirmation_provenance = last_calibration_provenance()

    deltas = _confirmation_parameter_deltas(
        initial_payload=assembly.solver_payload,
        first_calibrated=first_calibrated,
        confirmation_calibrated=confirmation_calibrated,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    failed_deltas = [delta for delta in deltas if not delta["passed"]]
    if failed_deltas:
        preview = ", ".join(
            f"{delta['path']} (delta={delta['absolute_delta']:g}, limit={delta['limit']:g})"
            for delta in failed_deltas[:5]
        )
        remainder = len(failed_deltas) - min(len(failed_deltas), 5)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "solver calibration confirmation did not reach a parameter fixed point: "
            f"{preview}{suffix}"
        )

    warning_passes = (first_pass, confirmation_pass)
    negative_parameter_paths = sorted(
        {
            path
            for pass_summary in warning_passes
            for path in pass_summary["negative_parameter_paths"]
        }
    )
    large_ratio_paths = sorted(
        {
            path
            for pass_summary in warning_passes
            for path in pass_summary["large_ratio_paths"]
        }
    )
    warning_records = {
        "negative_parameters": [
            record
            for pass_summary in warning_passes
            for record in pass_summary["negative_parameters"]
        ],
        "large_parameter_ratios": [
            record
            for pass_summary in warning_passes
            for record in pass_summary["large_parameter_ratios"]
        ],
    }
    warning_records["negative_parameters"].sort(
        key=lambda item: (item["path"], item["pass"])
    )
    warning_records["large_parameter_ratios"].sort(
        key=lambda item: (item["path"], item["pass"])
    )

    confirmation_summary = {
        "status": "converged",
        "converged": True,
        "termination_reason": "confirmed_fixed_point",
        "convergence_tolerances": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
        "parameter_ratio_warning_threshold": float(
            calibration.solver.parameter_ratio_warning_threshold
        ),
        "inactive_parameters_preserved": True,
        "maximum_absolute_confirmation_delta": max(
            (delta["absolute_delta"] for delta in deltas),
            default=0.0,
        ),
        "maximum_relative_confirmation_delta": max(
            (delta["relative_delta"] for delta in deltas),
            default=0.0,
        ),
        "confirmation_deltas": deltas,
        "invocations": [
            {
                "pass": first_pass["pass"],
                "status": first_pass["status"],
                "selected_parameter_count": first_pass["selected_parameter_count"],
                "inactive_parameter_count": first_pass["inactive_parameter_count"],
                "negative_parameter_paths": first_pass["negative_parameter_paths"],
                "large_ratio_paths": first_pass["large_ratio_paths"],
                "solver_provenance": first_provenance,
            },
            {
                "pass": confirmation_pass["pass"],
                "status": confirmation_pass["status"],
                "selected_parameter_count": confirmation_pass["selected_parameter_count"],
                "inactive_parameter_count": confirmation_pass["inactive_parameter_count"],
                "negative_parameter_paths": confirmation_pass["negative_parameter_paths"],
                "large_ratio_paths": confirmation_pass["large_ratio_paths"],
                "solver_provenance": confirmation_provenance,
            },
        ],
        "negative_parameter_paths": negative_parameter_paths,
        "large_ratio_paths": large_ratio_paths,
        "warnings": warning_records,
    }
    confirmation_path = output_path.parent / "calibration_confirmation.json"
    confirmation_path.write_text(
        json.dumps(confirmation_summary, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    published = copy.deepcopy(confirmation_calibrated)
    published.pop("calibration_diagnostics", None)

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(published, stream, indent=4, allow_nan=False)

    return {
        "status": "ok",
        "output_config": str(output_path),
        "observation_count": assembly.observation_count,
        "variable_count": assembly.variable_count,
        "input_normalization": assembly.input_normalization,
        "interface_sampling": assembly.interface_sampling,
        "excluded_blocks": assembly.excluded_blocks,
        "observation_qc": assembly.observation_qc,
        "observation_qc_report": str(qc_path),
        "calibration_confirmation": confirmation_summary,
        "calibration_confirmation_report": str(confirmation_path),
        "solver_provenance": last_calibration_provenance(),
    }
