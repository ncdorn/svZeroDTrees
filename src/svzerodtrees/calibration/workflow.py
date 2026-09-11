from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable
import uuid

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .._pysvzerod import (
    calibrate_pysvzerod,
    clear_calibration_provenance,
    last_calibration_provenance,
    simulate_pysvzerod,
)
from ..config import CalibrationConfig
from .replay import build_replay_payload as build_settled_replay_payload
from .replay import validate_replay as validate_settled_replay
from .targets import evaluate_pulmonary_targets

_VESSEL_NAME_RE = re.compile(r"^branch(?P<branch_id>\d+)_seg(?P<seg_id>\d+)$")
_PATH_TOLERANCE_ABS = 1e-3
_MIN_USABLE_INTERFACE_SAMPLES = 3
_FLOW_EPS = 1e-8
_REPLAY_SCALE_FLOOR = 1e-12
_CALIBRATION_ONLY_TOP_LEVEL_KEYS = frozenset(
    {"y", "dy", "calibrate", "calibration_parameters", "calibration_diagnostics"}
)
_SUPPORTED_JUNCTION_TYPES = frozenset(
    {"NORMAL_JUNCTION", "BloodVesselJunction", "resistive_junction"}
)


def _is_calibration_only_top_level_key(key: str) -> bool:
    return key in _CALIBRATION_ONLY_TOP_LEVEL_KEYS or key.startswith("calibration_")


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
    target_observations: Dict[str, Any] = field(default_factory=dict)
    target_units: Dict[str, str] = field(default_factory=dict)
    target_phases: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ObservationTiming:
    timestamps_s: np.ndarray | None
    cycle_duration_s: float | None
    pressure_units: str | None = None
    flow_units: str | None = None


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
        resolved_flow_units = str(flow_contract["units"])
    elif flow_contract.get("quantity") != "velocity" or not flow_contract.get("units"):
        raise ValueError("velocity timeseries metadata must declare quantity=velocity and units")
    else:
        # Branch flow is converted to volumetric flow below when the source
        # arrays contain velocity.  Keep the target evaluator's unit contract
        # explicit rather than relabeling the external metadata.
        resolved_flow_units = "cm^3/s"
    resolved_pressure_units = str(pressure_contract["units"])

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
        return ObservationTiming(
            timestamps_array,
            None,
            pressure_units=resolved_pressure_units,
            flow_units=resolved_flow_units,
        )
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
    return ObservationTiming(
        timestamps_array,
        cycle_duration,
        pressure_units=resolved_pressure_units,
        flow_units=resolved_flow_units,
    )


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


def _resolve_target_interface(
    *,
    vessel: str,
    interface: Any,
    vessel_topology: Dict[str, VesselTopology],
    upstream_names: Dict[str, str],
    downstream_names: Dict[str, str],
    junction_names: set[str],
    interface_sampling: Dict[str, Dict[str, Any]],
) -> tuple[str | None, Dict[str, Any]]:
    """Resolve one operator-selected target interface without anatomy inference.

    ``upstream`` and ``downstream`` are endpoint selectors.  The explicit
    ``external_*`` forms additionally require the selected endpoint to be an
    external boundary.  ``internal`` is accepted only when exactly one of the
    two endpoints is an internal junction; accepting both would make the
    target location ambiguous.
    """
    detail: Dict[str, Any] = {
        "vessel": vessel,
        "requested_interface": interface,
        "resolved_endpoint": None,
        "sampling_key": None,
        "interface_kind": None,
        "error": None,
    }
    if vessel not in vessel_topology:
        detail["error"] = "target vessel is not present in the solver topology"
        return None, detail
    if not isinstance(interface, str):
        detail["error"] = "target interface must be a string"
        return None, detail
    interface = interface.strip().lower()
    if interface not in {
        "external_upstream",
        "external_downstream",
        "internal",
        "upstream",
        "downstream",
    }:
        detail["error"] = f"unsupported target interface '{interface}'"
        return None, detail

    endpoint: str | None
    if interface in {"upstream", "external_upstream"}:
        endpoint = "upstream"
    elif interface in {"downstream", "external_downstream"}:
        endpoint = "downstream"
    else:
        candidates = []
        if upstream_names.get(vessel) in junction_names:
            candidates.append("upstream")
        if downstream_names.get(vessel) in junction_names:
            candidates.append("downstream")
        if len(candidates) != 1:
            detail["error"] = (
                "internal target interface is ambiguous; exactly one endpoint "
                "must connect to a junction"
            )
            return None, detail
        endpoint = candidates[0]

    connection_name = (
        upstream_names[vessel] if endpoint == "upstream" else downstream_names[vessel]
    )
    actual_kind = (
        "internal" if connection_name in junction_names else f"external_{endpoint}"
    )
    if interface.startswith("external_") and interface != actual_kind:
        detail["error"] = (
            f"target interface '{interface}' does not match the topology's "
            f"'{actual_kind}' endpoint"
        )
        return None, detail

    sampling_key = f"{vessel}:{endpoint}"
    sample = interface_sampling.get(sampling_key)
    detail.update(
        {
            "resolved_endpoint": endpoint,
            "sampling_key": sampling_key,
            "interface_kind": actual_kind,
        }
    )
    if sample is None:
        detail["error"] = "target interface has no assembled observation sample"
        return None, detail
    detail["sampling"] = sample
    return endpoint, detail


def _target_interface_series(
    *,
    vessel: str,
    endpoint: str,
    upstream_names: Dict[str, str],
    downstream_names: Dict[str, str],
    y: Dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if endpoint == "upstream":
        connection = upstream_names.get(vessel)
        flow_key = f"flow:{connection}:{vessel}"
        pressure_key = f"pressure:{connection}:{vessel}"
    elif endpoint == "downstream":
        connection = downstream_names.get(vessel)
        flow_key = f"flow:{vessel}:{connection}"
        pressure_key = f"pressure:{vessel}:{connection}"
    else:
        return None
    if flow_key not in y or pressure_key not in y:
        return None
    flow = np.asarray(y[flow_key], dtype=np.float64)
    pressure = np.asarray(y[pressure_key], dtype=np.float64)
    if not np.isfinite(flow).all() or not np.isfinite(pressure).all():
        return None
    return pressure, flow


def _assemble_target_observations(
    *,
    targets: Any,
    observation_qc: Dict[str, Any],
    y: Dict[str, list[float]],
    observation_timing: ObservationTiming,
    upstream_names: Dict[str, str],
    downstream_names: Dict[str, str],
) -> tuple[Dict[str, Any], Dict[str, str], list[float]]:
    """Capture explicitly configured pulmonary target traces from ``y``.

    Target scoring consumes the same interface samples sent to the solver.  A
    target-focused run therefore needs the resolved endpoint and the metadata
    phase grid retained alongside the assembled payload; it must not recreate
    target traces from vessel numbers or replay output later.
    """
    if targets is None:
        return {}, {}, []
    timing = observation_timing
    if timing.timestamps_s is None or timing.cycle_duration_s is None:
        # The QC-only unit fixtures intentionally omit the optional metadata
        # sidecar.  Target scoring will fail clearly later if such a payload is
        # used for a target-aware solver run, but observation assembly remains
        # responsible for reporting topology and sampling diagnostics first.
        return {}, {}, []
    if not timing.pressure_units or not timing.flow_units:
        raise ValueError(
            "target-focused calibration requires explicit pressure and volumetric-flow units"
        )
    timestamps = np.asarray(timing.timestamps_s, dtype=np.float64)
    phases = (timestamps - float(timestamps[0])) / float(timing.cycle_duration_s)
    if not np.isfinite(phases).all() or phases.size < 2:
        raise ValueError("target-focused calibration requires at least two finite phases")

    target_interfaces = observation_qc.get("target_interfaces") or {}
    mpa = targets.mpa_pressure
    split = targets.rpa_flow_split
    role_specs = {
        "mpa_pressure": (str(mpa.vessel), "pressure"),
        "rpa_flow": (str(split.rpa_vessel), "flow"),
        "lpa_flow": (str(split.lpa_vessel), "flow"),
    }
    observations: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    for role, (vessel, quantity) in role_specs.items():
        detail = target_interfaces.get(role.split("_", 1)[0], {})
        endpoint = detail.get("resolved_endpoint")
        if endpoint not in {"upstream", "downstream"}:
            raise ValueError(
                f"target interface for {role} was not resolved against solver topology"
            )
        series = _target_interface_series(
            vessel=vessel,
            endpoint=endpoint,
            upstream_names=upstream_names,
            downstream_names=downstream_names,
            y=y,
        )
        if series is None:
            raise ValueError(f"target {role} does not have an assembled observation")
        pressure, flow = series
        if quantity == "pressure":
            values = pressure
            connection = upstream_names[vessel] if endpoint == "upstream" else downstream_names[vessel]
            variable_name = (
                f"pressure:{connection}:{vessel}"
                if endpoint == "upstream"
                else f"pressure:{vessel}:{connection}"
            )
        else:
            values = flow
            connection = upstream_names[vessel] if endpoint == "upstream" else downstream_names[vessel]
            variable_name = (
                f"flow:{connection}:{vessel}"
                if endpoint == "upstream"
                else f"flow:{vessel}:{connection}"
            )
        observations[role] = {
            "phases": [float(value) for value in phases],
            "values": [float(value) for value in values],
            "units": timing.pressure_units if quantity == "pressure" else timing.flow_units,
            "orientation": "away_from_mpa",
            "vessel": vessel,
            "interface": detail.get("requested_interface"),
            "endpoint": endpoint,
            "variable": variable_name,
        }
        units[role] = timing.pressure_units if quantity == "pressure" else timing.flow_units
    return observations, units, [float(value) for value in phases]


def _build_target_observation_qc(
    *,
    targets: Any,
    vessel_topology: Dict[str, VesselTopology],
    upstream_names: Dict[str, str],
    downstream_names: Dict[str, str],
    junction_names: set[str],
    interface_sampling: Dict[str, Dict[str, Any]],
    y: Dict[str, list[float]],
    minimum_usable_samples: int,
) -> tuple[Dict[str, bool], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Validate explicit pulmonary target locations and required observations.

    This deliberately stops at target-data qualification.  It does not
    calculate MPA waveform error or an RPA split; those belong to the pure
    target evaluator.  The split denominator check is retained here because a
    zero denominator makes the target data contract invalid before calibration.
    """
    target_checks = {
        "target_topology": False,
        "target_sampling_resolution": False,
        "target_split_denominator": False,
    }
    target_metrics: Dict[str, Any] = {}
    target_interfaces: Dict[str, Dict[str, Any]] = {}
    try:
        mpa = targets.mpa_pressure
        split = targets.rpa_flow_split
        roles = {
            "mpa": str(mpa.vessel),
            "rpa": str(split.rpa_vessel),
            "lpa": str(split.lpa_vessel),
        }
        target_metrics["roles"] = roles
    except (AttributeError, TypeError) as exc:
        target_metrics["error"] = f"target configuration is incomplete: {exc}"
        return target_checks, target_metrics, target_interfaces

    target_checks["target_topology"] = len(set(roles.values())) == 3
    if not target_checks["target_topology"]:
        target_metrics["topology_error"] = (
            "MPA, LPA, and RPA target vessel roles must be distinct"
        )

    for role, vessel, interface in (
        ("mpa", roles["mpa"], getattr(mpa, "interface", None)),
        ("rpa", roles["rpa"], getattr(split, "interface", None)),
        ("lpa", roles["lpa"], getattr(split, "interface", None)),
    ):
        endpoint, detail = _resolve_target_interface(
            vessel=vessel,
            interface=interface,
            vessel_topology=vessel_topology,
            upstream_names=upstream_names,
            downstream_names=downstream_names,
            junction_names=junction_names,
            interface_sampling=interface_sampling,
        )
        target_interfaces[role] = detail
        detail["role"] = role
        if endpoint is None:
            target_checks["target_topology"] = False

    qualified = True
    for detail in target_interfaces.values():
        sample = detail.get("sampling")
        if not isinstance(sample, dict):
            qualified = False
            continue
        sample_count = int(sample.get("usable_sample_count", 0))
        detail["target_sample_count"] = sample_count
        detail["target_sample_qualified"] = bool(
            not sample.get("excluded_from_calibration", False)
            and sample_count >= minimum_usable_samples
            and sample.get("quality_status")
            in {"qualified_interior", "interpolated_internal"}
        )
        if not detail["target_sample_qualified"]:
            qualified = False
    target_checks["target_sampling_resolution"] = qualified

    split_series: Dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for role in ("rpa", "lpa"):
        detail = target_interfaces.get(role, {})
        endpoint = detail.get("resolved_endpoint")
        split_series[role] = (
            _target_interface_series(
                vessel=roles[role],
                endpoint=endpoint,
                upstream_names=upstream_names,
                downstream_names=downstream_names,
                y=y,
            )
            if endpoint is not None
            else None
        )
    if all(series is not None for series in split_series.values()):
        _rpa_pressure, rpa_flow = split_series["rpa"]  # type: ignore[misc]
        _lpa_pressure, lpa_flow = split_series["lpa"]  # type: ignore[misc]
        rpa_mean = float(np.mean(rpa_flow))
        lpa_mean = float(np.mean(lpa_flow))
        denominator = rpa_mean + lpa_mean
        denominator_valid = bool(
            np.isfinite(denominator) and abs(denominator) > _FLOW_EPS
        )
        target_metrics["rpa_flow_split_denominator"] = {
            "rpa_mean_flow": rpa_mean,
            "lpa_mean_flow": lpa_mean,
            "value": denominator,
            "nonzero": denominator_valid,
        }
        target_checks["target_split_denominator"] = denominator_valid
    else:
        target_metrics["rpa_flow_split_denominator"] = {
            "rpa_mean_flow": None,
            "lpa_mean_flow": None,
            "value": None,
            "nonzero": False,
            "error": "RPA/LPA target flow samples are unavailable",
        }

    return target_checks, target_metrics, target_interfaces


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
    targets: Any = None,
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
    # The target-focused profile intentionally separates data/target-contract
    # checks from diagnostics about the complete mapped network.  The latter
    # remain in the report so an operator can inspect the observations, but a
    # poor non-target continuity or pressure-direction metric must not prevent
    # the unchanged calibrator from receiving the complete state.
    target_checks: Dict[str, bool] = {}
    target_metrics: Dict[str, Any] = {}
    target_interfaces: Dict[str, Dict[str, Any]] = {}
    if targets is not None:
        target_checks, target_metrics, target_interfaces = _build_target_observation_qc(
            targets=targets,
            vessel_topology=vessel_topology,
            upstream_names=upstream_names,
            downstream_names=downstream_names,
            junction_names={
                str(junction["junction_name"])
                for junction in solver_config.get("junctions", []) or []
            },
            interface_sampling=interface_sampling,
            y=y,
            minimum_usable_samples=int(qc_config.minimum_usable_samples),
        )

    all_checks = {**checks, **target_checks}
    enforcement = str(getattr(qc_config, "enforcement", "strict_network")).lower()
    if enforcement == "target_focused" and targets is None:
        target_checks = {"target_configuration": False}
        target_metrics = {
            "error": (
                "target_focused observation QC requires explicit MPA, LPA, and "
                "RPA target roles"
            )
        }
        all_checks = {**checks, **target_checks}
    if enforcement == "target_focused":
        # Root inflow agreement is a fatal input/data-contract check in the
        # target-focused profile.  The remaining existing checks describe
        # whole-network consistency and therefore remain advisory.
        fatal_checks = {
            "root_waveform_agreement": checks["root_waveform_agreement"],
            **target_checks,
        }
        advisory_checks = {
            name: passed
            for name, passed in checks.items()
            if name != "root_waveform_agreement"
        }
    else:
        # strict_network retains the historical all-checks gate.  When target
        # roles are present, their contract checks are included in that gate.
        fatal_checks = dict(all_checks)
        advisory_checks = {}

    severity = {
        name: "fatal" if name in fatal_checks else "advisory"
        for name in all_checks
    }
    return {
        "status": "pass" if all(fatal_checks.values()) else "fail",
        "enforcement": enforcement,
        "checks": all_checks,
        "severity": severity,
        "fatal_checks": fatal_checks,
        "advisory_checks": advisory_checks,
        "failed_fatal_checks": sorted(
            name for name, passed in fatal_checks.items() if not passed
        ),
        "failed_advisory_checks": sorted(
            name for name, passed in advisory_checks.items() if not passed
        ),
        "metrics": metrics,
        "target_metrics": target_metrics,
        "target_interfaces": target_interfaces,
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
        targets=calibration.targets,
    )
    target_observations, target_units, target_phases = _assemble_target_observations(
        targets=calibration.targets,
        observation_qc=observation_qc,
        y=y,
        observation_timing=observation_timing,
        upstream_names=upstream_names,
        downstream_names=downstream_names,
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
        target_observations=target_observations,
        target_units=target_units,
        target_phases=target_phases,
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


def _replay_settings(calibration: CalibrationConfig) -> Dict[str, float]:
    """Resolve and validate numerical settings used only by replay validation."""
    solver = calibration.solver
    settings = {
        "pressure_bound_multiplier": float(solver.pressure_bound_multiplier),
        "flow_bound_multiplier": float(solver.flow_bound_multiplier),
        "cycle_stability_tolerance": float(solver.cycle_stability_tolerance),
    }
    if (
        not np.isfinite(settings["pressure_bound_multiplier"])
        or settings["pressure_bound_multiplier"] <= 0.0
        or not np.isfinite(settings["flow_bound_multiplier"])
        or settings["flow_bound_multiplier"] <= 0.0
        or not np.isfinite(settings["cycle_stability_tolerance"])
        or settings["cycle_stability_tolerance"] < 0.0
    ):
        raise ValueError(
            "calibration.solver pressure_bound_multiplier and "
            "flow_bound_multiplier must be finite and positive, and "
            "cycle_stability_tolerance must be finite and non-negative"
        )
    return settings


def _junction_connection_count(junction: Dict[str, Any], key: str) -> int | None:
    values = junction.get(key)
    if values is None:
        block_key = key.replace("_vessels", "_blocks")
        values = junction.get(block_key)
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"junction {junction.get('junction_name', '<unnamed>')} {key} must be a list")
    return len(values)


def _normalize_calibrated_config(
    calibrated: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Remove calibration fields and normalize junctions for simulation.

    ``pysvzerod.calibrate`` intentionally returns its optimization payload with
    observation and selection fields.  Those fields are useful while checking
    the calibration but are not part of a solver input configuration.
    """
    if not isinstance(calibrated, dict):
        raise ValueError("solver calibration returned a non-object result")

    normalized = copy.deepcopy(calibrated)
    removed_top_level = sorted(
        key for key in normalized if _is_calibration_only_top_level_key(key)
    )
    for key in removed_top_level:
        normalized.pop(key, None)

    removed_block_selections: list[str] = []
    for block_kind, name_key in (("vessels", "vessel_name"), ("junctions", "junction_name")):
        blocks = normalized.get(block_kind, []) or []
        if not isinstance(blocks, list):
            raise ValueError(f"solver output '{block_kind}' must be a list")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"solver output {block_kind} entries must be objects")
            if "calibrate" in block:
                block_name = str(block.get(name_key, "<unnamed>"))
                removed_block_selections.append(f"{block_kind}.{block_name}.calibrate")
                block.pop("calibrate", None)

    junction_type_changes: list[Dict[str, str]] = []
    removed_junction_values: list[str] = []
    for junction in normalized.get("junctions", []) or []:
        junction_name = str(junction.get("junction_name", "<unnamed>"))
        junction_type = str(junction.get("junction_type", ""))
        outlet_count = _junction_connection_count(junction, "outlet_vessels")

        # ``internal_junction`` is an old svZeroDTrees spelling for a
        # single-vessel connection.  The solver represents that topology as a
        # parameter-free normal junction.  A multi-outlet block with values is
        # a blood-vessel junction; without values it is a normal junction.
        if junction_type.lower() == "internal_junction":
            if outlet_count is None:
                raise ValueError(
                    f"junction {junction_name} internal_junction is missing outlet connections"
                )
            target_type = (
                "BloodVesselJunction"
                if outlet_count > 1 and junction.get("junction_values")
                else "NORMAL_JUNCTION"
            )
            junction["junction_type"] = target_type
            if target_type == "NORMAL_JUNCTION" and "junction_values" in junction:
                junction.pop("junction_values", None)
                removed_junction_values.append(
                    f"junctions.{junction_name}.junction_values"
                )
            junction_type_changes.append(
                {
                    "junction_name": junction_name,
                    "from": junction_type,
                    "to": target_type,
                }
            )
        elif junction_type == "BloodVesselJunction" and not junction.get("junction_values"):
            # A parameter-free BloodVesselJunction cannot be reconstructed as
            # a calibrated block.  Normal junction is its valid equivalent.
            junction["junction_type"] = "NORMAL_JUNCTION"
            if "junction_values" in junction:
                junction.pop("junction_values", None)
                removed_junction_values.append(
                    f"junctions.{junction_name}.junction_values"
                )
            junction_type_changes.append(
                {
                    "junction_name": junction_name,
                    "from": junction_type,
                    "to": "NORMAL_JUNCTION",
                }
            )

    normalization = {
        "removed_top_level_fields": removed_top_level,
        "removed_block_selection_fields": sorted(removed_block_selections),
        "removed_junction_values": sorted(removed_junction_values),
        "junction_type_changes": sorted(
            junction_type_changes,
            key=lambda item: item["junction_name"],
        ),
    }
    return normalized, normalization


def _validate_publishable_solver_config(config: Dict[str, Any]) -> None:
    """Perform deterministic structural checks before the solver replay."""
    if not isinstance(config, dict):
        raise ValueError("normalized solver configuration must be an object")

    calibration_fields = sorted(
        key for key in config if _is_calibration_only_top_level_key(key)
    )
    if calibration_fields:
        raise ValueError(
            "normalized solver configuration retains calibration-only fields: "
            + ", ".join(calibration_fields)
        )

    nonfinite_paths = _collect_nonfinite_paths(config)
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:5])
        remainder = len(nonfinite_paths) - min(len(nonfinite_paths), 5)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "normalized solver configuration contains non-finite values: "
            f"{preview}{suffix}"
        )

    required_lists = ("boundary_conditions", "vessels", "junctions")
    for key in required_lists:
        if not isinstance(config.get(key), list):
            raise ValueError(f"normalized solver configuration '{key}' must be a list")
    if not isinstance(config.get("simulation_parameters"), dict):
        raise ValueError(
            "normalized solver configuration requires a simulation_parameters object"
        )

    boundary_condition_names: set[str] = set()
    for index, boundary_condition in enumerate(config["boundary_conditions"]):
        if not isinstance(boundary_condition, dict):
            raise ValueError(f"boundary_conditions[{index}] must be an object")
        for key in ("bc_name", "bc_type", "bc_values"):
            if key not in boundary_condition:
                raise ValueError(
                    f"boundary_conditions[{index}] is missing required field '{key}'"
                )
        if not isinstance(boundary_condition["bc_name"], str) or not isinstance(
            boundary_condition["bc_type"], str
        ):
            raise ValueError(
                f"boundary_conditions[{index}].bc_name and bc_type must be strings"
            )
        boundary_condition_name = str(boundary_condition["bc_name"])
        if boundary_condition_name in boundary_condition_names:
            raise ValueError(
                "duplicate bc_name in normalized solver configuration: "
                f"{boundary_condition_name}"
            )
        boundary_condition_names.add(boundary_condition_name)
        if not isinstance(boundary_condition["bc_values"], dict):
            raise ValueError(
                f"boundary_conditions[{index}].bc_values must be an object"
            )

    simparams = config["simulation_parameters"]
    for key in ("number_of_cardiac_cycles", "number_of_time_pts_per_cardiac_cycle"):
        if key not in simparams:
            raise ValueError(
                "normalized solver configuration requires "
                f"simulation_parameters.{key} for replay validation"
            )
        value = simparams[key]
        if isinstance(value, bool):
            raise ValueError(f"simulation_parameters.{key} must be an integer")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"simulation_parameters.{key} must be numeric") from exc
        if not np.isfinite(numeric_value) or numeric_value != int(numeric_value):
            raise ValueError(f"simulation_parameters.{key} must be a finite integer")
        minimum = 1 if key == "number_of_cardiac_cycles" else 2
        if int(numeric_value) < minimum:
            raise ValueError(
                f"simulation_parameters.{key} must be at least {minimum} for replay"
            )

    vessel_ids: set[int] = set()
    vessel_names: set[str] = set()
    for index, vessel in enumerate(config["vessels"]):
        if not isinstance(vessel, dict):
            raise ValueError(f"vessels[{index}] must be an object")
        for key in ("vessel_id", "vessel_name", "zero_d_element_type", "zero_d_element_values"):
            if key not in vessel:
                raise ValueError(f"vessels[{index}] is missing required field '{key}'")
        if isinstance(vessel["vessel_id"], bool) or not isinstance(
            vessel["vessel_id"], int
        ):
            raise ValueError(f"vessels[{index}].vessel_id must be an integer")
        vessel_id = vessel["vessel_id"]
        if not isinstance(vessel["vessel_name"], str) or not vessel["vessel_name"]:
            raise ValueError(f"vessels[{index}].vessel_name must be a non-empty string")
        vessel_name = vessel["vessel_name"]
        if vessel_id in vessel_ids:
            raise ValueError(f"duplicate vessel_id in normalized solver configuration: {vessel_id}")
        if vessel_name in vessel_names:
            raise ValueError(f"duplicate vessel_name in normalized solver configuration: {vessel_name}")
        vessel_ids.add(vessel_id)
        vessel_names.add(vessel_name)
        if not isinstance(vessel["zero_d_element_values"], dict):
            raise ValueError(f"vessels[{index}].zero_d_element_values must be an object")
        if not isinstance(vessel["zero_d_element_type"], str):
            raise ValueError(f"vessels[{index}].zero_d_element_type must be a string")
        if "calibrate" in vessel:
            raise ValueError(f"vessels[{index}] retains calibration-only field 'calibrate'")

    junction_names: set[str] = set()
    for index, junction in enumerate(config["junctions"]):
        if not isinstance(junction, dict):
            raise ValueError(f"junctions[{index}] must be an object")
        for key in ("junction_name", "junction_type"):
            if key not in junction:
                raise ValueError(f"junctions[{index}] is missing required field '{key}'")
        if not isinstance(junction["junction_name"], str) or not junction["junction_name"]:
            raise ValueError(
                f"junctions[{index}].junction_name must be a non-empty string"
            )
        junction_name = junction["junction_name"]
        if junction_name in junction_names:
            raise ValueError(
                f"duplicate junction_name in normalized solver configuration: {junction_name}"
            )
        junction_names.add(junction_name)
        if not isinstance(junction["junction_type"], str):
            raise ValueError(f"junctions[{index}].junction_type must be a string")
        junction_type = junction["junction_type"]
        if junction_type not in _SUPPORTED_JUNCTION_TYPES:
            raise ValueError(
                f"unsupported junction type for {junction_name}: {junction_type}"
            )
        if "calibrate" in junction:
            raise ValueError(
                f"junctions[{index}] retains calibration-only field 'calibrate'"
            )
        vessel_connection_keys = ("inlet_vessels", "outlet_vessels")
        block_connection_keys = ("inlet_blocks", "outlet_blocks")
        has_vessel_connections = any(key in junction for key in vessel_connection_keys)
        has_block_connections = any(key in junction for key in block_connection_keys)
        if has_vessel_connections and has_block_connections:
            raise ValueError(
                f"junction {junction_name} mixes vessel and block connections"
            )
        if not has_vessel_connections and not has_block_connections:
            raise ValueError(f"junction {junction_name} has no connections")
        connection_keys = (
            vessel_connection_keys if has_vessel_connections else block_connection_keys
        )
        if any(key not in junction for key in connection_keys):
            raise ValueError(
                f"junction {junction_name} must define both inlet and outlet connections"
            )
        for connection_key in connection_keys:
            connections = junction[connection_key]
            if not isinstance(connections, list) or not connections:
                raise ValueError(
                    f"junction {junction_name} {connection_key} must be a non-empty list"
                )
            if connection_key.endswith("_vessels"):
                try:
                    unknown_ids = sorted(
                        int(vessel_id)
                        for vessel_id in connections
                        if int(vessel_id) not in vessel_ids
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"junction {junction_name} {connection_key} must contain integer IDs"
                    ) from exc
                if unknown_ids:
                    raise ValueError(
                        f"junction {junction_name} references unknown vessel IDs: {unknown_ids}"
                    )
            elif not all(isinstance(block_name, str) and block_name for block_name in connections):
                raise ValueError(
                    f"junction {junction_name} {connection_key} must contain block names"
                )
        if "junction_values" in junction and not isinstance(junction["junction_values"], dict):
            raise ValueError(f"junction {junction_name}.junction_values must be an object")
        if junction_type == "NORMAL_JUNCTION" and junction.get("junction_values"):
            raise ValueError(
                f"junction {junction_name}.NORMAL_JUNCTION cannot contain junction_values"
            )
        if junction_type != "NORMAL_JUNCTION" and not junction.get("junction_values"):
            raise ValueError(
                f"junction {junction_name}.{junction_type} requires junction_values"
            )
        if junction_type == "BloodVesselJunction" and "junction_values" in junction:
            outlet_count = _junction_connection_count(junction, "outlet_vessels")
            if outlet_count is not None:
                for parameter, values in junction["junction_values"].items():
                    if not isinstance(values, list) or len(values) != outlet_count:
                        raise ValueError(
                            f"junction {junction_name}.junction_values.{parameter} must "
                            f"contain one value per outlet ({outlet_count})"
                        )

    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalized solver configuration is not valid JSON: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _solver_result_table(result: Any) -> Dict[str, np.ndarray]:
    """Convert solver pandas output or a row-oriented test double to columns."""
    columns = getattr(result, "columns", None)
    if columns is not None:
        column_names = [str(column) for column in columns]
        table: Dict[str, np.ndarray] = {}
        for column_name, source_name in zip(column_names, columns):
            try:
                values = result[source_name]
            except Exception as exc:
                raise ValueError(f"solver replay result column '{column_name}' is unavailable") from exc
            table[column_name] = np.asarray(values)
        return table

    if isinstance(result, list) and all(isinstance(row, dict) for row in result):
        column_names = sorted({str(key) for row in result for key in row})
        return {
            column_name: np.asarray([row.get(column_name) for row in result])
            for column_name in column_names
        }

    if isinstance(result, dict) and result:
        if not all(isinstance(values, (list, tuple, np.ndarray)) for values in result.values()):
            raise ValueError("solver replay result must be a table of numeric columns")
        return {str(key): np.asarray(values) for key, values in result.items()}

    raise ValueError("solver replay returned no tabular result")


def _replay_series(
    result: Any,
) -> list[Dict[str, Any]]:
    """Extract pressure and flow traces from vessel- or variable-based output."""
    table = _solver_result_table(result)
    if "name" not in table or "time" not in table:
        raise ValueError("solver replay result must include name and time columns")
    row_count = len(table["name"])
    if len(table["time"]) != row_count:
        raise ValueError("solver replay result name and time columns have different lengths")

    try:
        times = np.asarray(table["time"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("solver replay result time column must be numeric") from exc
    if not np.isfinite(times).all():
        raise ValueError("solver replay result contains non-finite time values")

    names = table["name"]
    candidates: list[tuple[str, str, np.ndarray]] = []
    for column_name, values in table.items():
        lower_name = column_name.lower()
        if lower_name in {"name", "time", "y", "ydot"}:
            continue
        if lower_name.startswith("d_") or lower_name.startswith("dflow") or lower_name.startswith("dpressure"):
            continue
        if "flow" in lower_name:
            candidates.append((column_name, "flow", values))
        elif "pressure" in lower_name:
            candidates.append((column_name, "pressure", values))

    # Variable-based output stores the semantic kind in name and the numeric
    # values in y.  It is handled separately because y is otherwise ambiguous.
    if "y" in table:
        variable_values = table["y"]
        for raw_name in sorted({str(name) for name in names}):
            name = str(raw_name)
            kind = name.split(":", 1)[0].lower()
            if kind in {"flow", "pressure"}:
                candidates.append((f"variable:{name}", kind, variable_values))

    if not candidates:
        raise ValueError("solver replay result contains no pressure or flow values")

    series: list[Dict[str, Any]] = []
    for column_name, kind, raw_values in candidates:
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"solver replay result {kind} values in '{column_name}' must be numeric"
            ) from exc
        if values.ndim != 1 or len(values) != row_count:
            raise ValueError(
                f"solver replay result {kind} values in '{column_name}' must match result rows"
            )
        if values.size == 0:
            raise ValueError(
                f"solver replay result {kind} values in '{column_name}' are empty"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"solver replay result contains non-finite {kind} values in '{column_name}'"
            )

        groups: Dict[str, list[int]] = {}
        if column_name.startswith("variable:"):
            for index, raw_name in enumerate(names):
                variable_name = str(raw_name)
                if variable_name == column_name[len("variable:") :]:
                    groups.setdefault(variable_name, []).append(index)
        else:
            for index, raw_name in enumerate(names):
                groups.setdefault(str(raw_name), []).append(index)
        for series_name, indices in sorted(groups.items()):
            raw_times = times[indices]
            if raw_times.size > 1 and np.any(np.diff(raw_times) < 0.0):
                raise ValueError(
                    f"solver replay times are not ordered for {kind} series '{series_name}'"
                )
            order = np.argsort(times[indices], kind="mergesort")
            ordered_indices = np.asarray(indices, dtype=np.int64)[order]
            ordered_times = times[ordered_indices]
            ordered_values = values[ordered_indices]
            if ordered_times.size == 0:
                continue
            series.append(
                {
                    "name": series_name,
                    "kind": kind,
                    "times": ordered_times,
                    "values": ordered_values,
                }
            )
    return series


def _observation_scales(payload: Dict[str, Any]) -> Dict[str, float]:
    scales = {"pressure": 0.0, "flow": 0.0}
    observations = payload.get("y")
    if not isinstance(observations, dict):
        raise ValueError("calibration payload observations are required for replay scales")
    for variable_name, raw_values in observations.items():
        kind = str(variable_name).split(":", 1)[0].lower()
        if kind not in scales:
            continue
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration observation '{variable_name}' must be numeric") from exc
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"calibration observation '{variable_name}' is empty or non-finite")
        scales[kind] = max(scales[kind], float(np.max(np.abs(values))))
    return {
        kind: max(scale, _REPLAY_SCALE_FLOOR)
        for kind, scale in scales.items()
    }


def _build_replay_payload(
    published_config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Make an all-cycles validation copy without changing the published config."""
    replay_payload = copy.deepcopy(published_config)
    simparams = replay_payload["simulation_parameters"]
    requested_cycles = int(simparams["number_of_cardiac_cycles"])
    points_per_cycle = int(simparams["number_of_time_pts_per_cardiac_cycle"])
    if bool(simparams.get("coupled_simulation", False)):
        raise ValueError(
            "calibrated replay stability requires a non-coupled cardiac-cycle configuration"
        )

    replay_cycles = max(2, requested_cycles)
    simparams["number_of_cardiac_cycles"] = replay_cycles
    simparams["output_all_cycles"] = True
    simparams["output_mean_only"] = False
    simparams["output_derivative"] = False
    simparams["output_interval"] = 1
    return replay_payload, {
        "requested_number_of_cardiac_cycles": requested_cycles,
        "validation_number_of_cardiac_cycles": replay_cycles,
        "number_of_time_pts_per_cardiac_cycle": points_per_cycle,
        "output_all_cycles": True,
        "output_mean_only": False,
        "output_derivative": False,
        "output_interval": 1,
    }


def _evaluate_replay(
    *,
    result: Any,
    payload: Dict[str, Any],
    replay_settings: Dict[str, float],
    validation_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Check finite/bounded solver output and final two-cycle stability."""
    series = _replay_series(result)
    scales = _observation_scales(payload)
    max_abs = {"pressure": 0.0, "flow": 0.0}
    per_series: list[Dict[str, Any]] = []
    points_per_cycle = int(validation_settings["number_of_time_pts_per_cardiac_cycle"])
    cycle_span = points_per_cycle - 1
    minimum_rows = 2 * cycle_span + 1
    if points_per_cycle < 2:
        raise ValueError(
            "simulation_parameters.number_of_time_pts_per_cardiac_cycle must be at least 2"
        )

    for item in series:
        kind = item["kind"]
        values = item["values"]
        max_abs[kind] = max(max_abs[kind], float(np.max(np.abs(values))))
        bound_limit = replay_settings[f"{kind}_bound_multiplier"] * scales[kind]
        bounded = bool(np.max(np.abs(values)) <= bound_limit)
        if values.size < minimum_rows:
            per_series.append(
                {
                    "name": item["name"],
                    "kind": kind,
                    "sample_count": int(values.size),
                    "bounded": bounded,
                    "cycle_stability_relative_rms": None,
                    "cycle_stability_passed": False,
                    "error": (
                        f"expected at least {minimum_rows} samples for two complete "
                        f"cycles, received {values.size}"
                    ),
                }
            )
            continue

        final_start = values.size - (cycle_span + 1)
        preceding_start = final_start - cycle_span
        preceding = values[preceding_start : preceding_start + points_per_cycle]
        final = values[final_start : final_start + points_per_cycle]
        normalized_rms = _rms(final - preceding) / max(scales[kind], _REPLAY_SCALE_FLOOR)
        per_series.append(
            {
                "name": item["name"],
                "kind": kind,
                "sample_count": int(values.size),
                "bounded": bounded,
                "maximum_absolute_value": float(np.max(np.abs(values))),
                "bound_limit": float(bound_limit),
                "cycle_stability_relative_rms": float(normalized_rms),
                "cycle_stability_passed": bool(
                    normalized_rms <= replay_settings["cycle_stability_tolerance"]
                ),
                "final_cycle_start_time": float(item["times"][final_start]),
                "final_cycle_end_time": float(item["times"][final_start + points_per_cycle - 1]),
            }
        )

    pressure_series = [item for item in per_series if item["kind"] == "pressure"]
    flow_series = [item for item in per_series if item["kind"] == "flow"]
    bounds = {
        kind: {
            "observation_scale": float(scales[kind]),
            "multiplier": float(replay_settings[f"{kind}_bound_multiplier"]),
            "limit": float(
                replay_settings[f"{kind}_bound_multiplier"] * scales[kind]
            ),
            "maximum_absolute_value": float(max_abs[kind]),
            "passed": bool(
                max_abs[kind]
                <= replay_settings[f"{kind}_bound_multiplier"] * scales[kind]
            ),
        }
        for kind in ("pressure", "flow")
    }
    stability = {
        "tolerance": float(replay_settings["cycle_stability_tolerance"]),
        "maximum_pressure_relative_rms": max(
            (
                item["cycle_stability_relative_rms"]
                for item in pressure_series
                if item["cycle_stability_relative_rms"] is not None
            ),
            default=None,
        ),
        "maximum_flow_relative_rms": max(
            (
                item["cycle_stability_relative_rms"]
                for item in flow_series
                if item["cycle_stability_relative_rms"] is not None
            ),
            default=None,
        ),
        "passed": bool(
            pressure_series
            and flow_series
            and all(item["cycle_stability_passed"] for item in per_series)
        ),
    }
    checks = {
        "finite_pressure_and_flow": True,
        "bounded_pressure": bounds["pressure"]["passed"],
        "bounded_flow": bounds["flow"]["passed"],
        "cycle_stability": stability["passed"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "validation_settings": validation_settings,
        "bounds": bounds,
        "cycle_stability": stability,
        "series": sorted(per_series, key=lambda item: (item["kind"], item["name"])),
    }


def _write_json_atomically(path: Path, payload: Dict[str, Any], *, indent: int) -> None:
    """Publish one JSON document with a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=indent, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _json_digest(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_digest(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _run_identity(
    *,
    run_id: str,
    normalized_input: Dict[str, Any],
    assembly: CalibrationAssembly,
    calibration: CalibrationConfig | None,
) -> Dict[str, Any]:
    metadata_path = (
        calibration.data_source.metadata_json if calibration is not None else None
    )
    observations = {
        "y": assembly.solver_payload.get("y"),
        "dy": assembly.solver_payload.get("dy"),
        "targets": assembly.target_observations,
        "target_phases": assembly.target_phases,
        "target_units": assembly.target_units,
        "metadata_digest": _path_digest(metadata_path),
    }
    digests = {
        "normalized_input": _json_digest(normalized_input),
        "observations": _json_digest(observations),
        "observation_metadata": observations["metadata_digest"],
        "solver_module": None,
        "output": None,
    }
    return {
        "run_id": run_id,
        "digests": digests,
        # Keep flat aliases for consumers that do not want to know the report
        # envelope shape.  All reports receive the exact same mapping.
        "normalized_input_digest": digests["normalized_input"],
        "input_digest": digests["normalized_input"],
        "observation_digest": digests["observations"],
        "observation_metadata_digest": digests["observation_metadata"],
        "metadata_digest": digests["observation_metadata"],
        "solver_module_digest": digests["solver_module"],
        "solver_digest": digests["solver_module"],
        "solver_module_sha256": digests["solver_module"],
        "output_digest": digests["output"],
        "output_config_digest": digests["output"],
    }


def _set_solver_digest(run_identity: Dict[str, Any], provenance: Any) -> None:
    if not isinstance(provenance, dict):
        return
    digest = provenance.get("module_sha256")
    if digest is None:
        return
    digests = run_identity["digests"]
    digests["solver_module"] = str(digest)
    run_identity["solver_module_digest"] = str(digest)
    run_identity["solver_digest"] = str(digest)
    run_identity["solver_module_sha256"] = str(digest)


def _replay_target_observations(
    replay_summary: Dict[str, Any],
    targets: Any,
    target_observations_3d: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Map an accepted replay cycle to explicit MPA/LPA/RPA target series."""
    accepted = replay_summary.get("accepted_final_cycle")
    if not isinstance(accepted, dict):
        raise ValueError("settled replay did not provide an accepted final cycle")
    series = accepted.get("series")
    if not isinstance(series, list):
        raise ValueError("settled replay accepted cycle has no series")
    start = float(accepted.get("start_time"))
    end = float(accepted.get("end_time"))
    duration = end - start
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("settled replay accepted cycle has invalid period")
    def pick(vessel: str, endpoint: str, kind: str) -> Dict[str, Any]:
        expected_column = f"{kind}_{'in' if endpoint == 'upstream' else 'out'}"
        role = (
            "mpa_pressure"
            if kind == "pressure"
            else ("rpa_flow" if vessel == str(targets.rpa_flow_split.rpa_vessel) else "lpa_flow")
        )
        expected_variable = (target_observations_3d or {}).get(role, {}).get("variable")
        candidates = [
            item
            for item in series
            if str(item.get("kind")) == kind
            and (
                str(item.get("name")) == vessel
                or str(item.get("name")) == str(expected_variable)
                or str(item.get("name", "")).endswith(f":{vessel}")
            )
        ]
        selected = next(
            (item for item in candidates if str(item.get("column")) == expected_column),
            None,
        )
        if selected is None and len(candidates) == 1:
            selected = candidates[0]
        if not isinstance(selected, dict):
            raise ValueError(
                f"settled replay has no {kind} series for target vessel {vessel!r} "
                f"at {endpoint} endpoint"
            )
        times = np.asarray(selected.get("times"), dtype=np.float64)
        values = np.asarray(selected.get("values"), dtype=np.float64)
        if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
            raise ValueError(f"settled replay target series for {vessel!r} is malformed")
        local_phases = (times - start) / duration
        return {
            "phases": [float(value) for value in local_phases],
            "values": [float(value) for value in values],
            "units": "mmHg" if kind == "pressure" else "cm^3/s",
            "orientation": "away_from_mpa",
        }

    mpa = targets.mpa_pressure
    split = targets.rpa_flow_split
    # Endpoint resolution is retained in the 3D target records.  For replay,
    # topology has already been validated and the configured interface remains
    # the authoritative selector.
    def endpoint(value: Any, role: str) -> str:
        interface = str(value).lower()
        if interface in {"upstream", "external_upstream"}:
            return "upstream"
        if interface in {"downstream", "external_downstream"}:
            return "downstream"
        resolved = (target_observations_3d or {}).get(role, {}).get("endpoint")
        if resolved in {"upstream", "downstream"}:
            return str(resolved)
        raise ValueError(
            "settled replay target extraction requires an explicit upstream or downstream interface"
        )

    return {
        "mpa_pressure": pick(str(mpa.vessel), endpoint(mpa.interface, "mpa_pressure"), "pressure"),
        "rpa_flow": pick(str(split.rpa_vessel), endpoint(split.interface, "rpa_flow"), "flow"),
        "lpa_flow": pick(str(split.lpa_vessel), endpoint(split.interface, "lpa_flow"), "flow"),
    }


def _evaluate_target_replay(
    *,
    replay_summary: Dict[str, Any],
    assembly: CalibrationAssembly,
    calibration: CalibrationConfig,
) -> Dict[str, Any]:
    if calibration.targets is None:
        return {"status": "skipped", "reason": "calibration.targets is not configured"}
    if replay_summary.get("status") != "pass":
        return {
            "status": "fail",
            "reason": "settled replay did not pass",
            "replay_status": replay_summary.get("status"),
        }
    candidate = _replay_target_observations(
        replay_summary,
        calibration.targets,
        assembly.target_observations,
    )
    evaluation = evaluate_pulmonary_targets(
        assembly.target_observations,
        candidate,
        calibration.targets,
    )
    return {"status": "pass" if evaluation.passed else "fail", **evaluation.as_dict()}


def _run_settled_replay(
    *,
    config: Dict[str, Any],
    payload: Dict[str, Any],
    calibration: CalibrationConfig,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Simulate one validation copy and return its settled replay report."""
    solver = calibration.solver
    replay_payload, validation_settings = build_settled_replay_payload(
        config,
        minimum_cycles=solver.replay_minimum_cycles,
        maximum_cycles=solver.replay_maximum_cycles,
        required_consecutive_stable_pairs=solver.required_consecutive_stable_pairs,
    )
    replay_settings = _replay_settings(calibration)
    combined_settings = {**validation_settings, **replay_settings}
    try:
        simulated = simulate_pysvzerod(replay_payload)
    except Exception as exc:
        return (
            {
                "status": "fail",
                "checks": {
                    "finite_pressure_and_flow": False,
                    "cycle_boundaries": False,
                    "cycle_count": False,
                    "bounded_pressure": False,
                    "bounded_flow": False,
                    "cycle_stability": False,
                },
                "validation_settings": combined_settings,
                "error": f"simulate failed: {type(exc).__name__}: {exc}",
            },
            validation_settings,
        )
    replay_summary = validate_settled_replay(
        simulated,
        payload=payload,
        validation_settings=combined_settings,
    )
    replay_summary["replay_settings"] = replay_settings
    return replay_summary, validation_settings


def calibrate_0d_from_mapped_centerline(
    *,
    zerod_config_path: str,
    output_config_path: str,
    calibration: CalibrationConfig,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex
    assembly = assemble_calibration_payload(
        zerod_config_path=zerod_config_path,
        calibration=calibration,
    )
    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        normalized_input, _input_report = _normalize_calibrated_config(
            assembly.solver_payload
        )
    except (TypeError, ValueError):
        # QC failures must still leave a useful run identity even when the
        # mocked or incomplete payload cannot be normalized for simulation.
        normalized_input = assembly.solver_payload
    run_identity = _run_identity(
        run_id=run_id,
        normalized_input=normalized_input,
        assembly=assembly,
        calibration=calibration,
    )
    assembly.observation_qc.update(run_identity)
    qc_path = output_path.parent / "calibration_observation_qc.json"
    _write_json_atomically(qc_path, assembly.observation_qc, indent=2)
    if assembly.observation_qc.get("status") != "pass":
        failed_checks = assembly.observation_qc.get("failed_fatal_checks") or [
            name
            for name, passed in assembly.observation_qc.get("checks", {}).items()
            if not passed
        ]
        raise ValueError(
            "calibration observation QC failed before solver dispatch: "
            f"{', '.join(failed_checks)}; report: {qc_path}"
        )
    if calibration.targets is not None and not assembly.target_observations:
        target_path = output_path.parent / "calibration_targets.json"
        _write_json_atomically(
            target_path,
            {
                **run_identity,
                "status": "fail",
                "reason": (
                    "target-focused calibration requires periodic target observations "
                    "with explicit metadata units and cycle timing"
                ),
            },
            indent=2,
        )
        raise ValueError(
            "target-focused calibration requires periodic target observations; "
            f"report: {target_path}"
        )

    # Baseline evaluation is deliberately simulation-only.  It never feeds
    # values back into either calibrator invocation and an unstable baseline
    # simply disables the relative policy for the candidate.
    baseline_summary: Dict[str, Any] = {
        "status": "skipped",
        "reason": "calibration.targets is not configured",
    }
    baseline_config: Dict[str, Any] | None = None
    if calibration.targets is not None:
        try:
            baseline_config, _baseline_normalization = _normalize_calibrated_config(
                assembly.solver_payload
            )
            baseline_replay, _baseline_validation = _run_settled_replay(
                config=baseline_config,
                payload=assembly.solver_payload,
                calibration=calibration,
            )
            baseline_summary = {
                "status": baseline_replay.get("status", "fail"),
                "replay": baseline_replay,
            }
            baseline_summary["targets"] = _evaluate_target_replay(
                replay_summary=baseline_replay,
                assembly=assembly,
                calibration=calibration,
            )
        except (TypeError, ValueError) as exc:
            baseline_summary["status"] = "fail"
            baseline_summary["reason"] = f"baseline evaluation failed: {exc}"
            baseline_summary["targets"] = {
                "status": "fail",
                "reason": f"baseline evaluation failed: {exc}",
            }

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
    _set_solver_digest(run_identity, first_provenance)

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
    _set_solver_digest(run_identity, confirmation_provenance)

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
        **run_identity,
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
    _write_json_atomically(confirmation_path, confirmation_summary, indent=2)

    published, output_normalization = _normalize_calibrated_config(
        confirmation_calibrated
    )
    _validate_publishable_solver_config(published)
    replay_path = output_path.parent / "calibration_replay.json"
    if calibration.targets is not None:
        try:
            replay_summary, _validation_settings = _run_settled_replay(
                config=published,
                payload=assembly.solver_payload,
                calibration=calibration,
            )
        except (TypeError, ValueError) as exc:
            replay_summary = {
                "status": "fail",
                "checks": {
                    "finite_pressure_and_flow": False,
                    "cycle_boundaries": False,
                    "cycle_count": False,
                    "bounded_pressure": False,
                    "bounded_flow": False,
                    "cycle_stability": False,
                },
                "error": f"replay setup failed: {exc}",
            }
    else:
        replay_settings = _replay_settings(calibration)
        replay_payload, validation_settings = _build_replay_payload(published)
        try:
            replay_result = simulate_pysvzerod(replay_payload)
            replay_summary = _evaluate_replay(
                result=replay_result,
                payload=assembly.solver_payload,
                replay_settings=replay_settings,
                validation_settings=validation_settings,
            )
        except ValueError as exc:
            replay_summary = {
                "status": "fail",
                "checks": {
                    "finite_pressure_and_flow": False,
                    "bounded_pressure": False,
                    "bounded_flow": False,
                    "cycle_stability": False,
                },
                "validation_settings": validation_settings,
                "replay_settings": replay_settings,
                "error": str(exc),
            }
    replay_summary = {**run_identity, **replay_summary}
    _write_json_atomically(replay_path, replay_summary, indent=2)
    if replay_summary["status"] != "pass":
        failed_checks = [
            name
            for name, passed in replay_summary.get("checks", {}).items()
            if not passed
        ]
        if calibration.targets is not None:
            _write_json_atomically(
                output_path.parent / "calibration_targets.json",
                {
                    **run_identity,
                    "status": "fail",
                    "reason": "settled replay failed before target evaluation",
                    "baseline": baseline_summary,
                },
                indent=2,
            )
        raise ValueError(
            "calibrated solver replay stability checks failed: "
            + ", ".join(failed_checks)
            + f"; report: {replay_path}"
        )

    target_summary: Dict[str, Any] = {
        **run_identity,
        "status": "skipped",
        "reason": "calibration.targets is not configured",
        "baseline": baseline_summary,
    }
    if calibration.targets is not None:
        target_summary["configuration"] = {
            "mpa_pressure": {
                "vessel": calibration.targets.mpa_pressure.vessel,
                "interface": calibration.targets.mpa_pressure.interface,
                "weight": float(calibration.targets.mpa_pressure.weight),
                "normalized_rms_tolerance": float(
                    calibration.targets.mpa_pressure.normalized_rms_tolerance
                ),
            },
            "rpa_flow_split": {
                "rpa_vessel": calibration.targets.rpa_flow_split.rpa_vessel,
                "lpa_vessel": calibration.targets.rpa_flow_split.lpa_vessel,
                "interface": calibration.targets.rpa_flow_split.interface,
                "weight": float(calibration.targets.rpa_flow_split.weight),
                "absolute_tolerance": float(
                    calibration.targets.rpa_flow_split.absolute_tolerance
                ),
            },
            "require_improvement_over_baseline": bool(
                calibration.targets.require_improvement_over_baseline
            ),
        }
        target_summary["observations_3d"] = assembly.target_observations
        target_summary["target_phases"] = assembly.target_phases
        target_summary["target_units"] = assembly.target_units
        try:
            candidate_targets = _evaluate_target_replay(
                replay_summary=replay_summary,
                assembly=assembly,
                calibration=calibration,
            )
        except (TypeError, ValueError) as exc:
            candidate_targets = {
                "status": "fail",
                "reason": f"target evaluation failed: {exc}",
            }
        baseline_targets = baseline_summary.get("targets") or {}
        baseline_replay_passed = baseline_summary.get("replay", {}).get("status") == "pass"
        baseline_score = baseline_targets.get("composite_score")
        candidate_score = candidate_targets.get("composite_score")
        require_improvement = bool(
            calibration.targets.require_improvement_over_baseline
        )
        if not baseline_replay_passed or not isinstance(baseline_score, (int, float)):
            baseline_policy = {
                "required": require_improvement,
                "status": "not_applied",
                "reason": "baseline replay/target score was unavailable",
                "passed": True,
            }
        elif not require_improvement:
            baseline_policy = {
                "required": False,
                "status": "disabled",
                "baseline_composite_score": float(baseline_score),
                "candidate_composite_score": candidate_score,
                "passed": True,
            }
        else:
            policy_passed = isinstance(candidate_score, (int, float)) and bool(
                float(candidate_score) <= float(baseline_score)
            )
            baseline_policy = {
                "required": True,
                "status": "pass" if policy_passed else "fail",
                "baseline_composite_score": float(baseline_score),
                "candidate_composite_score": candidate_score,
                "passed": policy_passed,
            }
        target_summary.update(
            {
                "status": "pass"
                if candidate_targets.get("status") == "pass"
                and baseline_policy["passed"]
                else "fail",
                "baseline": baseline_summary,
                "candidate": candidate_targets,
                "baseline_policy": baseline_policy,
                "component_gates": candidate_targets.get("gate_results", {}),
            }
        )
        target_path = output_path.parent / "calibration_targets.json"
        _write_json_atomically(target_path, target_summary, indent=2)
        if target_summary["status"] != "pass":
            raise ValueError(
                "calibrated pulmonary target gates failed; report: "
                f"{target_path}"
            )

    # A target report is part of the stable artifact set even for legacy
    # configurations where target scoring is intentionally disabled.
    target_path = output_path.parent / "calibration_targets.json"
    baseline_summary.update(run_identity)
    if isinstance(baseline_summary.get("replay"), dict):
        baseline_summary["replay"].update(run_identity)
    if isinstance(baseline_summary.get("targets"), dict):
        baseline_summary["targets"].update(run_identity)
    if not target_path.exists():
        _write_json_atomically(target_path, target_summary, indent=2)

    run_identity["digests"]["output"] = _json_digest(published)
    run_identity["output_digest"] = run_identity["digests"]["output"]
    run_identity["output_config_digest"] = run_identity["digests"]["output"]
    baseline_summary.update(run_identity)
    if isinstance(baseline_summary.get("replay"), dict):
        baseline_summary["replay"].update(run_identity)
    if isinstance(baseline_summary.get("targets"), dict):
        baseline_summary["targets"].update(run_identity)
    assembly.observation_qc.update(run_identity)
    confirmation_summary.update(run_identity)
    replay_summary.update(run_identity)
    _write_json_atomically(qc_path, assembly.observation_qc, indent=2)
    _write_json_atomically(confirmation_path, confirmation_summary, indent=2)
    _write_json_atomically(replay_path, replay_summary, indent=2)
    target_summary.update(run_identity)
    target_summary["baseline"] = baseline_summary
    _write_json_atomically(target_path, target_summary, indent=2)

    summary = {
        **run_identity,
        "status": "ok",
        "output_config": str(output_path),
        "observation_count": assembly.observation_count,
        "variable_count": assembly.variable_count,
        "solver_provenance": confirmation_provenance,
        "input_normalization": assembly.input_normalization,
        "observation_qc": assembly.observation_qc,
        "interface_sampling": assembly.interface_sampling,
        "exclusions": assembly.excluded_blocks,
        "calibrated_output_normalization": output_normalization,
        "calibration_confirmation": confirmation_summary,
        "replay_stability": replay_summary,
        "calibration_targets": target_summary,
        "target_quality": target_summary,
    }
    summary_path = output_path.parent / "calibration_summary.json"
    _write_json_atomically(summary_path, summary, indent=2)

    # The only operation that publishes the solver config is the final atomic
    # replacement.  Every validation and replay check above runs on copies.
    _write_json_atomically(output_path, published, indent=4)

    return {
        **run_identity,
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
        "calibrated_output_normalization": output_normalization,
        "replay_stability": replay_summary,
        "calibration_replay_report": str(replay_path),
        "calibration_targets": target_summary,
        "calibration_targets_report": str(target_path),
        "target_quality": target_summary,
        "calibration_summary": summary,
        "calibration_summary_report": str(summary_path),
        "solver_provenance": confirmation_provenance,
    }
