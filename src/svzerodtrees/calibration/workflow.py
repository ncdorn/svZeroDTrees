from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .._pysvzerod import calibrate_pysvzerod
from ..config import CalibrationConfig

_VESSEL_NAME_RE = re.compile(r"^branch(?P<branch_id>\d+)_seg(?P<seg_id>\d+)$")


@dataclass
class BranchObservation:
    proximal_pressure: float
    distal_pressure: float
    proximal_flow: float
    distal_flow: float


@dataclass
class CalibrationAssembly:
    solver_payload: Dict[str, Any]
    observation_count: int
    variable_count: int


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


def _require_finite_scalar(values: Iterable[float], *, label: str) -> list[float]:
    resolved = [float(value) for value in values]
    if any(not np.isfinite(value) for value in resolved):
        raise ValueError(f"{label} contains non-finite values")
    return resolved


def _branch_observations_from_mapped_centerline(
    *,
    centerline_path: str,
    mapped_centerline_path: str,
    pressure_array: str,
    flow_array: str,
    branch_id_array: str,
    path_array: str,
) -> Dict[int, BranchObservation]:
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

    pressure = _point_array(mapped_poly, pressure_array)
    flow = _point_array(mapped_poly, flow_array)

    observations: Dict[int, BranchObservation] = {}
    unique_branch_ids = np.unique(branch_ids[np.isfinite(branch_ids)])
    for raw_branch_id in unique_branch_ids:
        branch_id = int(round(float(raw_branch_id)))
        indices = np.flatnonzero(np.isclose(branch_ids, raw_branch_id))
        if indices.size == 0:
            continue

        branch_paths = paths[indices]
        order = np.argsort(branch_paths, kind="mergesort")
        sorted_indices = indices[order]

        pressure_values = pressure[sorted_indices]
        flow_values = flow[sorted_indices]

        if pressure_values.size == 0 or flow_values.size == 0:
            raise ValueError(f"branch {branch_id} has no mapped observation samples")

        proximal_pressure, distal_pressure = _require_finite_scalar(
            [pressure_values[0], pressure_values[-1]],
            label=f"pressure samples for branch {branch_id}",
        )
        proximal_flow, distal_flow = _require_finite_scalar(
            [flow_values[0], flow_values[-1]],
            label=f"flow samples for branch {branch_id}",
        )

        observations[branch_id] = BranchObservation(
            proximal_pressure=proximal_pressure,
            distal_pressure=distal_pressure,
            proximal_flow=proximal_flow,
            distal_flow=distal_flow,
        )

    if not observations:
        raise ValueError("no branch observations could be assembled from the mapped centerline result")
    return observations


def _branch_id_for_vessel(vessel_name: str) -> int:
    match = _VESSEL_NAME_RE.match(vessel_name)
    if match is None:
        raise ValueError(
            "stage-1 calibration requires vessel names of the form 'branch<id>_seg0'; "
            f"received '{vessel_name}'"
        )
    seg_id = int(match.group("seg_id"))
    if seg_id != 0:
        raise ValueError(
            "stage-1 calibration currently supports one 0D vessel per centerline branch; "
            f"received multi-segment vessel '{vessel_name}'"
        )
    return int(match.group("branch_id"))


def _network_topology(config: Dict[str, Any]) -> tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    vessel_id_to_name: Dict[int, str] = {}
    vessel_branch_ids: Dict[str, int] = {}
    upstream_names: Dict[str, str] = {}
    downstream_names: Dict[str, str] = {}

    for vessel in config.get("vessels", []) or []:
        vessel_name = str(vessel["vessel_name"])
        vessel_id = int(vessel["vessel_id"])
        branch_id = _branch_id_for_vessel(vessel_name)
        if branch_id in vessel_branch_ids.values():
            raise ValueError(
                "stage-1 calibration currently supports exactly one vessel per centerline branch"
            )
        vessel_id_to_name[vessel_id] = vessel_name
        vessel_branch_ids[vessel_name] = branch_id

        boundary_conditions = vessel.get("boundary_conditions") or {}
        if boundary_conditions.get("inlet"):
            upstream_names[vessel_name] = str(boundary_conditions["inlet"])
        if boundary_conditions.get("outlet"):
            downstream_names[vessel_name] = str(boundary_conditions["outlet"])

    for junction in config.get("junctions", []) or []:
        junction_name = str(junction["junction_name"])
        for vessel_id in junction.get("inlet_vessels", []) or []:
            vessel_name = vessel_id_to_name[int(vessel_id)]
            downstream_names[vessel_name] = junction_name
        for vessel_id in junction.get("outlet_vessels", []) or []:
            vessel_name = vessel_id_to_name[int(vessel_id)]
            upstream_names[vessel_name] = junction_name

    missing_upstream = sorted(name for name in vessel_branch_ids if name not in upstream_names)
    missing_downstream = sorted(name for name in vessel_branch_ids if name not in downstream_names)
    if missing_upstream or missing_downstream:
        details = []
        if missing_upstream:
            details.append(f"missing upstream connection for {missing_upstream}")
        if missing_downstream:
            details.append(f"missing downstream connection for {missing_downstream}")
        raise ValueError("; ".join(details))

    return vessel_branch_ids, upstream_names, downstream_names


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
    return {
        name: list(overrides.get(name, default))
        for name in valid_names
    }


def assemble_calibration_payload(
    *,
    zerod_config_path: str,
    calibration: CalibrationConfig,
) -> CalibrationAssembly:
    with open(zerod_config_path, "r", encoding="utf-8") as stream:
        solver_config = json.load(stream)

    observations = _branch_observations_from_mapped_centerline(
        centerline_path=calibration.data_source.centerline or "",
        mapped_centerline_path=calibration.data_source.mapped_centerline_result or "",
        pressure_array=calibration.data_source.pressure_array,
        flow_array=calibration.data_source.flow_array,
        branch_id_array=calibration.data_source.branch_id_array,
        path_array=calibration.data_source.path_array,
    )
    vessel_branch_ids, upstream_names, downstream_names = _network_topology(solver_config)

    y: Dict[str, list[float]] = {}
    dy: Dict[str, list[float]] = {}
    for vessel_name, branch_id in vessel_branch_ids.items():
        if branch_id not in observations:
            raise ValueError(
                f"mapped centerline result does not contain observations for branch {branch_id} ({vessel_name})"
            )
        obs = observations[branch_id]
        upstream_name = upstream_names[vessel_name]
        downstream_name = downstream_names[vessel_name]

        variables = {
            f"flow:{upstream_name}:{vessel_name}": obs.proximal_flow,
            f"pressure:{upstream_name}:{vessel_name}": obs.proximal_pressure,
            f"flow:{vessel_name}:{downstream_name}": obs.distal_flow,
            f"pressure:{vessel_name}:{downstream_name}": obs.distal_pressure,
        }
        for variable_name, value in variables.items():
            y[variable_name] = [float(value)]
            dy[variable_name] = [0.0]

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
        observation_count=1,
        variable_count=len(y),
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
    calibrated = calibrate_pysvzerod(assembly.solver_payload)

    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(calibrated, stream, indent=4)

    return {
        "status": "ok",
        "output_config": str(output_path),
        "observation_count": assembly.observation_count,
        "variable_count": assembly.variable_count,
    }
