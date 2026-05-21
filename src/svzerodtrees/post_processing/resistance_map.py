from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

_FLOW_EPSILON = 1e-12
_AUTO_WORKER_CAP = 4


@dataclass
class _BranchGeometry:
    branch_id: int
    point_indices: np.ndarray
    length_cm: float
    warnings: List[str]


def _read_polydata(path: str | Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())
    return poly


def _write_polydata(poly: vtk.vtkPolyData, path: str | Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def _polydata_points(poly: vtk.vtkPolyData) -> np.ndarray:
    return np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=float)


def _point_array(poly: vtk.vtkPolyData, name: str) -> np.ndarray:
    point_data = poly.GetPointData()
    array = point_data.GetArray(name)
    if array is None:
        normalized_name = name.casefold()
        for index in range(point_data.GetNumberOfArrays()):
            candidate_name = point_data.GetArrayName(index)
            if candidate_name and candidate_name.casefold() == normalized_name:
                array = point_data.GetArray(index)
                break
    if array is None:
        available = [
            point_data.GetArrayName(index)
            for index in range(point_data.GetNumberOfArrays())
        ]
        raise ValueError(
            f"required point-data array '{name}' is missing; "
            f"available={available}"
        )
    data = np.asarray(vtk_to_numpy(array), dtype=float)
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:, 0]
    return data


def _coerce_scalar_array(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim == 1:
        return values.astype(float)
    if values.ndim == 2 and values.shape[1] >= 1:
        return values[:, 0].astype(float)
    raise ValueError(f"array '{name}' must be one-dimensional or single-component")


def _load_frames_csv(
    frames_csv: str | Path,
    *,
    require_existing_paths: bool = True,
) -> pd.DataFrame:
    path = Path(frames_csv)
    if not path.exists():
        raise FileNotFoundError(f"frames_csv not found: {path}")

    df = pd.read_csv(path)
    required = {"path", "time_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "frames_csv must contain columns: path,time_s; missing "
            + ",".join(sorted(missing))
        )

    has_timestep_id = "timestep_id" in df.columns
    records = []
    for row in df.itertuples(index=False):
        raw_path = Path(str(row.path)).expanduser()
        if not raw_path.is_absolute():
            raw_path = (path.parent / raw_path).resolve()
        if require_existing_paths and not raw_path.exists():
            raise FileNotFoundError(f"3D result frame not found: {raw_path}")
        time_s = float(row.time_s)
        if not math.isfinite(time_s):
            raise ValueError(f"frames_csv contains non-finite time_s for {raw_path}")
        record = {"path": str(raw_path), "time_s": time_s}
        if has_timestep_id:
            timestep_id = getattr(row, "timestep_id")
            if pd.isna(timestep_id):
                raise ValueError(f"frames_csv contains missing timestep_id for {raw_path}")
            timestep_value = float(timestep_id)
            if not math.isfinite(timestep_value) or not timestep_value.is_integer():
                raise ValueError(f"frames_csv contains invalid timestep_id for {raw_path}: {timestep_id}")
            record["timestep_id"] = int(timestep_value)
        records.append(record)

    sort_columns = ["time_s"]
    if has_timestep_id:
        sort_columns.append("timestep_id")
    return pd.DataFrame.from_records(records).sort_values(sort_columns).reset_index(drop=True)


def _selection_tolerance_s(frames: pd.DataFrame, cycle_duration_s: float) -> float:
    time_values = frames["time_s"].to_numpy(dtype=float)
    unique_times = np.unique(time_values)
    positive_diffs = np.diff(unique_times)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    if positive_diffs.size:
        dt = float(np.min(positive_diffs))
    else:
        dt = float(cycle_duration_s)
    scale = max(1.0, float(np.max(np.abs(time_values))), float(cycle_duration_s))
    return max(dt * 1e-6, np.finfo(float).eps * scale * 8.0)


def _select_last_cycle_frames(
    frames: pd.DataFrame,
    cycle_duration_s: float,
    *,
    return_metadata: bool = False,
    include_endpoint: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, float, float, float]:
    cycle_duration = float(cycle_duration_s)
    if not math.isfinite(cycle_duration) or cycle_duration <= 0.0:
        raise ValueError("cycle_duration_s must be positive and finite")
    t_end = float(frames["time_s"].max())
    window_start = t_end - cycle_duration
    tol = _selection_tolerance_s(frames, cycle_duration)
    upper_mask = (
        frames["time_s"] <= t_end + tol
        if include_endpoint
        else frames["time_s"] < t_end - tol
    )
    # By default the helper includes the terminal frame for simple direct use.
    # Workflow callers that need the final completed cycle pass
    # ``include_endpoint=False`` to use the half-open interval [t_end - T, t_end).
    selected = frames.loc[(frames["time_s"] >= window_start - tol) & upper_mask].copy()
    if selected.empty:
        raise ValueError("last full-cycle frame selection is empty")
    selected = selected.reset_index(drop=True)
    if not return_metadata:
        return selected
    return selected, window_start, t_end, tol


def _subsample_frames(frames: pd.DataFrame, max_frames: int | None) -> pd.DataFrame:
    if max_frames is None:
        return frames.reset_index(drop=True)
    limit = int(max_frames)
    if limit <= 0:
        raise ValueError("max_frames must be positive when provided")
    if len(frames) <= limit:
        return frames.reset_index(drop=True)
    indices = np.linspace(0, len(frames) - 1, num=limit, dtype=int)
    indices = np.unique(indices)
    return frames.iloc[indices].reset_index(drop=True)


def _branch_geometry(
    poly: vtk.vtkPolyData,
    *,
    branch_id_array: str,
    path_array: str,
) -> Dict[int, _BranchGeometry]:
    branch_ids = _coerce_scalar_array(_point_array(poly, branch_id_array), branch_id_array)
    path_values = _coerce_scalar_array(_point_array(poly, path_array), path_array)
    points = _polydata_points(poly)

    geometries: Dict[int, _BranchGeometry] = {}
    finite_mask = np.isfinite(branch_ids) & np.isfinite(path_values)
    for raw_branch in sorted(set(int(round(x)) for x in branch_ids[finite_mask])):
        point_indices = np.where(finite_mask & np.isclose(branch_ids, raw_branch))[0]
        warnings: List[str] = []
        if point_indices.size < 2:
            geometries[raw_branch] = _BranchGeometry(
                branch_id=raw_branch,
                point_indices=point_indices,
                length_cm=float("nan"),
                warnings=["branch has fewer than 2 valid points"],
            )
            continue

        ordered = point_indices[np.argsort(path_values[point_indices], kind="mergesort")]
        ordered_path = path_values[ordered]
        if np.any(np.diff(ordered_path) == 0.0):
            warnings.append("duplicate Path values within branch ordering")
        segment_points = points[ordered]
        length_cm = float(np.sum(np.linalg.norm(np.diff(segment_points, axis=0), axis=1)))
        geometries[raw_branch] = _BranchGeometry(
            branch_id=raw_branch,
            point_indices=ordered,
            length_cm=length_cm,
            warnings=warnings,
        )

    if not geometries:
        raise ValueError("no valid branches found in centerline data")
    return geometries


def _run_svslicer(
    *,
    svslicer_path: str,
    result_path: str,
    centerline_path: str,
    output_path: str,
) -> None:
    cmd = [svslicer_path, result_path, centerline_path, output_path]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to execute svSlicer at '{svslicer_path}': {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "no output"
        raise RuntimeError(f"svSlicer failed for '{result_path}': {detail}")


def _mapped_output_path(intermediate_dir: Path, frame_index: int, frame_path: str) -> Path:
    stem = Path(frame_path).stem
    return intermediate_dir / f"{frame_index:04d}_{stem}_centerline.vtp"


def _warning_message(branch_id: int, message: str) -> Dict[str, Any]:
    return {"branch_id": branch_id, "message": message}


def _resolve_workers(
    workers: int | Literal["auto"] | None,
) -> tuple[int | str | None, int]:
    if workers is None:
        return None, 1
    if workers == "auto":
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        available_cpus: int | None = None
        if slurm_cpus is not None:
            try:
                parsed = int(slurm_cpus)
            except ValueError:
                parsed = 0
            if parsed > 0:
                available_cpus = parsed
        if available_cpus is None:
            detected = os.cpu_count() or 1
            available_cpus = max(1, int(detected))
        return "auto", max(1, min(_AUTO_WORKER_CAP, available_cpus))
    resolved = int(workers)
    if resolved <= 0:
        raise ValueError("workers must be >= 1 when provided")
    return resolved, resolved


def _map_frame(
    *,
    frame_index: int,
    frame_path: str,
    time_s: float,
    intermediate_path: Path,
    svslicer_executable: Path,
    centerline_path: Path,
    total_frames: int,
) -> tuple[int, Path, float]:
    mapped_path = _mapped_output_path(intermediate_path, frame_index, frame_path)
    print(
        f"[svzerodtrees] resistance-map frame {frame_index + 1}/{total_frames} "
        f"source={frame_path}",
        flush=True,
    )
    _run_svslicer(
        svslicer_path=str(svslicer_executable),
        result_path=str(frame_path),
        centerline_path=str(centerline_path),
        output_path=str(mapped_path),
    )
    print(
        f"[svzerodtrees] resistance-map frame {frame_index + 1}/{total_frames} "
        f"mapped={mapped_path}",
        flush=True,
    )
    return frame_index, mapped_path, time_s


def _precomputed_mapped_path(row: pd.Series) -> Path | None:
    if "mapped_path" not in row.index:
        return None
    raw_value = row["mapped_path"]
    if raw_value is None or pd.isna(raw_value):
        return None
    mapped_path = Path(str(raw_value)).expanduser().resolve()
    if not mapped_path.exists():
        raise FileNotFoundError(f"precomputed mapped centerline not found: {mapped_path}")
    return mapped_path


def _metric_output_names(metric_suffix: str) -> tuple[dict[str, str], dict[str, str]]:
    metric = str(metric_suffix).strip()
    if not metric:
        raise ValueError("metric_suffix must be non-empty")
    columns = {
        "pressure_prox": f"pressure_prox_{metric}",
        "pressure_dist": f"pressure_dist_{metric}",
        "pressure_drop": f"pressure_drop_{metric}",
        "flow": f"flow_{metric}",
        "resistance": f"resistance_{metric}",
        "resistance_per_cm": f"resistance_per_cm_{metric}",
        "rank": "rank",
    }
    arrays = {
        "pressure_prox": f"branch_pressure_prox_{metric}",
        "pressure_dist": f"branch_pressure_dist_{metric}",
        "pressure_drop": f"branch_pressure_drop_{metric}",
        "flow": f"branch_flow_{metric}",
        "resistance": f"branch_resistance_{metric}",
        "resistance_per_cm": f"branch_resistance_per_cm_{metric}",
        "rank": "branch_rank",
    }
    return columns, arrays


def _artifact_paths(output_dir: str | Path, metric_suffix: str) -> dict[str, Path]:
    output_path = Path(output_dir)
    if metric_suffix == "mean":
        return {
            "summary_csv": output_path / "branch_resistance_summary.csv",
            "ranked_csv": output_path / "ranked_stent_candidates.csv",
            "resistance_map": output_path / "resistance_map_mean.vtp",
            "metadata_json": output_path / "resistance_map_metadata.json",
        }
    suffix = f"_{metric_suffix}"
    return {
        "summary_csv": output_path / f"branch_resistance_summary{suffix}.csv",
        "ranked_csv": output_path / f"ranked_stent_candidates{suffix}.csv",
        "resistance_map": output_path / f"resistance_map{suffix}.vtp",
        "metadata_json": output_path / f"resistance_map{suffix}_metadata.json",
    }


def _aggregate_branch_metrics(
    mapped_files: Iterable[tuple[Path, float]],
    *,
    branch_id_array: str,
    path_array: str,
    pressure_array: str,
    flow_array: str,
    metric_suffix: str = "mean",
) -> tuple[vtk.vtkPolyData, pd.DataFrame, List[Dict[str, Any]]]:
    mapped_files = list(mapped_files)
    if not mapped_files:
        raise ValueError("no mapped centerline files were produced")

    columns, arrays = _metric_output_names(metric_suffix)
    base_poly = _read_polydata(mapped_files[0][0])
    branch_geometry = _branch_geometry(
        base_poly,
        branch_id_array=branch_id_array,
        path_array=path_array,
    )
    warnings: List[Dict[str, Any]] = []

    per_branch: Dict[int, Dict[str, List[float]]] = {
        branch_id: {
            "pressure_prox": [],
            "pressure_dist": [],
            "pressure_drop": [],
            "flow": [],
        }
        for branch_id in branch_geometry
    }

    for branch_id, geometry in branch_geometry.items():
        for message in geometry.warnings:
            warnings.append(_warning_message(branch_id, message))

    for mapped_path, _time_s in mapped_files:
        poly = _read_polydata(mapped_path)
        pressure = _coerce_scalar_array(_point_array(poly, pressure_array), pressure_array)
        flow = _coerce_scalar_array(_point_array(poly, flow_array), flow_array)

        for branch_id, geometry in branch_geometry.items():
            indices = geometry.point_indices
            if indices.size < 2:
                continue

            pressure_values = pressure[indices]
            flow_values = flow[indices]
            finite_pressure = np.isfinite(pressure_values)
            finite_flow = np.isfinite(flow_values)

            if finite_pressure.sum() < 2:
                warnings.append(
                    _warning_message(branch_id, f"insufficient finite pressure samples in {mapped_path.name}")
                )
                continue

            prox_pressure = float(pressure_values[np.where(finite_pressure)[0][0]])
            dist_pressure = float(pressure_values[np.where(finite_pressure)[0][-1]])
            per_branch[branch_id]["pressure_prox"].append(prox_pressure)
            per_branch[branch_id]["pressure_dist"].append(dist_pressure)
            per_branch[branch_id]["pressure_drop"].append(prox_pressure - dist_pressure)

            if finite_flow.any():
                per_branch[branch_id]["flow"].append(float(np.mean(flow_values[finite_flow])))
            else:
                warnings.append(
                    _warning_message(branch_id, f"no finite flow samples in {mapped_path.name}")
                )

    rows: List[Dict[str, Any]] = []
    for branch_id, geometry in branch_geometry.items():
        branch_warning_messages = [entry["message"] for entry in warnings if entry["branch_id"] == branch_id]
        pressure_prox_mean = float(np.mean(per_branch[branch_id]["pressure_prox"])) if per_branch[branch_id]["pressure_prox"] else float("nan")
        pressure_dist_mean = float(np.mean(per_branch[branch_id]["pressure_dist"])) if per_branch[branch_id]["pressure_dist"] else float("nan")
        pressure_drop_mean = float(np.mean(per_branch[branch_id]["pressure_drop"])) if per_branch[branch_id]["pressure_drop"] else float("nan")
        flow_mean = float(np.mean(per_branch[branch_id]["flow"])) if per_branch[branch_id]["flow"] else float("nan")

        resistance_mean = float("nan")
        resistance_per_cm = float("nan")
        if np.isfinite(flow_mean):
            if abs(flow_mean) <= _FLOW_EPSILON:
                flow_warning = f"{metric_suffix} flow is zero or near zero"
                branch_warning_messages.append(flow_warning)
                warnings.append(_warning_message(branch_id, flow_warning))
            elif np.isfinite(pressure_drop_mean):
                resistance_mean = pressure_drop_mean / flow_mean
                if np.isfinite(geometry.length_cm) and geometry.length_cm > 0.0:
                    resistance_per_cm = resistance_mean / geometry.length_cm
                else:
                    branch_warning_messages.append("branch length is zero or invalid")
                    warnings.append(_warning_message(branch_id, "branch length is zero or invalid"))

        rows.append(
            {
                "branch_id": branch_id,
                "segment_length_cm": geometry.length_cm,
                columns["pressure_prox"]: pressure_prox_mean,
                columns["pressure_dist"]: pressure_dist_mean,
                columns["pressure_drop"]: pressure_drop_mean,
                columns["flow"]: flow_mean,
                columns["resistance"]: resistance_mean,
                columns["resistance_per_cm"]: resistance_per_cm,
                "warning": "; ".join(dict.fromkeys(branch_warning_messages)),
            }
        )

    summary = pd.DataFrame.from_records(rows).sort_values("branch_id").reset_index(drop=True)
    valid = summary[columns["resistance"]].replace([np.inf, -np.inf], np.nan).notna()
    ranked = summary.loc[valid].sort_values(
        [columns["resistance"], columns["resistance_per_cm"]],
        ascending=[False, False],
    )
    rank_map = {int(branch_id): idx + 1 for idx, branch_id in enumerate(ranked["branch_id"].tolist())}
    summary[columns["rank"]] = summary["branch_id"].map(rank_map)

    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(base_poly)
    for column, array_name in (
        (columns["pressure_prox"], arrays["pressure_prox"]),
        (columns["pressure_dist"], arrays["pressure_dist"]),
        (columns["pressure_drop"], arrays["pressure_drop"]),
        (columns["flow"], arrays["flow"]),
        (columns["resistance"], arrays["resistance"]),
        (columns["resistance_per_cm"], arrays["resistance_per_cm"]),
        (columns["rank"], arrays["rank"]),
    ):
        values = np.full(output_poly.GetNumberOfPoints(), np.nan, dtype=float)
        for row in summary.itertuples(index=False):
            indices = branch_geometry[int(row.branch_id)].point_indices
            values[indices] = getattr(row, column)
        vtk_array = numpy_to_vtk(values, deep=True)
        vtk_array.SetName(array_name)
        output_poly.GetPointData().AddArray(vtk_array)

    ranked_summary = summary.sort_values([columns["rank"], "branch_id"], na_position="last").reset_index(drop=True)
    return output_poly, ranked_summary, warnings


def _compute_pulmonary_resistance_map_for_selected_frames(
    *,
    svslicer_path: str,
    centerline: str,
    selected_frames: pd.DataFrame,
    output_dir: str,
    cycle_duration_s: float,
    available_frame_count: int,
    selection_window_start_s: float,
    selection_window_end_s: float,
    selection_tolerance_s: float,
    selection_policy: str,
    metric_suffix: str = "mean",
    keep_intermediate_centerlines: bool = False,
    intermediate_dir: str | None = None,
    workers: int | Literal["auto"] | None = None,
    pressure_array: str = "pressure",
    flow_array: str = "velocity",
    branch_id_array: str = "BranchId",
    path_array: str = "Path",
    metadata_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    centerline_path = Path(centerline).expanduser().resolve()
    svslicer_executable = Path(svslicer_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not centerline_path.exists():
        raise FileNotFoundError(f"centerline file not found: {centerline_path}")
    if not svslicer_executable.exists():
        raise FileNotFoundError(f"svSlicer executable not found: {svslicer_executable}")

    selected = selected_frames.reset_index(drop=True).copy()
    if selected.empty:
        raise ValueError("selected_frames must contain at least one frame")

    workers_requested, workers_used = _resolve_workers(workers)
    output_path.mkdir(parents=True, exist_ok=True)
    if intermediate_dir is None:
        intermediate_path = output_path / "intermediate_centerlines"
    else:
        intermediate_path = Path(intermediate_dir).expanduser().resolve()
    intermediate_path.mkdir(parents=True, exist_ok=True)

    artifact_paths = _artifact_paths(output_path, metric_suffix)
    metadata: Dict[str, Any] = {
        "svslicer_path": str(svslicer_executable),
        "centerline": str(centerline_path),
        "cycle_duration_s": float(cycle_duration_s),
        "selection_window_start_s": float(selection_window_start_s),
        "selection_window_end_s": float(selection_window_end_s),
        "selection_tolerance_s": float(selection_tolerance_s),
        "selection_policy": selection_policy,
        "available_frame_count": int(available_frame_count),
        "selected_frame_count": int(len(selected)),
        "max_frames": None,
        "keep_intermediate_centerlines": bool(keep_intermediate_centerlines),
        "intermediate_dir": str(intermediate_path),
        "pressure_array": pressure_array,
        "flow_array": flow_array,
        "branch_id_array": branch_id_array,
        "path_array": path_array,
        "workers_requested": workers_requested,
        "workers_used": int(workers_used),
        "metric_suffix": metric_suffix,
        "selected_frames": [],
        "warnings": [],
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    mapped_files_by_index: dict[int, tuple[Path, float]] = {}
    try:
        scheduled_rows: list[tuple[int, pd.Series]] = []
        for frame_index in range(len(selected)):
            row = selected.iloc[frame_index]
            mapped_path = _precomputed_mapped_path(row)
            if mapped_path is not None:
                mapped_files_by_index[frame_index] = (mapped_path, float(row["time_s"]))
            else:
                scheduled_rows.append((frame_index, row))

        if workers_used == 1:
            for frame_index, row in scheduled_rows:
                _, mapped_path, time_s = _map_frame(
                    frame_index=frame_index,
                    frame_path=str(row["path"]),
                    time_s=float(row["time_s"]),
                    intermediate_path=intermediate_path,
                    svslicer_executable=svslicer_executable,
                    centerline_path=centerline_path,
                    total_frames=len(selected),
                )
                mapped_files_by_index[frame_index] = (mapped_path, time_s)
        elif scheduled_rows:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers_used) as executor:
                future_map = {
                    executor.submit(
                        _map_frame,
                        frame_index=frame_index,
                        frame_path=str(row["path"]),
                        time_s=float(row["time_s"]),
                        intermediate_path=intermediate_path,
                        svslicer_executable=svslicer_executable,
                        centerline_path=centerline_path,
                        total_frames=len(selected),
                    ): frame_index
                    for frame_index, row in scheduled_rows
                }
                for future in concurrent.futures.as_completed(future_map):
                    frame_index, mapped_path, time_s = future.result()
                    mapped_files_by_index[frame_index] = (mapped_path, time_s)

        mapped_files = [mapped_files_by_index[index] for index in sorted(mapped_files_by_index)]
        selected_records: list[Dict[str, Any]] = []
        for frame_index, (mapped_path, time_s) in enumerate(mapped_files):
            row = selected.iloc[frame_index]
            record: Dict[str, Any] = {
                "path": str(mapped_path),
                "time_s": float(time_s),
                "source_frame_path": str(row["source_frame_path"]) if "source_frame_path" in row.index else str(row["path"]),
            }
            if "timestep_id" in selected.columns:
                record["timestep_id"] = int(row["timestep_id"])
            selected_records.append(record)
        metadata["selected_frames"] = selected_records

        resistance_map_poly, summary, warnings = _aggregate_branch_metrics(
            mapped_files,
            branch_id_array=branch_id_array,
            path_array=path_array,
            pressure_array=pressure_array,
            flow_array=flow_array,
            metric_suffix=metric_suffix,
        )

        summary.to_csv(artifact_paths["summary_csv"], index=False)
        summary.sort_values(["rank", "branch_id"], na_position="last").to_csv(
            artifact_paths["ranked_csv"],
            index=False,
        )
        _write_polydata(resistance_map_poly, artifact_paths["resistance_map"])

        metadata["warnings"] = warnings
        artifact_paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "kind": "pulmonary_resistance_map",
            "metric_suffix": metric_suffix,
            "output_dir": str(output_path),
            "resistance_map": str(artifact_paths["resistance_map"]),
            "summary_csv": str(artifact_paths["summary_csv"]),
            "ranked_csv": str(artifact_paths["ranked_csv"]),
            "metadata_json": str(artifact_paths["metadata_json"]),
            "selected_frame_count": len(mapped_files),
            "available_frame_count": int(available_frame_count),
            "intermediate_dir": str(intermediate_path) if keep_intermediate_centerlines else None,
        }
    finally:
        if not keep_intermediate_centerlines and intermediate_path.exists():
            shutil.rmtree(intermediate_path)


def compute_pulmonary_resistance_map(
    *,
    svslicer_path: str,
    centerline: str,
    frames_csv: str,
    output_dir: str,
    cycle_duration_s: float,
    max_frames: int | None = None,
    keep_intermediate_centerlines: bool = False,
    intermediate_dir: str | None = None,
    workers: int | Literal["auto"] | None = None,
    pressure_array: str = "pressure",
    flow_array: str = "velocity",
    branch_id_array: str = "BranchId",
    path_array: str = "Path",
) -> Dict[str, Any]:
    frames = _load_frames_csv(frames_csv)
    selected_all, window_start_s, window_end_s, selection_tolerance_s = _select_last_cycle_frames(
        frames,
        cycle_duration_s,
        return_metadata=True,
        include_endpoint=False,
    )
    selected = _subsample_frames(selected_all, max_frames)
    return _compute_pulmonary_resistance_map_for_selected_frames(
        svslicer_path=svslicer_path,
        centerline=centerline,
        selected_frames=selected,
        output_dir=output_dir,
        cycle_duration_s=cycle_duration_s,
        available_frame_count=len(selected_all),
        selection_window_start_s=window_start_s,
        selection_window_end_s=window_end_s,
        selection_tolerance_s=selection_tolerance_s,
        selection_policy="all_frames_last_full_cycle",
        metric_suffix="mean",
        keep_intermediate_centerlines=keep_intermediate_centerlines,
        intermediate_dir=intermediate_dir,
        workers=workers,
        pressure_array=pressure_array,
        flow_array=flow_array,
        branch_id_array=branch_id_array,
        path_array=path_array,
        metadata_extra={
            "frames_csv": str(Path(frames_csv).expanduser().resolve()),
            "max_frames": None if max_frames is None else int(max_frames),
        },
    )
