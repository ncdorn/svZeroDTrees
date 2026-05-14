from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

_FLOW_EPSILON = 1e-12


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


def _load_frames_csv(frames_csv: str | Path) -> pd.DataFrame:
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

    records = []
    for row in df.itertuples(index=False):
        raw_path = Path(str(row.path)).expanduser()
        if not raw_path.is_absolute():
            raw_path = (path.parent / raw_path).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"3D result frame not found: {raw_path}")
        time_s = float(row.time_s)
        if not math.isfinite(time_s):
            raise ValueError(f"frames_csv contains non-finite time_s for {raw_path}")
        records.append({"path": str(raw_path), "time_s": time_s})

    return pd.DataFrame.from_records(records).sort_values("time_s").reset_index(drop=True)


def _select_last_cycle_frames(frames: pd.DataFrame, cycle_duration_s: float) -> pd.DataFrame:
    cycle_duration = float(cycle_duration_s)
    if not math.isfinite(cycle_duration) or cycle_duration <= 0.0:
        raise ValueError("cycle_duration_s must be positive and finite")
    cutoff = float(frames["time_s"].max()) - cycle_duration
    selected = frames.loc[frames["time_s"] > cutoff].copy()
    if selected.empty:
        raise ValueError("last-cycle frame selection is empty")
    return selected.reset_index(drop=True)


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


def _aggregate_branch_metrics(
    mapped_files: Iterable[tuple[Path, float]],
    *,
    branch_id_array: str,
    path_array: str,
    pressure_array: str,
    flow_array: str,
) -> tuple[vtk.vtkPolyData, pd.DataFrame, List[Dict[str, Any]]]:
    mapped_files = list(mapped_files)
    if not mapped_files:
        raise ValueError("no mapped centerline files were produced")

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
                branch_warning_messages.append("mean flow is zero or near zero")
                warnings.append(_warning_message(branch_id, "mean flow is zero or near zero"))
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
                "pressure_prox_mean": pressure_prox_mean,
                "pressure_dist_mean": pressure_dist_mean,
                "pressure_drop_mean": pressure_drop_mean,
                "flow_mean": flow_mean,
                "resistance_mean": resistance_mean,
                "resistance_per_cm": resistance_per_cm,
                "warning": "; ".join(dict.fromkeys(branch_warning_messages)),
            }
        )

    summary = pd.DataFrame.from_records(rows).sort_values("branch_id").reset_index(drop=True)
    valid = summary["resistance_mean"].replace([np.inf, -np.inf], np.nan).notna()
    ranked = summary.loc[valid].sort_values(
        ["resistance_mean", "resistance_per_cm"],
        ascending=[False, False],
    )
    rank_map = {int(branch_id): idx + 1 for idx, branch_id in enumerate(ranked["branch_id"].tolist())}
    summary["rank"] = summary["branch_id"].map(rank_map)

    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(base_poly)
    for column, array_name in (
        ("pressure_prox_mean", "branch_pressure_prox_mean"),
        ("pressure_dist_mean", "branch_pressure_dist_mean"),
        ("pressure_drop_mean", "branch_pressure_drop_mean"),
        ("flow_mean", "branch_flow_mean"),
        ("resistance_mean", "branch_resistance_mean"),
        ("resistance_per_cm", "branch_resistance_per_cm"),
        ("rank", "branch_rank"),
    ):
        values = np.full(output_poly.GetNumberOfPoints(), np.nan, dtype=float)
        for row in summary.itertuples(index=False):
            indices = branch_geometry[int(row.branch_id)].point_indices
            values[indices] = getattr(row, column)
        vtk_array = numpy_to_vtk(values, deep=True)
        vtk_array.SetName(array_name)
        output_poly.GetPointData().AddArray(vtk_array)

    ranked_summary = summary.sort_values(["rank", "branch_id"], na_position="last").reset_index(drop=True)
    return output_poly, ranked_summary, warnings


def compute_pulmonary_resistance_map(
    *,
    svslicer_path: str,
    centerline: str,
    frames_csv: str,
    output_dir: str,
    cycle_duration_s: float,
    max_frames: int | None = 8,
    keep_intermediate_centerlines: bool = False,
    intermediate_dir: str | None = None,
    pressure_array: str = "pressure",
    flow_array: str = "velocity",
    branch_id_array: str = "BranchId",
    path_array: str = "Path",
) -> Dict[str, Any]:
    centerline_path = Path(centerline).expanduser().resolve()
    svslicer_executable = Path(svslicer_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not centerline_path.exists():
        raise FileNotFoundError(f"centerline file not found: {centerline_path}")
    if not svslicer_executable.exists():
        raise FileNotFoundError(f"svSlicer executable not found: {svslicer_executable}")

    frames = _load_frames_csv(frames_csv)
    selected_all = _select_last_cycle_frames(frames, cycle_duration_s)
    selected = _subsample_frames(selected_all, max_frames)

    output_path.mkdir(parents=True, exist_ok=True)
    if intermediate_dir is None:
        intermediate_path = output_path / "intermediate_centerlines"
    else:
        intermediate_path = Path(intermediate_dir).expanduser().resolve()
    intermediate_path.mkdir(parents=True, exist_ok=True)

    mapped_files: List[tuple[Path, float]] = []
    try:
        for frame_index, row in enumerate(selected.itertuples(index=False)):
            mapped_path = _mapped_output_path(intermediate_path, frame_index, row.path)
            print(
                f"[svzerodtrees] resistance-map frame {frame_index + 1}/{len(selected)} "
                f"source={row.path}",
                flush=True,
            )
            _run_svslicer(
                svslicer_path=str(svslicer_executable),
                result_path=str(row.path),
                centerline_path=str(centerline_path),
                output_path=str(mapped_path),
            )
            print(
                f"[svzerodtrees] resistance-map frame {frame_index + 1}/{len(selected)} "
                f"mapped={mapped_path}",
                flush=True,
            )
            mapped_files.append((mapped_path, float(row.time_s)))

        resistance_map_poly, summary, warnings = _aggregate_branch_metrics(
            mapped_files,
            branch_id_array=branch_id_array,
            path_array=path_array,
            pressure_array=pressure_array,
            flow_array=flow_array,
        )

        summary_path = output_path / "branch_resistance_summary.csv"
        ranked_path = output_path / "ranked_stent_candidates.csv"
        vtp_path = output_path / "resistance_map_mean.vtp"
        metadata_path = output_path / "resistance_map_metadata.json"

        summary.to_csv(summary_path, index=False)
        summary.sort_values(["rank", "branch_id"], na_position="last").to_csv(ranked_path, index=False)
        _write_polydata(resistance_map_poly, vtp_path)

        metadata = {
            "svslicer_path": str(svslicer_executable),
            "centerline": str(centerline_path),
            "frames_csv": str(Path(frames_csv).expanduser().resolve()),
            "cycle_duration_s": float(cycle_duration_s),
            "available_frame_count": int(len(selected_all)),
            "selected_frame_count": int(len(selected)),
            "max_frames": None if max_frames is None else int(max_frames),
            "keep_intermediate_centerlines": bool(keep_intermediate_centerlines),
            "intermediate_dir": str(intermediate_path),
            "pressure_array": pressure_array,
            "flow_array": flow_array,
            "branch_id_array": branch_id_array,
            "path_array": path_array,
            "selected_frames": [
                {"path": str(path), "time_s": time_s} for path, time_s in mapped_files
            ],
            "warnings": warnings,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "kind": "pulmonary_resistance_map",
            "output_dir": str(output_path),
            "resistance_map": str(vtp_path),
            "summary_csv": str(summary_path),
            "ranked_csv": str(ranked_path),
            "metadata_json": str(metadata_path),
            "selected_frame_count": len(mapped_files),
            "available_frame_count": len(selected_all),
            "intermediate_dir": str(intermediate_path) if keep_intermediate_centerlines else None,
        }
    finally:
        if not keep_intermediate_centerlines and intermediate_path.exists():
            shutil.rmtree(intermediate_path)
