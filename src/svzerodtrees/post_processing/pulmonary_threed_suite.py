from __future__ import annotations

import heapq
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from ..simulation.simulation_directory import SimulationDirectory
from ..tune_bcs.clinical_targets import ClinicalTargets
from .resistance_map import compute_pulmonary_resistance_map

CGS_TO_MMHG = 1333.224


def _extract_result_step(path: str | Path) -> int:
    match = re.search(r"result_(\d+)\.vtu$", Path(path).name)
    if match is None:
        raise ValueError(f"could not parse timestep ID from {path}")
    return int(match.group(1))


def _result_vtu_files(simulation_dir: str | Path) -> list[Path]:
    sim_dir = Path(simulation_dir)
    files = sorted(sim_dir.glob("*-procs/result_*.vtu"), key=_extract_result_step)
    if not files:
        files = sorted(sim_dir.glob("result_*.vtu"), key=_extract_result_step)
    if not files:
        raise FileNotFoundError(f"no result_*.vtu files found in {sim_dir}")
    return files


def _simulation_timestep_seconds(simulation_dir: str | Path) -> float:
    xml_path = Path(simulation_dir) / "svFSIplus.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"svFSIplus.xml not found in {simulation_dir}")
    root = ET.parse(xml_path).getroot()
    dt_text = root.findtext(".//Time_step_size")
    if dt_text is None:
        raise ValueError(f"Time_step_size missing in {xml_path}")
    dt = float(dt_text)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"invalid Time_step_size in {xml_path}: {dt}")
    return dt


def _compute_arc_length(polydata: pv.PolyData) -> np.ndarray:
    pts = polydata.points
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    polydata.point_data["ArcLength"] = arc
    return arc


def _build_graph(centerline: pv.PolyData) -> dict[int, dict[int, float]]:
    lines = centerline.lines
    graph: dict[int, dict[int, float]] = {}
    pts = centerline.points

    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        for a, b in zip(ids[:-1], ids[1:]):
            a = int(a)
            b = int(b)
            weight = float(np.linalg.norm(pts[a] - pts[b]))
            graph.setdefault(a, {})[b] = weight
            graph.setdefault(b, {})[a] = weight
        i += n + 1
    return graph


def _dijkstra(graph: dict[int, dict[int, float]], start: int) -> tuple[dict[int, float], dict[int, int]]:
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, start)]

    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > dist.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, {}).items():
            new_dist = current_dist + weight
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(queue, (new_dist, neighbor))
    return dist, prev


def _find_first_bifurcation(graph: dict[int, dict[int, float]], root_id: int) -> int:
    dist, _ = _dijkstra(graph, root_id)
    candidates = [
        node_id
        for node_id, neighbors in graph.items()
        if len(neighbors) >= 3 and node_id != root_id and node_id in dist
    ]
    if not candidates:
        raise ValueError("could not detect a bifurcation node with degree >= 3")
    candidates.sort(key=lambda node_id: dist[node_id])
    return candidates[0]


def _reconstruct_path(prev: dict[int, int], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
    path = [end]
    current = end
    while current != start:
        if current not in prev:
            raise ValueError("no path found from root to bifurcation")
        current = prev[current]
        path.append(current)
    path.reverse()
    return path


def _sample_pressure_on_centerline(
    centerline: pv.PolyData,
    mesh: pv.DataSet,
    pressure_field: str,
    already_mmhg: bool,
) -> np.ndarray:
    sampled = centerline.sample(mesh)
    if pressure_field not in sampled.point_data:
        raise KeyError(
            f"pressure field '{pressure_field}' not found after sampling; "
            f"available={list(sampled.point_data.keys())}"
        )
    pressure = sampled.point_data[pressure_field].astype(np.float64)
    if pressure.ndim > 1:
        pressure = pressure[:, 0]
    if not already_mmhg:
        pressure = pressure / CGS_TO_MMHG
    return pressure


def _weighted_mean_on_path(points: np.ndarray, values: np.ndarray, path_ids: list[int]) -> float:
    path_points = points[path_ids]
    path_values = values[path_ids]
    if len(path_ids) == 1:
        return float(path_values[0])
    seg_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    seg_mean = 0.5 * (path_values[:-1] + path_values[1:])
    total_length = float(np.sum(seg_lengths))
    if total_length <= 0.0:
        return float(np.mean(path_values))
    return float(np.sum(seg_mean * seg_lengths) / total_length)


def _clinical_targets_from_input(
    clinical_targets: str | Path | Mapping[str, Any] | None,
) -> dict[str, float] | None:
    if clinical_targets is None:
        return None
    if isinstance(clinical_targets, Mapping):
        normalized_keys = {"mpa_sys", "mpa_dia", "mpa_mean", "rpa_split"}
        if normalized_keys.issubset(clinical_targets):
            return {
                "mpa_sys": float(clinical_targets["mpa_sys"]),
                "mpa_dia": float(clinical_targets["mpa_dia"]),
                "mpa_mean": float(clinical_targets["mpa_mean"]),
                "rpa_split": float(clinical_targets["rpa_split"]),
            }
        if "mpa_p" in clinical_targets:
            pressures = clinical_targets["mpa_p"]
        else:
            pressures = clinical_targets.get("mpa_pressure")
        try:
            pressure_values = np.asarray(pressures, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "clinical_targets mapping requires mpa_p or mpa_pressure length-3 sequence"
            ) from exc
        if pressure_values.size < 3:
            raise ValueError("clinical_targets mapping requires mpa_p or mpa_pressure length-3 sequence")
        if "rpa_split" not in clinical_targets:
            raise ValueError("clinical_targets mapping requires rpa_split")
        return {
            "mpa_sys": float(pressure_values[0]),
            "mpa_dia": float(pressure_values[1]),
            "mpa_mean": float(pressure_values[2]),
            "rpa_split": float(clinical_targets["rpa_split"]),
        }
    targets = ClinicalTargets.from_csv(str(clinical_targets))
    return {
        "mpa_sys": float(targets.mpa_p[0]),
        "mpa_dia": float(targets.mpa_p[1]),
        "mpa_mean": float(targets.mpa_p[2]),
        "rpa_split": float(targets.rpa_split),
    }


def _cycle_duration_from_inflow_csv(inflow_csv: str | Path) -> float:
    df = pd.read_csv(inflow_csv)
    time_col = None
    for candidate in ("t", "time", "time_s"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(f"inflow CSV is missing time column: {inflow_csv}")
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    times = times[np.isfinite(times)]
    if times.size < 2:
        raise ValueError(f"inflow CSV must contain at least two finite time values: {inflow_csv}")
    duration = float(np.max(times) - np.min(times))
    if duration <= 0.0:
        raise ValueError(f"inflow CSV duration must be > 0: {inflow_csv}")
    return duration


def _resolve_suite_targets(
    *,
    stage: str,
    clinical_targets: str | Path | Mapping[str, Any] | None,
    warnings: list[str],
) -> dict[str, float] | None:
    if clinical_targets is None:
        warnings.append(
            f"{stage} clinical targets unavailable; generating non-comparison artifacts only"
        )
        return None
    try:
        return _clinical_targets_from_input(clinical_targets)
    except (TypeError, ValueError) as exc:
        warnings.append(
            f"{stage} clinical targets invalid for overlay/comparison; "
            f"generating non-comparison artifacts only ({exc})"
        )
        return None


def write_mpa_pressure_timeseries_csv(
    *,
    simulation_dir: str | Path,
    centerline: str | Path,
    output_csv: str | Path,
    output_plot: str | Path | None = None,
    clinical_targets: str | Path | Mapping[str, Any] | None = None,
    pressure_field: str = "Pressure",
    already_mmhg: bool = False,
    root_id: int = 0,
    bifurcation_id: int | None = None,
) -> dict[str, Any]:
    centerline_poly = pv.read(str(centerline))
    _compute_arc_length(centerline_poly)

    graph = _build_graph(centerline_poly)
    if root_id not in graph:
        raise ValueError(f"root_id {root_id} is not a valid connected centerline point")

    resolved_bifurcation_id = (
        int(bifurcation_id)
        if bifurcation_id is not None
        else _find_first_bifurcation(graph, root_id)
    )
    dist, prev = _dijkstra(graph, root_id)
    if resolved_bifurcation_id not in dist:
        raise ValueError(
            f"bifurcation_id {resolved_bifurcation_id} is not reachable from root_id {root_id}"
        )
    path_ids = _reconstruct_path(prev, root_id, resolved_bifurcation_id)

    dt = _simulation_timestep_seconds(simulation_dir)
    vtu_files = _result_vtu_files(simulation_dir)

    timestep_ids: list[int] = []
    mpa_pressure_values: list[float] = []
    for vtu_path in vtu_files:
        mesh = pv.read(str(vtu_path))
        pressure = _sample_pressure_on_centerline(
            centerline_poly,
            mesh,
            pressure_field,
            already_mmhg,
        )
        timestep_ids.append(_extract_result_step(vtu_path))
        mpa_pressure_values.append(
            _weighted_mean_on_path(centerline_poly.points, pressure, path_ids)
        )

    time_s = np.array(timestep_ids, dtype=float) * dt
    pressure_values = np.array(mpa_pressure_values, dtype=float)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestep_id": np.array(timestep_ids, dtype=int),
            "time_s": time_s,
            "mpa_pressure_mmhg": pressure_values,
        }
    ).to_csv(output_csv_path, index=False)

    targets = _clinical_targets_from_input(clinical_targets)
    if output_plot is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(time_s, pressure_values, linewidth=1.8, label="Simulated MPA pressure")
        if targets is not None:
            ax.axhline(targets["mpa_sys"], color="tab:red", linestyle="--", linewidth=1.2, label="Clinical systolic")
            ax.axhline(targets["mpa_dia"], color="tab:purple", linestyle="--", linewidth=1.2, label="Clinical diastolic")
            ax.axhline(targets["mpa_mean"], color="tab:green", linestyle="--", linewidth=1.2, label="Clinical mean")
        ax.axhline(float(np.mean(pressure_values)), color="tab:blue", linewidth=1.1, label="Simulated mean")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average MPA pressure (mmHg)")
        ax.set_title("MPA Pressure vs Time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        output_plot_path = Path(output_plot)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300)
        plt.close(fig)

    return {
        "csv_path": str(output_csv_path),
        "plot_path": str(output_plot) if output_plot is not None else None,
        "root_id": root_id,
        "bifurcation_id": resolved_bifurcation_id,
        "segment_length": float(dist[resolved_bifurcation_id]),
        "point_count": len(path_ids),
        "file_count": len(vtu_files),
        "pressure_field": pressure_field,
    }


def write_frames_csv_for_simulation(
    *,
    simulation_dir: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    dt = _simulation_timestep_seconds(simulation_dir)
    vtu_files = _result_vtu_files(simulation_dir)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "path": [str(path.resolve()) for path in vtu_files],
            "time_s": [float(_extract_result_step(path)) * dt for path in vtu_files],
        }
    ).to_csv(output_csv_path, index=False)
    return {
        "csv_path": str(output_csv_path),
        "file_count": len(vtu_files),
        "dt": dt,
    }


def write_flow_split_comparison_artifacts(
    *,
    simulation_dir: str | Path,
    output_csv: str | Path,
    output_plot: str | Path,
    clinical_targets: str | Path | Mapping[str, Any] | None = None,
    stage: str = "preop",
) -> dict[str, Any]:
    sim = SimulationDirectory.from_directory(str(simulation_dir))
    lpa_flow, rpa_flow = sim.flow_split(get_mean=True, verbose=False)
    lpa_total = float(sum(float(value) for value in lpa_flow.values()))
    rpa_total = float(sum(float(value) for value in rpa_flow.values()))
    total_flow = lpa_total + rpa_total
    if total_flow <= 0.0:
        raise ValueError("total flow must be positive to compute flow split artifacts")

    simulated_splits = {
        "lpa": lpa_total / total_flow,
        "rpa": rpa_total / total_flow,
    }
    targets = _clinical_targets_from_input(clinical_targets)
    target_rpa_split = targets["rpa_split"] if targets is not None else None
    target_lpa_split = (1.0 - target_rpa_split) if target_rpa_split is not None else None

    rows = [
        {
            "stage": stage,
            "vessel": "lpa",
            "simulated_flow": lpa_total,
            "simulated_split": simulated_splits["lpa"],
            "clinical_split": target_lpa_split,
            "delta_split": (
                simulated_splits["lpa"] - target_lpa_split
                if target_lpa_split is not None
                else np.nan
            ),
        },
        {
            "stage": stage,
            "vessel": "rpa",
            "simulated_flow": rpa_total,
            "simulated_split": simulated_splits["rpa"],
            "clinical_split": target_rpa_split,
            "delta_split": (
                simulated_splits["rpa"] - target_rpa_split
                if target_rpa_split is not None
                else np.nan
            ),
        },
    ]
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    width = 0.35
    sim_values = [simulated_splits["lpa"], simulated_splits["rpa"]]
    ax.bar(x - width / 2, sim_values, width=width, label="Simulated")
    if target_lpa_split is not None and target_rpa_split is not None:
        ax.bar(
            x + width / 2,
            [target_lpa_split, target_rpa_split],
            width=width,
            label="Clinical target",
        )
    ax.set_xticks(x, ["LPA", "RPA"])
    ax.set_ylabel("Flow split")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{stage.capitalize()} Flow Split Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_plot_path = Path(output_plot)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=300)
    plt.close(fig)

    return {
        "csv_path": str(output_csv_path),
        "plot_path": str(output_plot_path),
        "lpa_flow": lpa_total,
        "rpa_flow": rpa_total,
        "lpa_split": simulated_splits["lpa"],
        "rpa_split": simulated_splits["rpa"],
        "clinical_targets_available": targets is not None,
    }


def render_resistance_map_png(
    *,
    resistance_map_vtp: str | Path,
    output_png: str | Path,
    scalar_name: str = "branch_resistance_mean",
) -> str:
    poly = pv.read(str(resistance_map_vtp))
    if scalar_name not in poly.point_data:
        raise KeyError(
            f"scalar '{scalar_name}' not found in resistance map; "
            f"available={list(poly.point_data.keys())}"
        )

    bounds = poly.bounds
    span = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )
    tube_radius = max(span * 0.01, 1e-3)

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    mesh = poly.tube(radius=tube_radius)
    plotter.add_mesh(mesh, scalars=scalar_name, cmap="viridis", show_scalar_bar=True)
    plotter.view_isometric()
    plotter.set_background("white")
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    return str(output_path)


def run_pulmonary_threed_postprocess_suite(
    *,
    simulation_dir: str | Path,
    output_dir: str | Path,
    centerline: str | Path,
    stage: str,
    svslicer_path: str,
    clinical_targets: str | Path | Mapping[str, Any] | None = None,
    cycle_duration_s: float | None = None,
    inflow_csv: str | Path | None = None,
    pressure_field: str = "Pressure",
    already_mmhg: bool = False,
    resistance_map_workers: int | Literal["auto"] | None = None,
) -> dict[str, Any]:
    if cycle_duration_s is None:
        if inflow_csv is None:
            raise ValueError("one of cycle_duration_s or inflow_csv is required")
        cycle_duration_s = _cycle_duration_from_inflow_csv(inflow_csv)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "postprocess_suite_metadata.json"
    warnings: list[str] = []
    targets = _resolve_suite_targets(
        stage=stage,
        clinical_targets=clinical_targets,
        warnings=warnings,
    )
    target_input = clinical_targets if targets is not None else None

    pressure_csv = output_path / "mpa_pressure_vs_time.csv"
    pressure_png = output_path / "mpa_pressure_vs_time.png"
    flow_split_csv = output_path / "flow_split_comparison.csv"
    flow_split_png = output_path / "flow_split_comparison.png"
    frames_csv = output_path / "frames.csv"
    resistance_dir = output_path / "resistance_map"
    resistance_map_vtp = output_path / "resistance_map_mean.vtp"
    branch_summary_csv = output_path / "branch_resistance_summary.csv"
    ranked_candidates_csv = output_path / "ranked_stent_candidates.csv"
    resistance_metadata_json = output_path / "resistance_map_metadata.json"
    resistance_png = output_path / "resistance_map_mean.png"
    outputs = {
        "mpa_pressure_csv": str(pressure_csv),
        "mpa_pressure_png": str(pressure_png),
        "flow_split_csv": str(flow_split_csv),
        "flow_split_png": str(flow_split_png),
        "frames_csv": str(frames_csv),
        "resistance_map_vtp": str(resistance_map_vtp),
        "resistance_map_png": str(resistance_png),
        "branch_resistance_summary_csv": str(branch_summary_csv),
        "ranked_stent_candidates_csv": str(ranked_candidates_csv),
        "resistance_map_metadata_json": str(resistance_metadata_json),
    }
    steps: dict[str, dict[str, Any]] = {
        "pressure": {"status": "pending"},
        "flow_split": {"status": "pending"},
        "frames": {"status": "pending"},
        "resistance_map": {"status": "pending"},
    }
    metadata: dict[str, Any] = {
        "kind": "pulmonary_threed_suite",
        "status": "running",
        "stage": stage,
        "simulation_dir": str(Path(simulation_dir).resolve()),
        "output_dir": str(output_path.resolve()),
        "centerline": str(Path(centerline).resolve()),
        "svslicer_path": str(Path(svslicer_path).expanduser()),
        "cycle_duration_s": float(cycle_duration_s),
        "clinical_targets_available": targets is not None,
        "warnings": warnings,
        "outputs": outputs,
        "steps": steps,
        "metadata_json": str(metadata_path),
    }
    active_step: str | None = None

    try:
        active_step = "pressure"
        pressure_result = write_mpa_pressure_timeseries_csv(
            simulation_dir=simulation_dir,
            centerline=centerline,
            output_csv=pressure_csv,
            output_plot=pressure_png,
            clinical_targets=target_input,
            pressure_field=pressure_field,
            already_mmhg=already_mmhg,
        )
        steps["pressure"] = {"status": "completed", "result": pressure_result}
        metadata["pressure"] = pressure_result

        active_step = "flow_split"
        flow_split_result = write_flow_split_comparison_artifacts(
            simulation_dir=simulation_dir,
            output_csv=flow_split_csv,
            output_plot=flow_split_png,
            clinical_targets=target_input,
            stage=stage,
        )
        steps["flow_split"] = {"status": "completed", "result": flow_split_result}
        metadata["flow_split"] = flow_split_result

        active_step = "frames"
        frames_result = write_frames_csv_for_simulation(
            simulation_dir=simulation_dir,
            output_csv=frames_csv,
        )
        steps["frames"] = {"status": "completed", "result": frames_result}
        metadata["frames"] = frames_result

        active_step = "resistance_map"
        resistance_result = compute_pulmonary_resistance_map(
            svslicer_path=svslicer_path,
            centerline=str(centerline),
            frames_csv=str(frames_csv),
            output_dir=str(resistance_dir),
            cycle_duration_s=float(cycle_duration_s),
            workers=resistance_map_workers,
        )
        shutil.copyfile(resistance_result["resistance_map"], resistance_map_vtp)
        shutil.copyfile(resistance_result["summary_csv"], branch_summary_csv)
        shutil.copyfile(resistance_result["ranked_csv"], ranked_candidates_csv)
        shutil.copyfile(resistance_result["metadata_json"], resistance_metadata_json)
        render_resistance_map_png(
            resistance_map_vtp=resistance_map_vtp,
            output_png=resistance_png,
        )
        steps["resistance_map"] = {"status": "completed", "result": resistance_result}
        metadata["resistance_map"] = resistance_result

        metadata["status"] = "completed"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return metadata
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = {
            "step": active_step,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if active_step is not None:
            steps[active_step] = {
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
