"""
Waveform visualizations by generation for structured tree simulations.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ...microvasculature.structured_tree.results import StructuredTreeResults
from ...microvasculature.structured_tree.storage import StructuredTreeStorage

if TYPE_CHECKING:
    from ...microvasculature.structured_tree.structuredtree import StructuredTree

MMHG_PER_BARYE = 1.0 / 1333.22


@dataclass
class WaveformStats:
    mean: np.ndarray  # shape (G, T)
    median: np.ndarray  # shape (G, T)
    q25: np.ndarray  # shape (G, T)
    q75: np.ndarray  # shape (G, T)


@dataclass
class GenerationWaveformData:
    generations: np.ndarray
    counts: np.ndarray
    time: np.ndarray
    metrics: Dict[str, WaveformStats]
    normalized_time: bool


def _resolve_collapsed_mask(results: StructuredTreeResults, store: Optional[StructuredTreeStorage]) -> np.ndarray:
    mask = np.zeros(results.n_vessels, dtype=bool)
    if store is None:
        return mask
    ids_store = np.asarray(store.ids, dtype=np.int32)
    collapsed_store = np.asarray(store.collapsed, dtype=bool)
    lookup = {int(vid): int(idx) for idx, vid in enumerate(ids_store)}
    for row, vid in enumerate(np.asarray(results.vessel_ids, dtype=np.int32)):
        idx = lookup.get(int(vid))
        if idx is not None:
            mask[row] = bool(collapsed_store[idx])
    return mask


def _select_time_indices(times: np.ndarray, time_window: Optional[Tuple[float, float]]) -> np.ndarray:
    if time_window is None:
        return np.ones(times.shape[0], dtype=bool)
    t0, t1 = time_window
    if t1 < t0:
        t0, t1 = t1, t0
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        raise ValueError(f"Time window {time_window} selects no samples.")
    return mask


def _per_generation_stats(
    generations: np.ndarray,
    values: np.ndarray,
    gens_unique: np.ndarray,
) -> WaveformStats:
    G = gens_unique.shape[0]
    T = values.shape[1]
    mean = np.full((G, T), np.nan, dtype=float)
    median = np.full((G, T), np.nan, dtype=float)
    q25 = np.full((G, T), np.nan, dtype=float)
    q75 = np.full((G, T), np.nan, dtype=float)

    for idx, g in enumerate(gens_unique):
        rows = generations == g
        if not np.any(rows):
            continue
        subset = values[rows]
        mean[idx] = np.nanmean(subset, axis=0)
        median[idx] = np.nanmedian(subset, axis=0)
        q25[idx] = np.nanpercentile(subset, 25.0, axis=0)
        q75[idx] = np.nanpercentile(subset, 75.0, axis=0)

    return WaveformStats(mean=mean, median=median, q25=q25, q75=q75)


def _resolve_wall_thickness(
    diameters: np.ndarray,
    wall_thickness: Optional[np.ndarray],
    *,
    wall_thickness_ratio: float,
    min_wall_thickness: float,
) -> np.ndarray:
    r = 0.5 * np.asarray(diameters, dtype=float)
    if wall_thickness is None:
        thickness = r * float(wall_thickness_ratio)
    else:
        thickness = np.asarray(wall_thickness, dtype=float)
        if thickness.ndim == 0:
            thickness = np.full_like(r, float(thickness))
    thickness = np.maximum(thickness, float(min_wall_thickness))
    return thickness


def _intramural_stress_timeseries(
    pressure_ts: np.ndarray,
    diameters: np.ndarray,
    *,
    wall_thickness: Optional[np.ndarray],
    wall_thickness_ratio: float,
    min_wall_thickness: float,
) -> np.ndarray:
    r = 0.5 * np.asarray(diameters, dtype=float)[:, None]
    thickness = _resolve_wall_thickness(
        diameters,
        wall_thickness,
        wall_thickness_ratio=wall_thickness_ratio,
        min_wall_thickness=min_wall_thickness,
    )
    if thickness.ndim == 1:
        thickness = thickness[:, None]
    if thickness.shape[0] != r.shape[0]:
        raise ValueError("wall_thickness must be scalar or have length equal to n_vessels.")
    if thickness.shape[1] not in (1, pressure_ts.shape[1]):
        raise ValueError("wall_thickness must be shape (N,), (N,1), or (N,T).")
    return np.asarray(pressure_ts, dtype=float) * r / thickness


def compute_generation_waveforms(
    results: StructuredTreeResults,
    store: Optional[StructuredTreeStorage] = None,
    *,
    time_window: Optional[Tuple[float, float]] = None,
    flow_field: str = "flow_in",
    pressure_field: str = "pressure_in",
    wss_flow: str = "in",
    include_collapsed: bool = True,
    take_abs_flow_for_wss: bool = False,
    convert_pressure_to_mmhg: bool = True,
    normalize_time: bool = False,
    include_intramural: bool = False,
    wall_thickness: Optional[np.ndarray] = None,
    wall_thickness_ratio: float = 0.1,
    min_wall_thickness: float = 1e-9,
) -> GenerationWaveformData:
    """
    Compute per-generation waveform statistics without plotting.

    Returns
    -------
    GenerationWaveformData
        generations: unique generation indices (shape (G,))
        counts: number of vessels per generation (shape (G,))
        time: time axis used for statistics (shape (T,))
        metrics: dict with keys 'flow', 'pressure', 'wss', and optionally 'intramural_stress'
        normalized_time: whether the time axis was normalized to [0, 1]
    """
    if results is None:
        raise ValueError("StructuredTreeResults is required.")
    if wss_flow not in {"in", "out"}:
        raise ValueError("wss_flow must be 'in' or 'out'.")

    time_mask = _select_time_indices(results.time, time_window)
    time = np.asarray(results.time, dtype=float)[time_mask]
    collapsed = _resolve_collapsed_mask(results, store)

    try:
        flow_ts = np.asarray(getattr(results, flow_field), dtype=float)
    except AttributeError as exc:
        raise AttributeError(f"StructuredTreeResults is missing flow field '{flow_field}'.") from exc
    try:
        pressure_ts = np.asarray(getattr(results, pressure_field), dtype=float)
    except AttributeError as exc:
        raise AttributeError(f"StructuredTreeResults is missing pressure field '{pressure_field}'.") from exc

    if convert_pressure_to_mmhg:
        pressure_ts = pressure_ts * MMHG_PER_BARYE

    pressure_ts = pressure_ts[:, time_mask]
    flow_ts = flow_ts[:, time_mask]

    if take_abs_flow_for_wss:
        try:
            flow_for_wss = np.asarray(getattr(results, f"flow_{wss_flow}"), dtype=float)
        except AttributeError as exc:
            raise AttributeError(
                f"StructuredTreeResults is missing flow field 'flow_{wss_flow}' required for WSS computation."
            ) from exc
        flow_for_wss = np.abs(flow_for_wss)
        r = 0.5 * np.asarray(results.d, dtype=float)[:, None]
        eps = np.finfo(np.float64).tiny
        wss_ts = 4.0 * float(results.eta) * flow_for_wss / (np.pi * np.maximum(r**3, eps))
    else:
        wss_ts = results.wss_timeseries(use_flow=wss_flow)
    wss_ts = np.asarray(wss_ts, dtype=float)[:, time_mask]

    ims_ts = None
    if include_intramural:
        ims_ts = _intramural_stress_timeseries(
            pressure_ts,
            results.d,
            wall_thickness=wall_thickness,
            wall_thickness_ratio=wall_thickness_ratio,
            min_wall_thickness=min_wall_thickness,
        )

    mask = np.isfinite(results.gen)
    if not include_collapsed:
        mask &= ~collapsed

    if not np.any(mask):
        raise ValueError("No vessels available after applying masks.")

    generations = np.asarray(results.gen, dtype=np.int32)[mask]
    flow = flow_ts[mask]
    pressure = pressure_ts[mask]
    wss = wss_ts[mask]

    gens_unique = np.unique(generations)
    counts = np.asarray([np.count_nonzero(generations == g) for g in gens_unique], dtype=int)

    flow_stats = _per_generation_stats(generations, flow, gens_unique)
    pressure_stats = _per_generation_stats(generations, pressure, gens_unique)
    wss_stats = _per_generation_stats(generations, wss, gens_unique)

    metrics = {"flow": flow_stats, "pressure": pressure_stats, "wss": wss_stats}
    if include_intramural and ims_ts is not None:
        intramural = ims_ts[mask]
        ims_stats = _per_generation_stats(generations, intramural, gens_unique)
        metrics["intramural_stress"] = ims_stats

    t_plot = time.copy()
    if normalize_time:
        t_span = float(t_plot[-1] - t_plot[0]) if t_plot.size else 0.0
        if t_span <= 0.0:
            raise ValueError("Cannot normalize time when all samples share the same timestamp.")
        t_plot = (t_plot - t_plot[0]) / t_span

    return GenerationWaveformData(
        generations=gens_unique,
        counts=counts,
        time=t_plot,
        metrics=metrics,
        normalized_time=normalize_time,
    )


def compute_generation_waveforms_for_tree(
    tree: "StructuredTree",
    **kwargs,
) -> GenerationWaveformData:
    """
    Convenience wrapper accepting a StructuredTree instance.
    The tree must have `.store` and `.results` populated (e.g., after simulate()).
    """
    if not hasattr(tree, "results") or tree.results is None:
        raise ValueError("StructuredTree instance does not contain results. Run simulate() first.")
    if not hasattr(tree, "store") or tree.store is None:
        raise ValueError("StructuredTree instance does not contain storage (call build() first).")
    return compute_generation_waveforms(tree.results, tree.store, **kwargs)


def plot_generation_waveforms(
    results: StructuredTreeResults,
    store: Optional[StructuredTreeStorage] = None,
    *,
    time_window: Optional[Tuple[float, float]] = None,
    flow_field: str = "flow_in",
    pressure_field: str = "pressure_in",
    wss_flow: str = "in",
    include_collapsed: bool = True,
    take_abs_flow_for_wss: bool = False,
    convert_pressure_to_mmhg: bool = True,
    normalize_time: bool = False,
    include_intramural: bool = False,
    wall_thickness: Optional[np.ndarray] = None,
    wall_thickness_ratio: float = 0.1,
    min_wall_thickness: float = 1e-9,
    figsize: Tuple[float, float] = (10.5, 9.0),
    linewidth: float = 1.4,
    alpha_band: float = 0.20,
    cmap_name: str = "viridis",
    panel_titles: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
    legend: bool = True,
    figure_title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, GenerationWaveformData]:
    data = compute_generation_waveforms(
        results,
        store,
        time_window=time_window,
        flow_field=flow_field,
        pressure_field=pressure_field,
        wss_flow=wss_flow,
        include_collapsed=include_collapsed,
        take_abs_flow_for_wss=take_abs_flow_for_wss,
        convert_pressure_to_mmhg=convert_pressure_to_mmhg,
        normalize_time=normalize_time,
        include_intramural=include_intramural,
        wall_thickness=wall_thickness,
        wall_thickness_ratio=wall_thickness_ratio,
        min_wall_thickness=min_wall_thickness,
    )

    if panel_titles is None:
        panel_titles = (
            "Flow by Generation (waveforms)",
            "Pressure by Generation (waveforms)",
            "Wall Shear Stress by Generation (waveforms)",
        )
        if include_intramural:
            panel_titles = panel_titles + ("Intramural Stress by Generation (waveforms)",)
    if ylabels is None:
        ylabels = (
            "Flow",
            "Pressure [mmHg]" if convert_pressure_to_mmhg else "Pressure",
            "WSS",
        )
        if include_intramural:
            ylabels = ylabels + ("Intramural Stress",)

    n_panels = 4 if include_intramural else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(hspace=0.28)

    cmap = plt.get_cmap(cmap_name)
    t = data.time

    metric_keys = ("flow", "pressure", "wss")
    if include_intramural:
        metric_keys = metric_keys + ("intramural_stress",)

    for ax, metric, title, ylabel in zip(axes, metric_keys, panel_titles, ylabels):
        stats = data.metrics[metric]
        for idx, gen in enumerate(data.generations):
            color = cmap(idx / max(1, len(data.generations) - 1))
            label = f"G{int(gen)} (n={int(data.counts[idx])})"
            mean = stats.mean[idx]
            q25 = stats.q25[idx]
            q75 = stats.q75[idx]
            ax.plot(t, mean, color=color, linewidth=linewidth, label=label)
            ax.fill_between(t, q25, q75, color=color, alpha=alpha_band, linewidth=0)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Normalized Time (0-1)" if data.normalized_time else "Time", fontsize=11)

    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncols=min(len(labels), 6),
                frameon=False,
                fontsize=10,
                bbox_to_anchor=(0.5, 0.97),
            )
            plt.subplots_adjust(top=0.92 if figure_title else 0.88)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14, y=0.995 if legend else 0.98)

    return fig, axes, data


def plot_generation_waveforms_for_tree(
    tree: "StructuredTree",
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray, GenerationWaveformData]:
    if not hasattr(tree, "results") or tree.results is None:
        raise ValueError("StructuredTree instance does not contain results. Run simulate() first.")
    if not hasattr(tree, "store") or tree.store is None:
        raise ValueError("StructuredTree instance does not contain storage (call build() first).")
    return plot_generation_waveforms(tree.results, tree.store, **kwargs)


def _load_tree_pickle(path: str) -> "StructuredTree":
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot structured tree flow/pressure/WSS waveforms by generation.",
    )
    parser.add_argument(
        "tree_pickle",
        help="Path to a pickled StructuredTree object with populated results.",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        metavar=("T0", "T1"),
        help="Restrict visualization to the time interval [T0, T1].",
    )
    parser.add_argument(
        "--flow-field",
        default="flow_in",
        help="Flow field name on StructuredTreeResults (default: flow_in).",
    )
    parser.add_argument(
        "--pressure-field",
        default="pressure_in",
        help="Pressure field name on StructuredTreeResults (default: pressure_in).",
    )
    parser.add_argument(
        "--wss-flow",
        default="in",
        choices=("in", "out"),
        help="Use inlet or outlet flow when computing WSS (default: in).",
    )
    parser.add_argument(
        "--exclude-collapsed",
        action="store_true",
        help="Exclude vessels flagged as collapsed.",
    )
    parser.add_argument(
        "--abs-flow-wss",
        action="store_true",
        help="Use |Q| when computing WSS.",
    )
    parser.add_argument(
        "--no-pressure-mmhg",
        dest="convert_pressure_to_mmhg",
        action="store_false",
        help="Keep pressure in original units.",
    )
    parser.add_argument(
        "--normalize-time",
        action="store_true",
        help="Normalize time axis to [0, 1].",
    )
    parser.add_argument(
        "--figure-title",
        help="Optional figure title.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    tree = _load_tree_pickle(args.tree_pickle)
    fig, _, _ = plot_generation_waveforms_for_tree(
        tree,
        time_window=tuple(args.time_window) if args.time_window else None,
        flow_field=args.flow_field,
        pressure_field=args.pressure_field,
        wss_flow=args.wss_flow,
        include_collapsed=not args.exclude_collapsed,
        take_abs_flow_for_wss=args.abs_flow_wss,
        convert_pressure_to_mmhg=args.convert_pressure_to_mmhg,
        normalize_time=args.normalize_time,
        figure_title=args.figure_title,
    )
    fig.show()


if __name__ == "__main__":
    _main()
