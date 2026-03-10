"""
Generation-level flow, pressure, and wall shear stress visualizations for structured trees.

Utilities here operate directly on StructuredTreeStorage/StructuredTreeResults and
provide a CLI for quick figure generation from pickled StructuredTree objects.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ...microvasculature.structured_tree.results import StructuredTreeResults
from ...microvasculature.structured_tree.storage import StructuredTreeStorage

if TYPE_CHECKING:
    from ...microvasculature.structured_tree.structuredtree import StructuredTree

TimeReducer = Callable[[np.ndarray], np.ndarray]

MMHG_PER_BARYE = 1.0 / 1333.22  # convert dyn/cm^2 → mmHg


def _default_time_reducer(values: np.ndarray) -> np.ndarray:
    """Average each row across time, ignoring NaNs."""
    return np.nanmean(values, axis=1)


@dataclass
class GenerationSummary:
    generation: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


def _summarize_by_generation(gen: np.ndarray, values: np.ndarray) -> GenerationSummary:
    """Compute per-generation summary stats for 1D values."""
    gen = np.asarray(gen, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(gen) & np.isfinite(values)
    gen = gen[valid]
    values = values[valid]

    if gen.size == 0:
        return GenerationSummary(
            generation=np.array([], dtype=np.int32),
            median=np.array([], dtype=float),
            q25=np.array([], dtype=float),
            q75=np.array([], dtype=float),
            mean=np.array([], dtype=float),
            std=np.array([], dtype=float),
            count=np.array([], dtype=int),
        )

    uniq = np.unique(gen)
    median = np.empty_like(uniq, dtype=float)
    q25 = np.empty_like(uniq, dtype=float)
    q75 = np.empty_like(uniq, dtype=float)
    mean = np.empty_like(uniq, dtype=float)
    std = np.empty_like(uniq, dtype=float)
    count = np.empty_like(uniq, dtype=int)

    for i, g in enumerate(uniq):
        subset = values[gen == g]
        subset = subset[np.isfinite(subset)]
        count[i] = subset.size
        if subset.size == 0:
            median[i] = q25[i] = q75[i] = mean[i] = std[i] = np.nan
        else:
            median[i] = np.nanmedian(subset)
            q25[i] = np.nanpercentile(subset, 25.0)
            q75[i] = np.nanpercentile(subset, 75.0)
            mean[i] = np.nanmean(subset)
            std[i] = np.nanstd(subset, ddof=0)

    return GenerationSummary(
        generation=uniq,
        median=median,
        q25=q25,
        q75=q75,
        mean=mean,
        std=std,
        count=count,
    )


def _resolve_collapsed_mask(results: StructuredTreeResults, store: Optional[StructuredTreeStorage]) -> np.ndarray:
    """Map the storage collapsed flag onto the results ordering."""
    collapsed = np.zeros(results.n_vessels, dtype=bool)
    if store is None:
        return collapsed

    ids_store = np.asarray(store.ids, dtype=np.int32)
    collapsed_store = np.asarray(store.collapsed, dtype=bool)
    lookup = {int(vid): int(idx) for idx, vid in enumerate(ids_store)}

    for row, vid in enumerate(np.asarray(results.vessel_ids, dtype=np.int32)):
        idx = lookup.get(int(vid))
        if idx is not None:
            collapsed[row] = bool(collapsed_store[idx])
    return collapsed


def _select_time_indices(times: np.ndarray, time_window: Optional[Tuple[float, float]]) -> np.ndarray:
    """Return a boolean mask selecting time samples within the requested window."""
    if time_window is None:
        return np.ones(times.shape[0], dtype=bool)
    t0, t1 = time_window
    if t1 < t0:
        t0, t1 = t1, t0
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        raise ValueError(f"Time window {time_window} selects no samples.")
    return mask


def _reduce_over_time(values: np.ndarray, mask: np.ndarray, reducer: TimeReducer) -> np.ndarray:
    """Apply reducer to each vessel row after masking time."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if mask is not None:
        arr = arr[:, mask]
    if arr.shape[1] == 0:
        raise ValueError("Selected time window contains zero samples.")
    reduced = reducer(arr)
    reduced = np.asarray(reduced, dtype=float)
    if reduced.shape != (arr.shape[0],):
        raise ValueError("Reducer must return a 1D array with length == n_vessels.")
    return reduced


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


def compute_generation_metrics(
    results: StructuredTreeResults,
    store: Optional[StructuredTreeStorage] = None,
    *,
    time_window: Optional[Tuple[float, float]] = None,
    flow_field: str = "flow_in",
    pressure_field: str = "pressure_in",
    wss_flow: str = "in",
    include_collapsed: bool = True,
    time_reducer: TimeReducer = _default_time_reducer,
    take_abs_flow_for_wss: bool = False,
    convert_pressure_to_mmhg: bool = True,
    include_intramural: bool = False,
    wall_thickness: Optional[np.ndarray] = None,
    wall_thickness_ratio: float = 0.1,
    min_wall_thickness: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """
    Build per-vessel aggregates (flow, pressure, WSS, optionally intramural stress)
    and metadata for plotting or downstream aggregation.

    Parameters
    ----------
    results : StructuredTreeResults
        Simulation outputs in structure-of-arrays form.
    store : StructuredTreeStorage, optional
        Provides collapsed flags and auxiliary metadata.
    time_window : tuple, optional
        If provided, restrict statistics to t0 <= t <= t1.
    include_collapsed : bool
        Include vessels marked as collapsed in StructuredTreeStorage. Defaults to True so
        that generational counts reflect every branch the builder created.
    time_reducer : callable
        Function mapping a (N, T_sel) array → (N,) aggregates (default: nanmean).
    take_abs_flow_for_wss : bool
        Use |Q| when computing WSS time-series.
    convert_pressure_to_mmhg : bool
        Convert pressure values from dyn/cm² to mmHg for readability.
    include_intramural : bool
        If True, compute intramural stress σθ = P * r / h.
    wall_thickness : array-like, optional
        Optional wall thickness values (scalar or length-N array). Defaults to r*wall_thickness_ratio.
    wall_thickness_ratio : float
        Thickness ratio h/r used when wall_thickness is None (default: 0.1).
    min_wall_thickness : float
        Minimum wall thickness for numerical stability.
    """
    if results is None:
        raise ValueError("StructuredTreeResults is required.")

    time_mask = _select_time_indices(results.time, time_window)
    collapsed = _resolve_collapsed_mask(results, store)

    flow_ts = getattr(results, flow_field)
    pressure_ts = getattr(results, pressure_field)
    if take_abs_flow_for_wss:
        flow_for_wss = getattr(results, f"flow_{wss_flow}")
        r = 0.5 * results.d[:, None]
        eps = np.finfo(np.float64).tiny
        wss_ts = 4.0 * results.eta * np.abs(flow_for_wss) / (np.pi * np.maximum(r**3, eps))
    else:
        wss_ts = results.wss_timeseries(use_flow=wss_flow)

    flow = _reduce_over_time(flow_ts, time_mask, time_reducer)
    pressure = _reduce_over_time(pressure_ts, time_mask, time_reducer)
    wss = _reduce_over_time(wss_ts, time_mask, time_reducer)

    if convert_pressure_to_mmhg:
        pressure = pressure * MMHG_PER_BARYE

    intramural = None
    if include_intramural:
        pressure_ts = np.asarray(pressure_ts, dtype=float)
        if convert_pressure_to_mmhg:
            pressure_ts = pressure_ts * MMHG_PER_BARYE
        ims_ts = _intramural_stress_timeseries(
            pressure_ts,
            results.d,
            wall_thickness=wall_thickness,
            wall_thickness_ratio=wall_thickness_ratio,
            min_wall_thickness=min_wall_thickness,
        )
        intramural = _reduce_over_time(ims_ts, time_mask, time_reducer)

    mask = np.isfinite(results.gen)
    if not include_collapsed:
        mask &= ~collapsed

    data = {
        "generation": np.asarray(results.gen, dtype=np.int32)[mask],
        "vessel_ids": np.asarray(results.vessel_ids, dtype=np.int32)[mask],
        "flow": flow[mask],
        "pressure": pressure[mask],
        "wss": wss[mask],
        "diameter": np.asarray(results.d, dtype=np.float64)[mask],
        "is_collapsed": collapsed[mask],
        "time_window": time_window,
    }
    if include_intramural and intramural is not None:
        data["intramural_stress"] = intramural[mask]
    return data


def summarize_generation_metrics(
    data: Dict[str, np.ndarray],
    *,
    flow_requires_positive: bool = True,
    metric_fields: Sequence[str] = ("flow", "pressure", "wss", "intramural_stress"),
) -> Dict[str, GenerationSummary]:
    """
    Compute per-generation summaries from the output of compute_generation_metrics.

    Parameters
    ----------
    data : dict
        Output of compute_generation_metrics.
    flow_requires_positive : bool
        If True, drop non-positive flow values before summarizing (log-scale safe).

    Returns
    -------
    dict
        Mapping of metric name to GenerationSummary with fields:
        - generation: unique generation indices
        - median, q25, q75, mean, std: per-generation statistics
        - count: number of vessels contributing to each generation
    """
    generations = np.asarray(data["generation"], dtype=float)
    summaries: Dict[str, GenerationSummary] = {}

    for field in metric_fields:
        if field not in data:
            continue
        values = np.asarray(data[field], dtype=float)
        if field == "flow" and flow_requires_positive:
            values = np.where(values > 0.0, values, np.nan)
        summaries[field] = _summarize_by_generation(generations, values)

    return summaries


def compute_generation_metric_summaries(
    results: StructuredTreeResults,
    store: Optional[StructuredTreeStorage] = None,
    *,
    time_window: Optional[Tuple[float, float]] = None,
    flow_field: str = "flow_in",
    pressure_field: str = "pressure_in",
    wss_flow: str = "in",
    include_collapsed: bool = True,
    time_reducer: TimeReducer = _default_time_reducer,
    take_abs_flow_for_wss: bool = False,
    convert_pressure_to_mmhg: bool = True,
    flow_requires_positive: bool = True,
    include_intramural: bool = False,
    wall_thickness: Optional[np.ndarray] = None,
    wall_thickness_ratio: float = 0.1,
    min_wall_thickness: float = 1e-9,
) -> Tuple[Dict[str, GenerationSummary], Dict[str, np.ndarray]]:
    """
    Compute per-generation summary stats without plotting.

    Returns the summaries and the per-vessel data dictionary.
    """
    data = compute_generation_metrics(
        results,
        store,
        time_window=time_window,
        flow_field=flow_field,
        pressure_field=pressure_field,
        wss_flow=wss_flow,
        include_collapsed=include_collapsed,
        time_reducer=time_reducer,
        take_abs_flow_for_wss=take_abs_flow_for_wss,
        convert_pressure_to_mmhg=convert_pressure_to_mmhg,
        include_intramural=include_intramural,
        wall_thickness=wall_thickness,
        wall_thickness_ratio=wall_thickness_ratio,
        min_wall_thickness=min_wall_thickness,
    )
    summaries = summarize_generation_metrics(
        data,
        flow_requires_positive=flow_requires_positive,
    )
    return summaries, data


def _jittered_scatter(
    ax: plt.Axes,
    generation: np.ndarray,
    values: np.ndarray,
    *,
    diameters: np.ndarray,
    cmap: plt.Colormap,
    seed: int,
    jitter: float,
    alpha: float,
    base_size: float,
    min_marker_scale: float,
) -> None:
    """Scatter with jitter and marker size scaled by diameter."""
    gen = np.asarray(generation, dtype=float)
    vals = np.asarray(values, dtype=float)
    diams = np.asarray(diameters, dtype=float)
    valid = np.isfinite(gen) & np.isfinite(vals)
    gen = gen[valid]
    vals = vals[valid]
    diams = diams[valid]
    if gen.size == 0:
        return

    rng = np.random.default_rng(seed)
    x = gen + rng.uniform(-jitter, jitter, size=gen.size)

    if np.all(~np.isfinite(diams)) or np.nanmax(diams) <= 0.0:
        sizes = np.full(gen.size, base_size)
    else:
        d_norm = diams / np.nanmax(diams)
        sizes = base_size * np.maximum(d_norm**2, min_marker_scale)

    colors = cmap((gen - gen.min()) / max(1.0, gen.max() - gen.min()))
    ax.scatter(x, vals, s=sizes, c=colors, alpha=alpha, edgecolors="none")


def plot_generation_metrics(
    results: StructuredTreeResults,
    store: Optional[StructuredTreeStorage] = None,
    *,
    time_window: Optional[Tuple[float, float]] = None,
    flow_field: str = "flow_in",
    pressure_field: str = "pressure_in",
    wss_flow: str = "in",
    include_collapsed: bool = True,
    time_reducer: TimeReducer = _default_time_reducer,
    take_abs_flow_for_wss: bool = False,
    convert_pressure_to_mmhg: bool = True,
    include_intramural: bool = False,
    wall_thickness: Optional[np.ndarray] = None,
    wall_thickness_ratio: float = 0.1,
    min_wall_thickness: float = 1e-9,
    figsize: Tuple[float, float] = (9.0, 9.5),
    jitter: float = 0.12,
    scatter_alpha: float = 0.65,
    base_marker_size: float = 240.0,
    min_marker_scale: float = 0.08,
    cmap_name: str = "viridis",
    panel_titles: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
    annot_n: bool = True,
    seed: int = 42,
    figure_title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, GenerationSummary], Dict[str, np.ndarray]]:
    """
    Create a 3-panel figure (flow, pressure, WSS) aggregated by generation.

    Returns the figure, axes, per-metric summaries, and per-vessel data.

    """
    data = compute_generation_metrics(
        results,
        store,
        time_window=time_window,
        flow_field=flow_field,
        pressure_field=pressure_field,
        wss_flow=wss_flow,
        include_collapsed=include_collapsed,
        time_reducer=time_reducer,
        take_abs_flow_for_wss=take_abs_flow_for_wss,
        convert_pressure_to_mmhg=convert_pressure_to_mmhg,
        include_intramural=include_intramural,
        wall_thickness=wall_thickness,
        wall_thickness_ratio=wall_thickness_ratio,
        min_wall_thickness=min_wall_thickness,
    )

    if panel_titles is None:
        panel_titles = (
            "Flow by Generation (log scale)",
            "Pressure by Generation",
            "Wall Shear Stress by Generation",
        )
        if include_intramural:
            panel_titles = panel_titles + ("Intramural Stress by Generation",)
    if ylabels is None:
        ylabels = ("Flow", "Pressure [mmHg]" if convert_pressure_to_mmhg else "Pressure", "WSS")
        if include_intramural:
            ylabels = ylabels + ("Intramural Stress",)

    n_panels = 4 if include_intramural else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(hspace=0.28)

    cmap = plt.get_cmap(cmap_name)
    summaries = summarize_generation_metrics(data, flow_requires_positive=True)

    metrics = (
        ("flow", panel_titles[0], ylabels[0]),
        ("pressure", panel_titles[1], ylabels[1]),
        ("wss", panel_titles[2], ylabels[2]),
    )
    if include_intramural:
        metrics = metrics + (("intramural_stress", panel_titles[3], ylabels[3]),)

    generations = data["generation"]
    diameters = data["diameter"]

    for ax, (field, title, ylabel) in zip(axes, metrics):
        values = np.asarray(data[field], dtype=float)
        values_plot = values.copy()

        if field == "flow":
            positive = values_plot > 0.0
            if not np.any(positive):
                raise ValueError("Cannot plot flow on a log scale because all flow values are non-positive.")
            values_plot = np.where(positive, values_plot, np.nan)

        _jittered_scatter(
            ax,
            generations,
            values_plot,
            diameters=diameters,
            cmap=cmap,
            seed=seed,
            jitter=jitter,
            alpha=scatter_alpha,
            base_size=base_marker_size,
            min_marker_scale=min_marker_scale,
        )
        summary = summaries[field]
        if summary.generation.size:
            ax.plot(summary.generation, summary.median, color="#303030", linewidth=2.4, marker="o")
            ax.fill_between(
                summary.generation,
                summary.q25,
                summary.q75,
                color="#bbbbbb",
                alpha=0.35,
                linewidth=0,
            )

            if field == "flow":
                ax.set_yscale("log")
                y_valid = values_plot[np.isfinite(values_plot)]
                y_min = float(np.nanmin(y_valid))
                y_max = float(np.nanmax(y_valid))
                ax.set_ylim(y_min * 0.85, y_max * 1.15)
            else:
                # Expand limits slightly for breathing room
                y_valid = values_plot[np.isfinite(values_plot)]
                if y_valid.size:
                    y_min, y_max = float(np.nanmin(y_valid)), float(np.nanmax(y_valid))
                    span = y_max - y_min if y_max > y_min else max(abs(y_min), abs(y_max), 1.0)
                    ax.set_ylim(y_min - 0.08 * span, y_max + 0.15 * span)

            if annot_n:
                ymin, ymax = ax.get_ylim()
                y_text = ymin + 0.95 * (ymax - ymin)
                for g, n in zip(summary.generation, summary.count):
                    ax.text(g, y_text, f"n={int(n)}", ha="center", va="top", fontsize=9, alpha=0.75)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

    unique_g = np.unique(generations[np.isfinite(generations)])
    axes[-1].set_xlabel("Generation", fontsize=11)
    axes[-1].set_xticks(unique_g)
    axes[-1].set_xticklabels([str(int(g)) for g in unique_g], fontsize=10)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.93)

    return fig, axes, summaries, data


def plot_generation_metrics_for_tree(
    tree: "StructuredTree",
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, GenerationSummary], Dict[str, np.ndarray]]:
    """
    Convenience wrapper accepting a StructuredTree instance.
    The tree must have `.store` and `.results` populated (e.g., after simulate()).
    """
    if not hasattr(tree, "results") or tree.results is None:
        raise ValueError("StructuredTree instance does not contain results. Run simulate() first.")
    if not hasattr(tree, "store") or tree.store is None:
        raise ValueError("StructuredTree instance does not contain storage (call build() first).")
    return plot_generation_metrics(tree.results, tree.store, **kwargs)


def _load_tree_pickle(path: str) -> "StructuredTree":
    with open(path, "rb") as fh:
        tree = pickle.load(fh)
    return tree


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot structured tree flow/pressure/WSS summaries by generation.",
    )
    parser.add_argument(
        "tree_pickle",
        help="Path to a pickled StructuredTree object with populated results.",
    )
    parser.add_argument(
        "--output",
        help="If provided, save the figure to this path (extension controls format).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots-per-inch when saving the figure.",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        metavar=("T0", "T1"),
        help="Restrict statistics to the inclusive time window [T0, T1].",
    )
    parser.add_argument(
        "--exclude-collapsed",
        action="store_true",
        help="Drop vessels flagged as collapsed from the summaries.",
    )
    parser.add_argument(
        "--no-pressure-conversion",
        action="store_true",
        help="Keep pressure in simulation units (dyn/cm^2) instead of converting to mmHg.",
    )
    parser.add_argument(
        "--title",
        help="Optional figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tree = _load_tree_pickle(args.tree_pickle)
    fig, _, _, _ = plot_generation_metrics_for_tree(
        tree,
        time_window=tuple(args.time_window) if args.time_window else None,
        include_collapsed=not args.exclude_collapsed,
        convert_pressure_to_mmhg=not args.no_pressure_conversion,
        figure_title=args.title,
    )
    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
