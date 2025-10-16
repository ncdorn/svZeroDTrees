import math
from typing import Iterable, Callable, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Helpers
# =========================
def _as_scalar(x: Any, reducer: Callable = np.nanmean):
    """Return a scalar from either a scalar or an array-like."""
    if x is None:
        return np.nan
    x = np.asarray(x)
    return reducer(x) if x.ndim > 0 else float(x)

def _compute_wss_from_q_d(q: float, d: float, mu: float) -> float:
    """
    Compute wall shear stress tau_w = 4*mu*Q / (pi * R^3), units consistent with inputs.
    q : flow
    d : diameter
    mu: dynamic viscosity
    """
    if np.isnan(q) or np.isnan(d) or d <= 0:
        return np.nan
    r = 0.5 * d
    return (4.0 * mu * q) / (math.pi * (r ** 3))

def _summary_by_generation(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Return per-generation summary stats: median, IQR (q25, q75), mean, std, n.
    """
    g = df.groupby("generation")[value_col]
    out = pd.DataFrame({
        "median": g.median(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
        "mean": g.mean(),
        "std": g.std(),
        "n": g.count()
    }).reset_index()
    return out

def _stripplot_with_iqr(ax, x, y, x_jitter=0.12, alpha=0.65, markersize=30):
    """Simple jittered scatter to show distribution per generation."""
    rng = np.random.default_rng(42)
    xj = x + rng.uniform(-x_jitter, x_jitter, size=len(x))
    ax.scatter(xj, y, s=markersize, alpha=alpha, linewidths=0)

def _format_panel(ax, ylabel: str):
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

def _annotate_n(ax, summary_df: pd.DataFrame, y_frac=0.95):
    """Annotate sample sizes per generation along the top of the panel."""
    ymin, ymax = ax.get_ylim()
    y = ymin + y_frac * (ymax - ymin)
    for _, row in summary_df.iterrows():
        ax.text(row["generation"], y, f"n={int(row['n'])}",
                ha="center", va="top", fontsize=9, alpha=0.75)

# =========================
# Main plotting function
# =========================
def plot_tree_metrics_by_generation(
    vessels: Iterable[Any],
    mu: float = 0.04,
    reducer: Callable = np.nanmean,
    flow_attr: str = "Q",
    pressure_attr: str = "P_in",
    diameter_attr: str = "d",
    generation_attr: str = "gen",
    figsize: Tuple[int, int] = (9, 9),
    wspace: float = 0.18,
    hspace: float = 0.28,
    titles: Tuple[str, str, str] = ("Flow by Generation",
                                    "Pressure by Generation",
                                    "Wall Shear Stress by Generation"),
    ylabels: Tuple[str, str, str] = ("Flow", "Pressure", "WSS"),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Build a tidy DataFrame from TreeVessel objects, compute WSS, and create a 3-row figure:
    row 1: Flow vs Generation
    row 2: Pressure vs Generation
    row 3: WSS vs Generation

    Parameters
    ----------
    vessels : iterable of objects with attributes:
        .generation (int), .flow, .pressure, .diameter (each scalar or array-like)
    mu : viscosity in units consistent with flow & diameter
    reducer : aggregation for time-series -> scalar (default: np.nanmean)
    *attr : names of attributes on each vessel (override if your class differs)
    """
    # Build the table
    rows = []
    for v in vessels:
        try:
            gen = getattr(v, generation_attr)
        except AttributeError:
            continue
        q = _as_scalar(getattr(v, flow_attr, np.nan), reducer=reducer)
        p = _as_scalar(getattr(v, pressure_attr, np.nan), reducer=reducer) / 1333.2
        d = _as_scalar(getattr(v, diameter_attr, np.nan), reducer=reducer)
        tau = _compute_wss_from_q_d(q, d, mu)
        rows.append({"generation": int(gen), "flow": q, "pressure": p, "wss": tau, "diameter": d})

    df = pd.DataFrame(rows).dropna(subset=["generation"])
    if df.empty:
        raise ValueError("No valid vessel data found (check attribute names/values).")

    # Summaries
    flow_s = _summary_by_generation(df, "flow")
    pres_s = _summary_by_generation(df, "pressure")
    wss_s  = _summary_by_generation(df, "wss")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    panels = [
        ("flow", flow_s, titles[0], ylabels[0]),
        ("pressure", pres_s, titles[1], ylabels[1]),
        ("wss", wss_s, titles[2], ylabels[2]),
    ]

    # Ensure integer ticks for generations
    gens_sorted = np.sort(df["generation"].unique())

    for ax, (col, summ, title, ylabel) in zip(axes, panels):
        # Raw jittered points
        _stripplot_with_iqr(ax, df["generation"].values, df[col].values, markersize=24, alpha=0.55)

        # Median line with IQR band
        ax.plot(summ["generation"], summ["median"], marker="o", linewidth=2)
        ax.fill_between(
            summ["generation"].values,
            summ["q25"].values,
            summ["q75"].values,
            alpha=0.25
        )

        ax.set_title(title, fontsize=12, pad=8)
        _format_panel(ax, ylabel)
        _annotate_n(ax, summ)

    axes[-1].set_xlabel("Generation", fontsize=11)
    axes[-1].set_xticks(gens_sorted)
    axes[-1].set_xticklabels([str(g) for g in gens_sorted], fontsize=10)

    # A little breathing room
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        ax.set_ylim(ymin - 0.05*span, ymax + 0.10*span)

    return fig, axes

import math
from typing import Iterable, Callable, Any, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# -----------------------------
# Utilities
# -----------------------------
def _ensure_1d(a) -> np.ndarray:
    a = np.asarray(a).squeeze()
    if a.ndim != 1:
        raise ValueError("Expected a 1D array for time or signal.")
    return a

def _broadcast_if_scalar(x, like: np.ndarray):
    """If x is scalar, broadcast to like's shape; else validate 1D length match."""
    x = np.asarray(x)
    if x.ndim == 0:
        return np.full_like(like, float(x), dtype=float)
    x = _ensure_1d(x)
    if x.size != like.size:
        raise ValueError("Signal length does not match its time vector.")
    return x.astype(float)

def _resample_to(t_src: np.ndarray, y_src: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """Linear interpolation to t_ref. Assumes t_src is increasing."""
    t_src = _ensure_1d(t_src)
    y_src = _ensure_1d(y_src)
    if not np.all(np.diff(t_src) >= 0):
        # sort just in case
        order = np.argsort(t_src)
        t_src = t_src[order]
        y_src = y_src[order]
    # Clip to bounds to avoid extrapolation surprises
    t_lo, t_hi = t_src[0], t_src[-1]
    tq = np.clip(t_ref, t_lo, t_hi)
    return np.interp(tq, t_src, y_src)

def _make_time_reference(
    vessels: Iterable[Any],
    time_attr: str,
    n_time: int,
    normalize_time: bool
) -> np.ndarray:
    """
    Build a common time vector:
      - If normalize_time: [0,1] with n_time samples
      - Else: tries to detect a common time array; if not, uses global min..max and uniform grid
    """
    if normalize_time:
        return np.linspace(0.0, 1.0, n_time)

    # Collect candidate times
    times: List[np.ndarray] = []
    for v in vessels:
        if hasattr(v, time_attr):
            t = getattr(v, time_attr)
            if t is not None:
                t = _ensure_1d(t)
                times.append(t)
    if not times:
        raise ValueError(f"No '{time_attr}' arrays found on vessels.")

    # If all identical length+values, reuse
    same = all(len(times[0]) == len(t) and np.allclose(times[0], t) for t in times[1:])
    if same:
        return times[0].astype(float)

    # Otherwise build a global reference spanning min..max
    t_min = min(float(t[0]) for t in times)
    t_max = max(float(t[-1]) for t in times)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("Could not infer a global time range from vessel time arrays.")
    return np.linspace(t_min, t_max, n_time)

def _wss_from_q_d(q: np.ndarray, d: np.ndarray, mu: float) -> np.ndarray:
    # tau = 4*mu*Q / (pi*R^3), R = D/2 -> pi*(D/2)^3
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 0.5 * d
        tau = (4.0 * mu * q) / (math.pi * np.power(r, 3))
        tau[~np.isfinite(tau)] = np.nan
    return tau

def _gen_colors(gens: np.ndarray, cmap_name: str = "viridis") -> Dict[int, Tuple[float, float, float, float]]:
    cmap = get_cmap(cmap_name)
    uniq = np.sort(np.unique(gens))
    return {g: cmap(i / max(1, len(uniq)-1)) for i, g in enumerate(uniq)}

def _percentile_stack(arr: np.ndarray, q: float, axis=0) -> np.ndarray:
    return np.nanpercentile(arr, q, axis=axis)

# -----------------------------
# Main plotting function
# -----------------------------
def plot_waveforms_by_generation(
    vessels: Iterable[Any],
    mu: float = 0.04,
    time_attr: str = "t",
    flow_attr: str = "Q",
    pressure_attr: str = "P_in",
    diameter_attr: str = "d",
    generation_attr: str = "gen",
    n_time: int = 500,
    normalize_time: bool = False,
    take_abs_flow_for_wss: bool = False,
    figsize: Tuple[int, int] = (10.5, 9.0),
    linewidth: float = 2.0,
    alpha_band: float = 0.20,
    cmap_name: str = "viridis",
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Plot mean +/- IQR waveforms of flow, pressure, and WSS by generation.

    Parameters
    ----------
    vessels : iterable of objects with:
        generation: int
        t: 1D array-like (shared or per-vessel)
        flow, pressure, diameter: 1D array-like (same length as t or scalar for diameter)
    mu : dynamic viscosity (units must match Q and D; e.g., 0.004 Pa·s in SI or ~0.035 dyn·s/cm² in cgs)
    normalize_time : if True, reparameterize each vessel's time to [0,1] before aggregation
    take_abs_flow_for_wss : if True, uses |Q(t)| to compute WSS
    Returns
    -------
    fig, axes, t_ref
    """
    vessels = list(vessels)
    if not vessels:
        raise ValueError("Empty vessel list.")

    # Build reference time base
    t_ref = _make_time_reference(vessels, time_attr, n_time, normalize_time)

    # Collect per-generation stacks
    stacks: Dict[int, Dict[str, List[np.ndarray]]] = {}
    gens_all = []

    for v in vessels:
        if not hasattr(v, generation_attr):
            continue
        gen = int(getattr(v, generation_attr))

        # Time handling
        if not hasattr(v, time_attr):
            continue
        t = _ensure_1d(getattr(v, time_attr))

        # Normalize time if requested
        if normalize_time:
            # Map t -> [0,1]
            t0, t1 = float(t[0]), float(t[-1])
            if t1 <= t0:
                continue
            t_norm = (t - t0) / (t1 - t0)
            t_use = t_norm
        else:
            t_use = t

        # Signals
        q = _broadcast_if_scalar(getattr(v, flow_attr, np.nan), t_use)
        p = _broadcast_if_scalar(getattr(v, pressure_attr, np.nan), t_use)
        d = _broadcast_if_scalar(getattr(v, diameter_attr, np.nan), t_use)

        if take_abs_flow_for_wss:
            q_for_wss = np.abs(q)
        else:
            q_for_wss = q

        # Resample to reference
        q_ref = _resample_to(t_use, q, t_ref)
        p_ref = _resample_to(t_use, p, t_ref)
        d_ref = _resample_to(t_use, d, t_ref)
        tau_ref = _wss_from_q_d(q_ref, d_ref, mu)

        # Store
        if gen not in stacks:
            stacks[gen] = {"flow": [], "pressure": [], "wss": []}
        stacks[gen]["flow"].append(q_ref)
        stacks[gen]["pressure"].append(p_ref)
        stacks[gen]["wss"].append(tau_ref)
        gens_all.append(gen)

    if not stacks:
        raise ValueError("No usable time-series found. Check attribute names and lengths.")

    gens_unique = np.sort(np.unique(gens_all))
    colors = _gen_colors(gens_unique, cmap_name)

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    titles = ["Flow by Generation (waveforms)", "Pressure by Generation (waveforms)", "Wall Shear Stress by Generation (waveforms)"]
    ylabels = ["Flow", "Pressure", "WSS"]

    # Plot per metric
    for ax, metric, title, ylabel in zip(axes, ["flow", "pressure", "wss"], titles, ylabels):
        for g in gens_unique:
            arr = np.vstack(stacks[g][metric])  # shape: (n_vessels_in_gen, n_time)
            mean = np.nanmean(arr, axis=0)
            q25  = _percentile_stack(arr, 25.0, axis=0)
            q75  = _percentile_stack(arr, 75.0, axis=0)

            ax.plot(t_ref, mean, label=f"G{g}", linewidth=linewidth, color=colors[g])
            ax.fill_between(t_ref, q25, q75, color=colors[g], alpha=alpha_band, linewidth=0)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

    # X axis labeling
    axes[-1].set_xlabel("Normalized Time (0–1)" if normalize_time else "Time", fontsize=11)

    # Legend (single, compact)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 8), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.95))

    plt.subplots_adjust(hspace=0.28, top=0.85)
    return fig, axes, t_ref

# -----------------------------
# Example
# -----------------------------
# class TreeVessel:
#     def __init__(self, generation, t, q, p, d):
#         self.generation = generation
#         self.t = np.asarray(t)
#         self.flow = np.asarray(q)
#         self.pressure = np.asarray(p)
#         self.diameter = np.asarray(d)
#
# # Suppose different vessels have different time vectors:
# t1 = np.linspace(0, 1.0, 250)
# t2 = np.linspace(0, 0.9, 200)
# v1 = TreeVessel(0, t1, 12*np.sin(2*np.pi*t1)+20, 90+5*np.sin(2*np.pi*(t1-0.1)), 0.004*np.ones_like(t1))
# v2 = TreeVessel(1, t2, 6*np.sin(2*np.pi*t2)+10, 88+4*np.sin(2*np.pi*(t2-0.15)), 0.003*np.ones_like(t2))
# fig, axes, t_ref = plot_waveforms_by_generation([v1, v2], mu=0.004, normalize_time=True)
# plt.show()