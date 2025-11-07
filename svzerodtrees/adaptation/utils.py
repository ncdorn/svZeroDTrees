import numpy as np
import matplotlib.pyplot as plt
from ..microvasculature.utils import assign_flow_to_root
from filelock import FileLock
import pandas as pd
import os

'''
adaptation utils
'''

import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def plot_adaptation_histories(sol_y, lpa_idx, figures_path=None):
    """
    Create 2x2 log-scale plots of radius and thickness distributions for LPA and RPA over time.
    Each subplot contains translucent histograms and smoothed KDE curves for 4 time points.

    Parameters:
    - sol_y: solution array from solve_ivp, shape (2N, M)
    - lpa_idx: number of LPA vessels
    - figures_path: optional path to save the figure

    Returns:
    - fig: matplotlib Figure object
    """

    sol_y = np.array(sol_y)
    time_indices = np.linspace(0, sol_y.shape[1] - 1, 4, dtype=int)

    colors = ['blue', 'orange', 'green', 'red']
    labels = ['Start', '1/3', '2/3', 'End']

    r_all = sol_y[0::2, :]
    h_all = sol_y[1::2, :]

    lpa_r = r_all[:lpa_idx, :]
    lpa_h = h_all[:lpa_idx, :]
    rpa_r = r_all[lpa_idx:, :]
    rpa_h = h_all[lpa_idx:, :]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    def plot_kde_hist(ax, data_matrix, title, x_label):
        x_min = np.min(data_matrix[data_matrix > 0])
        x_max = np.max(data_matrix)
        log_x = np.logspace(np.log10(x_min * 0.9), np.log10(x_max * 1.1), 500)

        for i, idx in enumerate(time_indices):
            values = data_matrix[:, idx]
            values = values[values > 0]
            if len(values) < 2:
                continue

            # Plot histogram
            ax.hist(values, bins=20, color=colors[i], alpha=0.2, label=None)

            # Plot KDE
            kde = gaussian_kde(values)
            y = kde(log_x)
            ax.plot(log_x, y, color=colors[i], lw=2, alpha=0.8, label=labels[i])

        ax.set_xscale('log')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.legend()

    plot_kde_hist(axs[0], lpa_r, "LPA Radius Distribution", "Radius")
    plot_kde_hist(axs[1], lpa_h, "LPA Thickness Distribution", "Thickness")
    plot_kde_hist(axs[2], rpa_r, "RPA Radius Distribution", "Radius")
    plot_kde_hist(axs[3], rpa_h, "RPA Thickness Distribution", "Thickness")

    fig.tight_layout()

    if figures_path is not None:
        os.makedirs(figures_path, exist_ok=True)
        out_path = os.path.join(figures_path, "adaptation_kde_hist_overlay.png")
        fig.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")

    return fig

def plot_adaptation_histories_nohist(sol_y, lpa_idx, figures_path=None):
    """
    Create 2x2 smoothed KDE plots (on log scale) of radius and thickness distributions
    for LPA and RPA over time from solve_ivp output.

    Parameters:
    - sol_y: solution array from solve_ivp, shape (2N, M)
    - lpa_idx: number of LPA vessels (RPA starts at lpa_idx)
    - figures_path: directory to save figure (optional)

    Returns:
    - fig: matplotlib Figure object
    """

    sol_y = np.array(sol_y)
    time_indices = np.linspace(0, sol_y.shape[1] - 1, 4, dtype=int)

    colors = ['blue', 'orange', 'green', 'red']
    labels = ['Start', '1/3', '2/3', 'End']

    # Unpack r and h across time
    r_all = sol_y[0::2, :]  # radii rows: vessels, cols: time
    h_all = sol_y[1::2, :]  # thickness

    # Split into LPA and RPA
    lpa_r = r_all[:lpa_idx, :]
    lpa_h = h_all[:lpa_idx, :]
    rpa_r = r_all[lpa_idx:, :]
    rpa_h = h_all[lpa_idx:, :]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    def plot_kde(ax, data_matrix, title, x_label):
        x_min = np.min(data_matrix)
        x_max = np.max(data_matrix)

        log_x = np.logspace(np.log10(x_min * 0.9), np.log10(x_max * 1.1), 500)

        for i, idx in enumerate(time_indices):
            values = data_matrix[:, idx]
            values = values[values > 0]  # KDE cannot handle zeros or negatives
            if len(values) > 1:
                kde = gaussian_kde(values)
                y = kde(log_x)
                ax.plot(log_x, y, color=colors[i], label=labels[i], lw=2, alpha=0.7)

        ax.set_xscale('log')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.legend()

    plot_kde(axs[0], lpa_r, "LPA Radius Distribution (log scale)", "Radius")
    plot_kde(axs[1], lpa_h, "LPA Thickness Distribution (log scale)", "Thickness")
    plot_kde(axs[2], rpa_r, "RPA Radius Distribution (log scale)", "Radius")
    plot_kde(axs[3], rpa_h, "RPA Thickness Distribution (log scale)", "Thickness")

    fig.tight_layout()

    if figures_path is not None:
        os.makedirs(figures_path, exist_ok=True)
        out_path = os.path.join(figures_path, "adaptation_kde_log_histograms.png")
        fig.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")

    return fig


def r_h_histogram(y, lpa_idx, path=None):
    """
    Create a 2x2 histogram figure showing radius and thickness distributions
    for LPA and RPA vessels, with vertical lines at the means.

    Parameters:
    - y: state vector (alternating r, h)
    - lpa_idx: number of LPA vessels; RPA starts at index lpa_idx
    - path: optional path to save the figure

    Returns:
    - fig: matplotlib Figure object
    """
    y = np.array(y)
    r_all = y[0::2]
    h_all = y[1::2]

    # Partition LPA and RPA
    lpa_r = r_all[:lpa_idx]
    lpa_h = h_all[:lpa_idx]
    rpa_r = r_all[lpa_idx:]
    rpa_h = h_all[lpa_idx:]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # LPA radius
    axs[0, 0].hist(lpa_r, bins=20, color='blue', alpha=0.7)
    mean_lpa_r = np.mean(lpa_r)
    axs[0, 0].axvline(mean_lpa_r, color='black', linestyle='--', linewidth=2, label=f"Mean = {mean_lpa_r:.3g}")
    axs[0, 0].set_title("LPA Radius Histogram")
    axs[0, 0].set_xlabel("Radius")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].legend()

    # LPA thickness
    axs[0, 1].hist(lpa_h, bins=20, color='green', alpha=0.7)
    mean_lpa_h = np.mean(lpa_h)
    axs[0, 1].axvline(mean_lpa_h, color='black', linestyle='--', linewidth=2, label=f"Mean = {mean_lpa_h:.3g}")
    axs[0, 1].set_title("LPA Thickness Histogram")
    axs[0, 1].set_xlabel("Thickness")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].legend()

    # RPA radius
    axs[1, 0].hist(rpa_r, bins=20, color='blue', alpha=0.7)
    mean_rpa_r = np.mean(rpa_r)
    axs[1, 0].axvline(mean_rpa_r, color='black', linestyle='--', linewidth=2, label=f"Mean = {mean_rpa_r:.3g}")
    axs[1, 0].set_title("RPA Radius Histogram")
    axs[1, 0].set_xlabel("Radius")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].legend()

    # RPA thickness
    axs[1, 1].hist(rpa_h, bins=20, color='green', alpha=0.7)
    mean_rpa_h = np.mean(rpa_h)
    axs[1, 1].axvline(mean_rpa_h, color='black', linestyle='--', linewidth=2, label=f"Mean = {mean_rpa_h:.3g}")
    axs[1, 1].set_title("RPA Thickness Histogram")
    axs[1, 1].set_xlabel("Thickness")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].legend()

    fig.tight_layout()

    if path is not None:
        fig.savefig(path)
        print(f"Saved histogram to {path}")

    return fig

def append_result_to_csv(df: pd.DataFrame, output_path: str):
    lock_path = output_path + '.lock'
    lock = FileLock(lock_path)

    with lock:
        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=not file_exists, index=False)

# packing/unpacking helper functions
def pack_state(*trees):
    """
    Flatten structured-tree radii and wall thicknesses into a 1-D state vector.

    Accepts individual StructuredTree instances or iterables containing them.
    Radii are taken directly from the stored diameters; thickness defaults to r/10
    unless a previous adaptation pass stored ``tree._thickness_state``.
    """
    if len(trees) == 1 and isinstance(trees[0], (list, tuple)):
        trees = tuple(trees[0])

    trees = [tree for tree in trees if tree is not None]
    if not trees:
        return np.empty(0, dtype=np.float64)

    radii_blocks = []
    thickness_blocks = []

    for tree in trees:
        store = getattr(tree, "store", None)
        if store is None:
            raise AttributeError("StructuredTree instance is missing built storage; call build() first.")

        n_nodes = int(store.n_nodes())
        if n_nodes == 0:
            continue

        radii = 0.5 * np.asarray(store.d, dtype=np.float64, order="C")

        thickness = getattr(tree, "_thickness_state", None)
        if thickness is None:
            thickness = radii / 10.0
        else:
            thickness = np.asarray(thickness, dtype=np.float64, order="C")
            if thickness.size != n_nodes:
                thickness = radii / 10.0

        thickness = np.maximum(thickness, 1e-9)
        tree._thickness_state = thickness.copy()

        radii_blocks.append(radii)
        thickness_blocks.append(thickness)

    if not radii_blocks:
        return np.empty(0, dtype=np.float64)

    all_radii = np.concatenate(radii_blocks)
    all_thickness = np.concatenate(thickness_blocks)

    y = np.empty(2 * all_radii.size, dtype=np.float64)
    y[0::2] = all_radii
    y[1::2] = all_thickness
    return y

def unpack_state(y, *trees):
    """
    Scatter a flattened state vector back into StructuredTree storage.
    Updates diameters in-place and caches the thickness per tree.
    """
    if len(trees) == 1 and isinstance(trees[0], (list, tuple)):
        trees = tuple(trees[0])

    trees = [tree for tree in trees if tree is not None]
    if not trees:
        return

    y = np.asarray(y, dtype=np.float64)

    expected_nodes = 0
    for tree in trees:
        store = getattr(tree, "store", None)
        if store is None:
            raise AttributeError("StructuredTree instance is missing built storage; call build() first.")
        expected_nodes += int(store.n_nodes())

    if y.size != 2 * expected_nodes:
        raise ValueError(f"State vector length {y.size} does not match {expected_nodes} vessels.")

    offset = 0
    for tree in trees:
        store = tree.store
        n_nodes = int(store.n_nodes())
        if n_nodes == 0:
            continue

        segment = y[2 * offset: 2 * (offset + n_nodes)]
        radii = segment[0::2]
        thickness = np.maximum(segment[1::2], 1e-9)

        store.d = (2.0 * radii).astype(store.d.dtype, copy=False)
        tree._thickness_state = thickness.copy()

        offset += n_nodes


def simulate_outlet_trees(simple_pa):
    '''
    get lpa/rpa flow, simulate the micro trees and assign flow to tree vessels
    '''
    lpa_flow = np.mean(simple_pa.result[simple_pa.result.name=='branch2_seg0']['flow_out'])
    rpa_flow = np.mean(simple_pa.result[simple_pa.result.name=='branch4_seg0']['flow_out'])
    simple_pa.lpa_tree.simulate([lpa_flow, lpa_flow])
    simple_pa.rpa_tree.simulate([rpa_flow, rpa_flow])


def rel_change(y, y_ref):
        """largest element-wise fractional change |Δy|/|y_ref|"""
        return np.max(np.abs((y - y_ref) / y_ref))
    

def time_to_95(sol):
    """first time point where every state is within 5 % of final value"""
    y_end = sol.y[:, -1]
    for ti, yi in zip(sol.t, sol.y.T):
        if rel_change(yi, y_end) < 0.05:
            return ti
    return np.nan


def wrap_event(event_func, *extra_args):
    def wrapped_event(t, y, *args):
        return event_func(t, y, *extra_args)
    wrapped_event.terminal = getattr(event_func, "terminal", False)
    wrapped_event.direction = getattr(event_func, "direction", 0)
    return wrapped_event
