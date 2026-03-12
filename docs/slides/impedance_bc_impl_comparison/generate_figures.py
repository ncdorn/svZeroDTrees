#!/usr/bin/env python3
"""Generate repo-derived figures for the impedance BC implementation deck."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SVZEROD = ROOT
OUTDIR = Path(__file__).resolve().parent / "assets" / "generated"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTDIR / name, bbox_inches="tight")
    plt.close(fig)


def _kernel_plot() -> None:
    fixture = SVZEROD / "tests" / "cases" / "pulsatileFlow_R_impedance.json"
    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    z = None
    for bc in cfg["boundary_conditions"]:
        if bc.get("bc_name") == "OUT":
            z = bc["bc_values"]["z"]
            break
    if z is None:
        raise RuntimeError("Could not find OUT impedance kernel in fixture.")

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.plot(range(len(z)), z, color="#0f766e", lw=2.4)
    ax.set_title("Impedance Kernel from Test Fixture")
    ax.set_xlabel("Kernel index")
    ax.set_ylabel("z[m] (Pa*s/m^3)")
    _save(fig, "kernel_plot.png")


def _flow_pressure_plot() -> None:
    result_path = (
        SVZEROD / "tests" / "cases" / "results" / "result_pulsatileFlow_R_impedance.json"
    )
    df = pd.read_json(result_path)
    vessel = df[df["name"] == "branch0_seg0"].copy()
    period = 1.0
    tmax = float(vessel["time"].max())
    cycle_start = max(0.0, tmax - period)
    one_cycle = vessel[vessel["time"] >= cycle_start].copy()
    one_cycle["t_cycle"] = one_cycle["time"] - cycle_start
    one_cycle["pressure_out_mmhg"] = one_cycle["pressure_out"] / 1333.22

    fig, ax1 = plt.subplots(figsize=(8.4, 3.9))
    ax1.plot(one_cycle["t_cycle"], one_cycle["flow_out"], color="#0a9396", lw=2.2)
    ax1.set_xlabel("Time in cycle (s)")
    ax1.set_ylabel("Flow out (mL/s)")
    ax1.set_title("Outlet Flow and Pressure (Last Cycle)")

    ax2 = ax1.twinx()
    ax2.plot(one_cycle["t_cycle"], one_cycle["pressure_out_mmhg"], color="#b45309", lw=2.0)
    ax2.set_ylabel("Outlet pressure (mmHg)")

    _save(fig, "flow_pressure_plot.png")


def _convolution_schematic() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()

    eq = r"$P_{n+1}=P_d+z_0Q_{n+1}+\sum_{m=1}^{N_k-1} z_m Q_{n+1-m}$"
    ax.text(0.02, 0.90, "IMPEDANCE Runtime Equation", fontsize=13, weight="bold")
    ax.text(0.02, 0.80, eq, fontsize=15, color="#111827")

    left = FancyBboxPatch(
        (0.03, 0.40),
        0.50,
        0.30,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#ecfeff",
        ec="#0e7490",
        lw=1.5,
    )
    ax.add_patch(left)
    ax.text(
        0.05,
        0.63,
        "Implicit term (z0*Q[n+1])",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.05,
        0.49,
        "Added directly to solver matrix.\n"
        "Lagged terms come from accepted-flow history.",
        fontsize=10.2,
    )

    right = FancyBboxPatch(
        (0.58, 0.26),
        0.38,
        0.50,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#fff7ed",
        ec="#b45309",
        lw=1.5,
    )
    ax.add_patch(right)
    ax.text(0.60, 0.70, "One-cycle ring buffer", fontsize=11, weight="bold")

    labels = ["Q[n]", "Q[n-1]", "Q[n-2]", "...", "Q[n-Np+1]"]
    y = 0.62
    for label in labels:
        ax.add_patch(
            FancyBboxPatch(
                (0.61, y - 0.04),
                0.29,
                0.065,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                fc="#ffffff",
                ec="#9a3412",
                lw=1.0,
            )
        )
        ax.text(0.64, y - 0.005, label, fontsize=10)
        y -= 0.09

    ax.annotate(
        "",
        xy=(0.58, 0.52),
        xytext=(0.53, 0.52),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#7c2d12"},
    )

    _save(fig, "convolution_ringbuffer_schematic.png")


def _coupling_state_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()
    ax.text(0.02, 0.92, "3D-0D Coupling Safety (Trial / Rollback / Commit)", fontsize=13, weight="bold")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            fc=color,
            ec="#334155",
            lw=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.065, title, fontsize=10.8, weight="bold")
        ax.text(x + 0.02, y + h - 0.14, body, fontsize=9.4, va="top")

    box(
        0.04,
        0.53,
        0.26,
        0.30,
        "1) Snapshot",
        "return_y/return_ydot\nstore accepted\npersistent state.",
        "#ecfeff",
    )
    box(
        0.36,
        0.53,
        0.26,
        0.30,
        "2) Trial reset",
        "update_state restores\nsnapshot, then applies\ntrial y/ydot.",
        "#eef2ff",
    )
    box(
        0.68,
        0.53,
        0.26,
        0.30,
        "3) Trial run",
        "run_simulation advances\nwithout committing\nnew history yet.",
        "#f0fdf4",
    )
    box(
        0.28,
        0.13,
        0.44,
        0.26,
        "4) Accept / Commit",
        "On accepted step, snapshot updates.\nRepeated retries from the same snapshot\nstay deterministic (test_interface/test_04).",
        "#fff7ed",
    )

    arrows = [
        ((0.30, 0.68), (0.36, 0.68)),
        ((0.62, 0.68), (0.68, 0.68)),
        ((0.81, 0.53), (0.63, 0.39)),
        ((0.42, 0.53), (0.46, 0.39)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#1f2937"})

    _save(fig, "coupling_state_flow.png")


def _iteration_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()
    ax.text(0.02, 0.92, "3D-0D Iteration Pipeline", fontsize=13, weight="bold")

    def stage(y: float, title: str, body: str) -> None:
        patch = FancyBboxPatch(
            (0.05, y),
            0.90,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            fc="#f8fafc",
            ec="#64748b",
            lw=1.1,
        )
        ax.add_patch(patch)
        ax.text(0.08, y + 0.085, title, fontsize=11.3, weight="bold")
        ax.text(0.08, y + 0.03, body, fontsize=10.2)

    stage(0.72, "1) Tune impedance BC", "Generate tuned 0D configuration")
    stage(0.54, "2) Run preop 3D", "Launch simulation job")
    stage(0.36, "3) Compute metrics", "Extract pressure and flow split")
    stage(0.18, "4) Gate decision", "Converged -> postop; not close -> retune")

    for y in [0.72, 0.54, 0.36]:
        ax.annotate("", xy=(0.50, y - 0.01), xytext=(0.50, y - 0.04), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})

    _save(fig, "iteration_pipeline.png")


def _architecture_context_simple() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()
    ax.text(0.02, 0.93, "Where IMPEDANCE Runs", fontsize=13, weight="bold")

    for y, label in [(0.60, "Full 0D"), (0.26, "3D-0D Coupling")]:
        lane = FancyBboxPatch(
            (0.02, y),
            0.96,
            0.25,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            fc="#f8fafc",
            ec="#cbd5e1",
            lw=1.0,
        )
        ax.add_patch(lane)
        ax.text(0.04, y + 0.2, label, fontsize=11.2, weight="bold")

    def box(x: float, y: float, text: str, w: float = 0.20) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            0.1,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            fc="#ffffff",
            ec="#94a3b8",
            lw=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + 0.012, y + 0.056, text, fontsize=9.6, va="center")

    box(0.06, 0.67, "JSON input")
    box(0.31, 0.67, "Model init")
    box(0.56, 0.67, "IMPEDANCE")
    box(0.81, 0.67, "Outputs", w=0.13)

    box(0.06, 0.33, "update_state")
    box(0.31, 0.33, "run_sim")
    box(0.56, 0.33, "rollback-safe\nstate", w=0.22)
    box(0.82, 0.33, "accept", w=0.12)

    arrows = [
        ((0.26, 0.72), (0.31, 0.72)),
        ((0.51, 0.72), (0.56, 0.72)),
        ((0.76, 0.72), (0.81, 0.72)),
        ((0.26, 0.38), (0.31, 0.38)),
        ((0.51, 0.38), (0.56, 0.38)),
        ((0.78, 0.38), (0.82, 0.38)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#475569"})

    _save(fig, "architecture_context_simple.png")


def _main_diff_touchpoints() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()

    ax.text(0.02, 0.92, "Change Footprint vs master", fontsize=13, weight="bold")
    ax.text(
        0.02,
        0.85,
        "Evidence source: impedance-bc commit set (bb926, 82ba, 161870) after merge with origin/master",
        fontsize=9.4,
        color="#334155",
    )

    left = FancyBboxPatch(
        (0.03, 0.14),
        0.44,
        0.66,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#ecfeff",
        ec="#0e7490",
        lw=1.4,
    )
    right = FancyBboxPatch(
        (0.53, 0.14),
        0.44,
        0.66,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#fff7ed",
        ec="#b45309",
        lw=1.4,
    )
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.05, 0.75, "IMPEDANCE-core changes", fontsize=11.2, weight="bold")
    ax.text(
        0.05,
        0.71,
        "- src/model/ImpedanceBC.{h,cpp}\n"
        "- src/solve/SimulationParameters.cpp\n"
        "- impedance test fixtures",
        fontsize=10.0,
        va="top",
    )

    ax.text(0.55, 0.75, "Cross-cutting support hooks", fontsize=11.2, weight="bold")
    ax.text(
        0.55,
        0.71,
        "- Block: persistent-state virtual hooks\n"
        "- Model: get/set/clear persistent state\n"
        "- Integrator: accept_timestep callback + dt handoff\n"
        "- Interface: snapshot/restore for retry safety\n"
        "- Solver: clear persistent state after steady init",
        fontsize=9.9,
        va="top",
    )

    ax.annotate(
        "No broad solver architecture rewrite",
        xy=(0.50, 0.12),
        xytext=(0.50, 0.06),
        ha="center",
        fontsize=10.6,
        weight="bold",
    )

    _save(fig, "main_diff_touchpoints.png")


def _impedance_vs_rcr_matrix() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()
    ax.text(0.02, 0.92, "IMPEDANCE vs RCR (Quick Selection)", fontsize=13, weight="bold")

    headers = ["Dimension", "IMPEDANCE", "RCR"]
    rows = [
        ("History", "One-cycle flow memory", "Compact internal state"),
        ("Behavior", "Richer waveform effects", "Simpler approximation"),
        ("Compute cost", "Higher", "Lower"),
        ("Best use", "Higher-fidelity studies", "Fast baseline sweeps"),
    ]
    x_cols = [0.04, 0.35, 0.66]
    col_w = [0.28, 0.28, 0.28]
    y0 = 0.74
    rh = 0.14

    for i, h in enumerate(headers):
        patch = FancyBboxPatch(
            (x_cols[i], y0),
            col_w[i],
            0.1,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            fc="#e2e8f0",
            ec="#94a3b8",
            lw=1.0,
        )
        ax.add_patch(patch)
        ax.text(x_cols[i] + 0.01, y0 + 0.056, h, fontsize=10.7, weight="bold", va="center")

    for ridx, (dim, imp, rcr) in enumerate(rows):
        y = y0 - (ridx + 1) * rh
        for i, val in enumerate([dim, imp, rcr]):
            patch = FancyBboxPatch(
                (x_cols[i], y),
                col_w[i],
                0.11,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                fc="#ffffff",
                ec="#cbd5e1",
                lw=0.9,
            )
            ax.add_patch(patch)
            ax.text(x_cols[i] + 0.01, y + 0.058, val, fontsize=10.1, va="center")

    _save(fig, "impedance_vs_rcr_matrix.png")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _style()
    _kernel_plot()
    _flow_pressure_plot()
    _convolution_schematic()
    _coupling_state_diagram()
    _iteration_pipeline_diagram()
    _architecture_context_simple()
    _main_diff_touchpoints()
    _impedance_vs_rcr_matrix()
    print(f"Wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
