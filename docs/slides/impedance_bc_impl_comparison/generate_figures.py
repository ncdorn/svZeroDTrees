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
SVZT_AGENT = ROOT.parent / "svzt-agent"
OUTDIR = Path(__file__).resolve().parent / "assets" / "generated"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
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

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(range(len(z)), z, color="#005f73", lw=2.2)
    ax.set_title("Impedance Kernel from pulsatileFlow_R_impedance.json")
    ax.set_xlabel("Kernel index m")
    ax.set_ylabel("z[m] (Pa·s/m^3)")
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

    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax1.plot(one_cycle["t_cycle"], one_cycle["flow_out"], color="#0a9396", lw=2)
    ax1.set_xlabel("Time in cycle (s)")
    ax1.set_ylabel("Flow out (mL/s)")
    ax1.set_title("Last-Cycle Outlet Flow and Pressure (Impedance Fixture)")

    ax2 = ax1.twinx()
    ax2.plot(
        one_cycle["t_cycle"],
        one_cycle["pressure_out_mmhg"],
        color="#bb3e03",
        lw=1.9,
    )
    ax2.set_ylabel("Outlet pressure (mmHg)")

    _save(fig, "flow_pressure_plot.png")


def _convolution_schematic() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_axis_off()

    eq = r"$P_{n+1}=P_d+z_0Q_{n+1}+\sum_{m=1}^{N_k-1} z_m Q_{n+1-m}$"
    ax.text(0.02, 0.88, "Olufsen Discrete Convolution", fontsize=12, weight="bold")
    ax.text(0.02, 0.76, eq, fontsize=14, color="#001219")

    main_box = FancyBboxPatch(
        (0.02, 0.38),
        0.53,
        0.27,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#e9f5f5",
        ec="#0a9396",
        lw=1.4,
    )
    ax.add_patch(main_box)
    ax.text(
        0.04,
        0.56,
        "Implicit term: z0*Q(n+1)\n"
        "Added directly to system matrix (F).\n"
        "Lagged terms use accepted flow history.",
        fontsize=9.6,
        va="center",
    )

    ring_box = FancyBboxPatch(
        (0.61, 0.23),
        0.36,
        0.58,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="#fff5eb",
        ec="#ca6702",
        lw=1.4,
    )
    ax.add_patch(ring_box)
    ax.text(0.63, 0.75, "One-cycle ring buffer", fontsize=10.5, weight="bold")

    y = 0.66
    labels = ["head -> Q(n)", "Q(n-1)", "Q(n-2)", "...", "Q(n-Np+1)"]
    for label in labels:
        ax.add_patch(
            FancyBboxPatch(
                (0.64, y - 0.04),
                0.29,
                0.06,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                fc="#fff",
                ec="#bb3e03",
                lw=1.0,
            )
        )
        ax.text(0.655, y - 0.01, label, fontsize=9)
        y -= 0.10

    ax.annotate(
        "",
        xy=(0.61, 0.49),
        xytext=(0.55, 0.49),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#7c2d12"},
    )
    ax.text(0.50, 0.53, "lagged\nlookup", fontsize=9, ha="center", color="#7c2d12")
    _save(fig, "convolution_ringbuffer_schematic.png")


def _coupling_state_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_axis_off()

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
        ax.text(x + 0.02, y + h - 0.06, title, fontsize=10.5, weight="bold")
        ax.text(x + 0.02, y + h - 0.11, body, fontsize=8.8, va="top")

    box(
        0.03,
        0.55,
        0.28,
        0.35,
        "Committed Snapshot",
        "Stored by return_y/return_ydot\nfrom accepted 0D state.",
        "#ecfeff",
    )
    box(
        0.36,
        0.55,
        0.28,
        0.35,
        "Trial Update",
        "update_state restores committed\npersistent memory, then applies\nincoming y/ydot for trial step.",
        "#eef2ff",
    )
    box(
        0.69,
        0.55,
        0.28,
        0.35,
        "run_simulation",
        "Integrator advances.\nImpedanceBC accepts timestep\nonly after solver accept.",
        "#f0fdf4",
    )
    box(
        0.19,
        0.08,
        0.28,
        0.32,
        "Rollback-Safe Retry",
        "Repeated trials from same\ncommitted state are deterministic\n(test_interface/test_04).",
        "#fff7ed",
    )
    box(
        0.53,
        0.08,
        0.28,
        0.32,
        "Commit",
        "Caller marks accepted state via\nreturn_y/return_ydot,\nupdating committed snapshot.",
        "#fefce8",
    )

    arrows = [
        ((0.31, 0.73), (0.36, 0.73)),
        ((0.64, 0.73), (0.69, 0.73)),
        ((0.83, 0.55), (0.67, 0.40)),
        ((0.36, 0.55), (0.33, 0.40)),
        ((0.47, 0.24), (0.53, 0.24)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#1f2937"})

    _save(fig, "coupling_state_flow.png")


def _iteration_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_axis_off()
    ax.text(0.02, 0.93, "3D-0D Iteration Pipeline (svzt-agent + svZeroDTrees)", fontsize=12, weight="bold")

    stages = [
        ("1. Run impedance tuning", "run_impedance_tuning_for_iteration\n-> tuned_zerod_config + artifacts"),
        ("2. Submit preop 3D", "_prepare_and_submit_stage(preop)\nwait for terminal state"),
        ("3. Extract metrics", "compute_centerline_mpa_metrics\ncompute_flow_split_metrics"),
        ("4. Gate decision", "evaluate_iteration_gate\nconverged / not_close / needs_review"),
        ("5A. Not close", "generate_reduced_pa_from_iteration\nseed next iteration"),
        ("5B. Converged", "submit postop with tuned 0D BCs"),
    ]

    y = 0.78
    for idx, (title, body) in enumerate(stages):
        x = 0.03 if idx < 4 else (0.03 if idx == 4 else 0.52)
        w = 0.43 if idx >= 4 else 0.94
        if idx < 4:
            patch = FancyBboxPatch((x, y), w, 0.11, boxstyle="round,pad=0.015,rounding_size=0.01", fc="#f8fafc", ec="#475569", lw=1.0)
            ax.add_patch(patch)
            ax.text(x + 0.015, y + 0.07, title, fontsize=10, weight="bold")
            ax.text(x + 0.015, y + 0.02, body, fontsize=8.7, va="bottom")
            if idx < 3:
                ax.annotate("", xy=(0.5, y - 0.01), xytext=(0.5, y - 0.05), arrowprops={"arrowstyle": "->", "lw": 1.5})
            if idx == 3:
                ax.annotate("", xy=(0.25, 0.34), xytext=(0.45, 0.42), arrowprops={"arrowstyle": "->", "lw": 1.5})
                ax.annotate("", xy=(0.74, 0.34), xytext=(0.55, 0.42), arrowprops={"arrowstyle": "->", "lw": 1.5})
            y -= 0.15
        else:
            color = "#ecfeff" if idx == 4 else "#f0fdf4"
            patch = FancyBboxPatch((x, 0.2), w, 0.12, boxstyle="round,pad=0.015,rounding_size=0.01", fc=color, ec="#0f766e", lw=1.1)
            ax.add_patch(patch)
            ax.text(x + 0.015, 0.27, title, fontsize=10, weight="bold")
            ax.text(x + 0.015, 0.215, body, fontsize=8.6, va="bottom")

    _save(fig, "iteration_pipeline.png")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _style()
    _kernel_plot()
    _flow_pressure_plot()
    _convolution_schematic()
    _coupling_state_diagram()
    _iteration_pipeline_diagram()
    print(f"Wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
