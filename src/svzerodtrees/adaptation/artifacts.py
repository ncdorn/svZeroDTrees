"""Artifact writers for structured-tree adaptation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_reduced_pa_flow_split_convergence_artifacts(
    *,
    output_dir: str | Path,
    flow_split_history: Sequence[Mapping[str, float]],
    preop_rpa_split: float,
    postop_rpa_split: float,
    target_rpa_split: float,
    final_rpa_split: float,
) -> dict[str, str]:
    """Write reduced-PA flow split convergence artifacts for M1 adaptation."""

    rows: list[dict[str, float]] = []
    for entry in flow_split_history:
        time_s = float(entry["t"])
        rpa_split = float(entry["rpa_split"])
        rows.append(
            {
                "time_s": time_s,
                "rpa_split": rpa_split,
                "lpa_split": 1.0 - rpa_split,
            }
        )

    if not rows:
        raise ValueError("flow_split_history must contain at least one accepted-step sample")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "reduced_pa_flow_split_convergence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "rpa_split", "lpa_split"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = output_root / "reduced_pa_flow_split_convergence.png"
    time_values = [row["time_s"] for row in rows]
    rpa_values = [row["rpa_split"] for row in rows]
    final_time = time_values[-1]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(time_values, rpa_values, color="#1f77b4", linewidth=2.0, label="RPA split")
    ax.axhline(
        float(preop_rpa_split),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="Preop",
    )
    ax.axhline(
        float(postop_rpa_split),
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label="Postop initial",
    )
    ax.axhline(
        float(target_rpa_split),
        color="#d62728",
        linestyle=":",
        linewidth=1.7,
        label="Clinical target",
    )
    ax.scatter(
        [final_time],
        [float(final_rpa_split)],
        color="#111111",
        s=42,
        zorder=5,
        label="Adapted final",
    )
    ax.set_xlabel("Solver time (s)")
    ax.set_ylabel("RPA flow split")
    ax.set_title("Reduced-PA Adaptation Flow Split Convergence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return {
        "flow_split_convergence_csv": str(csv_path),
        "flow_split_convergence_png": str(png_path),
    }
