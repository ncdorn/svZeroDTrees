from __future__ import annotations

import csv
from pathlib import Path

import pytest

from svzerodtrees.adaptation.artifacts import (
    write_reduced_pa_flow_split_convergence_artifacts,
)


def test_write_reduced_pa_flow_split_convergence_artifacts_writes_csv_and_plot(tmp_path):
    artifacts = write_reduced_pa_flow_split_convergence_artifacts(
        output_dir=tmp_path,
        flow_split_history=[
            {"t": 0.0, "rpa_split": 0.56},
            {"t": 120.0, "rpa_split": 0.55},
            {"t": 240.0, "rpa_split": 0.54},
        ],
        preop_rpa_split=0.67,
        postop_rpa_split=0.56,
        target_rpa_split=0.58,
        final_rpa_split=0.54,
    )

    csv_path = Path(artifacts["flow_split_convergence_csv"])
    png_path = Path(artifacts["flow_split_convergence_png"])
    assert csv_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert list(rows[0].keys()) == ["time_s", "rpa_split", "lpa_split"]
    assert len(rows) == 3
    assert float(rows[0]["time_s"]) == 0.0
    assert float(rows[0]["rpa_split"]) == 0.56
    assert float(rows[0]["lpa_split"]) == pytest.approx(0.44)
    assert float(rows[-1]["time_s"]) == 240.0
    assert float(rows[-1]["rpa_split"]) == 0.54
    assert float(rows[-1]["lpa_split"]) == pytest.approx(0.46)
