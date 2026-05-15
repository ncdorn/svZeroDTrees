from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import pytest

from svzerodtrees.post_processing.pulmonary_threed_suite import (
    _clinical_targets_from_input,
    run_pulmonary_threed_postprocess_suite,
    write_mpa_pressure_timeseries_csv,
)


def _write_centerline(path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    poly = pv.PolyData(points)
    poly.lines = np.array([3, 0, 1, 2, 3, 2, 3, 4], dtype=np.int64)
    poly.save(path)


def _make_point_dataset(tag: str):
    class DummyMesh:
        def __init__(self, name: str):
            self.name = name

    return DummyMesh(tag)


def test_write_mpa_pressure_timeseries_csv_matches_expected_contract(monkeypatch, tmp_path: Path):
    centerline_path = tmp_path / "centerlines.vtp"
    _write_centerline(centerline_path)
    sim_dir = tmp_path / "preop"
    sim_dir.mkdir()
    (sim_dir / "svFSIplus.xml").write_text(
        "<root><Time_step_size>0.1</Time_step_size></root>",
        encoding="utf-8",
    )
    for name in ("result_0001.vtu", "result_0002.vtu"):
        (sim_dir / name).write_text("dummy", encoding="utf-8")

    real_read = pv.read

    def fake_read(path: str):
        path_obj = Path(path)
        if path_obj.suffix == ".vtp":
            return real_read(path)
        return _make_point_dataset(path_obj.stem)

    def fake_sample(centerline, mesh, pressure_field, already_mmhg):
        if mesh.name.endswith("0001"):
            return np.array([10.0, 12.0, 14.0, 20.0, 22.0], dtype=float)
        return np.array([11.0, 13.0, 15.0, 21.0, 23.0], dtype=float)

    monkeypatch.setattr("svzerodtrees.post_processing.pulmonary_threed_suite.pv.read", fake_read)
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite._sample_pressure_on_centerline",
        fake_sample,
    )

    output_csv = tmp_path / "mpa_pressure_vs_time.csv"
    result = write_mpa_pressure_timeseries_csv(
        simulation_dir=sim_dir,
        centerline=centerline_path,
        output_csv=output_csv,
    )

    df = pd.read_csv(output_csv)
    assert list(df.columns) == ["timestep_id", "time_s", "mpa_pressure_mmhg"]
    assert df["timestep_id"].tolist() == [1, 2]
    assert df["time_s"].tolist() == pytest.approx([0.1, 0.2])
    assert df["mpa_pressure_mmhg"].tolist() == pytest.approx([12.0, 13.0])
    assert result["bifurcation_id"] == 2


def test_run_pulmonary_threed_postprocess_suite_writes_expected_outputs(monkeypatch, tmp_path: Path):
    centerline_path = tmp_path / "centerlines.vtp"
    _write_centerline(centerline_path)
    sim_dir = tmp_path / "preop"
    sim_dir.mkdir()
    (sim_dir / "svFSIplus.xml").write_text(
        "<root><Time_step_size>0.2</Time_step_size></root>",
        encoding="utf-8",
    )
    for name in ("result_0001.vtu", "result_0002.vtu", "result_0003.vtu"):
        (sim_dir / name).write_text("dummy", encoding="utf-8")

    real_read = pv.read

    def fake_read(path: str):
        path_obj = Path(path)
        if path_obj.suffix == ".vtp":
            return real_read(path)
        return _make_point_dataset(path_obj.stem)

    def fake_sample(centerline, mesh, pressure_field, already_mmhg):
        step = int(mesh.name.split("_")[-1])
        base = float(step)
        return np.array([base, base + 2.0, base + 4.0, base + 10.0, base + 12.0], dtype=float)

    class DummySimulationDirectory:
        @classmethod
        def from_directory(cls, path: str):
            return cls()

        def flow_split(self, get_mean=False, verbose=False):
            assert get_mean is True
            return (
                {"upper": 30.0, "middle": 10.0, "lower": 0.0},
                {"upper": 40.0, "middle": 20.0, "lower": 0.0},
            )

    def fake_compute_pulmonary_resistance_map(**kwargs):
        assert "max_frames" not in kwargs
        assert kwargs["workers"] == "auto"
        resistance_dir = Path(kwargs["output_dir"])
        resistance_dir.mkdir(parents=True, exist_ok=True)
        summary = resistance_dir / "branch_resistance_summary.csv"
        ranked = resistance_dir / "ranked_stent_candidates.csv"
        vtp = resistance_dir / "resistance_map_mean.vtp"
        metadata = resistance_dir / "resistance_map_metadata.json"
        pd.DataFrame([{"branch_id": 1, "resistance_mean": 3.0}]).to_csv(summary, index=False)
        pd.DataFrame([{"branch_id": 1, "rank": 1}]).to_csv(ranked, index=False)
        pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).save(vtp)
        metadata.write_text(json.dumps({"selected_frames": []}), encoding="utf-8")
        return {
            "resistance_map": str(vtp),
            "summary_csv": str(summary),
            "ranked_csv": str(ranked),
            "metadata_json": str(metadata),
        }

    def fake_render_png(**kwargs):
        output = Path(kwargs["output_png"])
        output.write_bytes(b"png")
        return str(output)

    monkeypatch.setattr("svzerodtrees.post_processing.pulmonary_threed_suite.pv.read", fake_read)
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite._sample_pressure_on_centerline",
        fake_sample,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.SimulationDirectory",
        DummySimulationDirectory,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.compute_pulmonary_resistance_map",
        fake_compute_pulmonary_resistance_map,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.render_resistance_map_png",
        fake_render_png,
    )

    output_dir = tmp_path / "results" / "postprocess"
    result = run_pulmonary_threed_postprocess_suite(
        simulation_dir=sim_dir,
        output_dir=output_dir,
        centerline=centerline_path,
        stage="preop",
        svslicer_path="/tmp/svslicer",
        clinical_targets={"mpa_p": [20.0, 10.0, 15.0], "rpa_split": 0.6},
        cycle_duration_s=0.4,
        resistance_map_workers="auto",
    )

    assert (output_dir / "mpa_pressure_vs_time.csv").exists()
    assert (output_dir / "mpa_pressure_vs_time.png").exists()
    assert (output_dir / "flow_split_comparison.csv").exists()
    assert (output_dir / "flow_split_comparison.png").exists()
    assert (output_dir / "frames.csv").exists()
    assert (output_dir / "resistance_map_mean.vtp").exists()
    assert (output_dir / "branch_resistance_summary.csv").exists()
    assert (output_dir / "ranked_stent_candidates.csv").exists()
    assert (output_dir / "resistance_map_mean.png").exists()
    assert Path(result["metadata_json"]).exists()

    flow_split = pd.read_csv(output_dir / "flow_split_comparison.csv")
    assert flow_split["vessel"].tolist() == ["lpa", "rpa"]
    assert flow_split["simulated_split"].tolist() == pytest.approx([0.4, 0.6])


def test_clinical_targets_mapping_accepts_mpa_pressure_alias():
    targets = _clinical_targets_from_input(
        {"mpa_pressure": np.array([20.0, 10.0, 15.0]), "rpa_split": 0.6}
    )

    assert targets == {
        "mpa_sys": 20.0,
        "mpa_dia": 10.0,
        "mpa_mean": 15.0,
        "rpa_split": 0.6,
    }


def test_clinical_targets_mapping_accepts_normalized_metric_mapping():
    targets = _clinical_targets_from_input(
        {
            "mpa_sys": 20.0,
            "mpa_dia": 10.0,
            "mpa_mean": 15.0,
            "rpa_split": 0.6,
        }
    )

    assert targets == {
        "mpa_sys": 20.0,
        "mpa_dia": 10.0,
        "mpa_mean": 15.0,
        "rpa_split": 0.6,
    }


def test_run_pulmonary_threed_postprocess_suite_tolerates_invalid_overlay_targets(monkeypatch, tmp_path: Path):
    centerline_path = tmp_path / "centerlines.vtp"
    _write_centerline(centerline_path)
    sim_dir = tmp_path / "preop"
    sim_dir.mkdir()
    (sim_dir / "svFSIplus.xml").write_text(
        "<root><Time_step_size>0.2</Time_step_size></root>",
        encoding="utf-8",
    )
    for name in ("result_0001.vtu", "result_0002.vtu", "result_0003.vtu"):
        (sim_dir / name).write_text("dummy", encoding="utf-8")

    real_read = pv.read

    def fake_read(path: str):
        path_obj = Path(path)
        if path_obj.suffix == ".vtp":
            return real_read(path)
        return _make_point_dataset(path_obj.stem)

    def fake_sample(centerline, mesh, pressure_field, already_mmhg):
        step = int(mesh.name.split("_")[-1])
        base = float(step)
        return np.array([base, base + 2.0, base + 4.0, base + 10.0, base + 12.0], dtype=float)

    class DummySimulationDirectory:
        @classmethod
        def from_directory(cls, path: str):
            return cls()

        def flow_split(self, get_mean=False, verbose=False):
            return ({"left": 40.0}, {"right": 60.0})

    def fake_compute_pulmonary_resistance_map(**kwargs):
        assert "max_frames" not in kwargs
        assert kwargs["workers"] is None
        resistance_dir = Path(kwargs["output_dir"])
        resistance_dir.mkdir(parents=True, exist_ok=True)
        summary = resistance_dir / "branch_resistance_summary.csv"
        ranked = resistance_dir / "ranked_stent_candidates.csv"
        vtp = resistance_dir / "resistance_map_mean.vtp"
        metadata = resistance_dir / "resistance_map_metadata.json"
        pd.DataFrame([{"branch_id": 2, "resistance_mean": 4.0}]).to_csv(summary, index=False)
        pd.DataFrame([{"branch_id": 2, "rank": 1}]).to_csv(ranked, index=False)
        pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).save(vtp)
        metadata.write_text(json.dumps({"selected_frames": []}), encoding="utf-8")
        return {
            "resistance_map": str(vtp),
            "summary_csv": str(summary),
            "ranked_csv": str(ranked),
            "metadata_json": str(metadata),
        }

    def fake_render_png(**kwargs):
        output = Path(kwargs["output_png"])
        output.write_bytes(b"png")
        return str(output)

    monkeypatch.setattr("svzerodtrees.post_processing.pulmonary_threed_suite.pv.read", fake_read)
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite._sample_pressure_on_centerline",
        fake_sample,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.SimulationDirectory",
        DummySimulationDirectory,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.compute_pulmonary_resistance_map",
        fake_compute_pulmonary_resistance_map,
    )
    monkeypatch.setattr(
        "svzerodtrees.post_processing.pulmonary_threed_suite.render_resistance_map_png",
        fake_render_png,
    )

    output_dir = tmp_path / "results" / "postprocess"
    result = run_pulmonary_threed_postprocess_suite(
        simulation_dir=sim_dir,
        output_dir=output_dir,
        centerline=centerline_path,
        stage="preop",
        svslicer_path="/tmp/svslicer",
        clinical_targets={"mpa_pressure": [20.0, 10.0], "rpa_split": 0.6},
        cycle_duration_s=0.4,
    )

    assert result["status"] == "completed"
    assert result["clinical_targets_available"] is False
    assert any("clinical targets invalid" in warning for warning in result["warnings"])
    assert (output_dir / "mpa_pressure_vs_time.png").exists()
    assert (output_dir / "flow_split_comparison.png").exists()
    assert (output_dir / "resistance_map_mean.png").exists()
    assert result["steps"]["pressure"]["status"] == "completed"
    assert result["steps"]["resistance_map"]["status"] == "completed"
