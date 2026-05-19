from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import pytest

from svzerodtrees.post_processing.pulmonary_threed_suite import (
    _select_systolic_frame_from_artifacts,
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


def test_select_systolic_frame_from_artifacts_uses_last_cycle_and_earliest_tie(tmp_path: Path):
    pressure_csv = tmp_path / "mpa_pressure_vs_time.csv"
    pressure_csv.write_text(
        "\n".join(
            [
                "timestep_id,time_s,mpa_pressure_mmhg",
                "1,0.0,10.0",
                "2,0.4,25.0",
                "3,0.8,25.0",
                "4,1.2,30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    frames_csv = tmp_path / "frames.csv"
    frames_csv.write_text(
        "\n".join(
            [
                "timestep_id,path,time_s",
                "1,/tmp/result_0001.vtu,0.0",
                "2,/tmp/result_0002.vtu,0.4",
                "3,/tmp/result_0003.vtu,0.8",
                "4,/tmp/result_0004.vtu,1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected, metadata = _select_systolic_frame_from_artifacts(
        pressure_csv=pressure_csv,
        frames_csv=frames_csv,
        cycle_duration_s=0.8,
    )

    assert selected["timestep_id"].tolist() == [2]
    assert selected["path"].tolist() == ["/tmp/result_0002.vtu"]
    assert metadata["selection_mode"] == "max_mpa_pressure_last_cycle"
    assert metadata["selection_window_start_s"] == pytest.approx(0.4)
    assert metadata["selection_window_end_s"] == pytest.approx(1.2)
    assert metadata["selected_timestep_id"] == 2
    assert metadata["selected_time_s"] == pytest.approx(0.4)
    assert metadata["selected_pressure_mmhg"] == pytest.approx(25.0)
    assert metadata["selected_frame_path"] == "/tmp/result_0002.vtu"
    assert metadata["tie_break_policy"] == "earliest_timestep_id"
    assert metadata["available_frame_count"] == 2


def test_select_systolic_frame_from_artifacts_errors_without_finite_last_cycle_pressure(tmp_path: Path):
    pressure_csv = tmp_path / "mpa_pressure_vs_time.csv"
    pressure_csv.write_text(
        "\n".join(
            [
                "timestep_id,time_s,mpa_pressure_mmhg",
                "1,0.0,10.0",
                "2,0.4,NaN",
                "3,0.8,NaN",
                "4,1.2,30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    frames_csv = tmp_path / "frames.csv"
    frames_csv.write_text(
        "\n".join(
            [
                "timestep_id,path,time_s",
                "1,/tmp/result_0001.vtu,0.0",
                "2,/tmp/result_0002.vtu,0.4",
                "3,/tmp/result_0003.vtu,0.8",
                "4,/tmp/result_0004.vtu,1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no overlapping last-cycle frames"):
        _select_systolic_frame_from_artifacts(
            pressure_csv=pressure_csv,
            frames_csv=frames_csv,
            cycle_duration_s=0.8,
        )


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
        assert kwargs["keep_intermediate_centerlines"] is True
        resistance_dir = Path(kwargs["output_dir"])
        resistance_dir.mkdir(parents=True, exist_ok=True)
        summary = resistance_dir / "branch_resistance_summary.csv"
        ranked = resistance_dir / "ranked_stent_candidates.csv"
        vtp = resistance_dir / "resistance_map_mean.vtp"
        metadata = resistance_dir / "resistance_map_metadata.json"
        intermediate_dir = resistance_dir / "intermediate_centerlines"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        mapped_frame = intermediate_dir / "0000_result_0002_centerline.vtp"
        mapped_frame.write_text("<vtk/>", encoding="utf-8")
        pd.DataFrame([{"branch_id": 1, "resistance_mean": 3.0}]).to_csv(summary, index=False)
        pd.DataFrame([{"branch_id": 1, "rank": 1}]).to_csv(ranked, index=False)
        pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).save(vtp)
        metadata.write_text(
            json.dumps(
                {
                    "selected_frames": [
                        {
                            "timestep_id": 2,
                            "source_frame_path": str(sim_dir / "result_0002.vtu"),
                            "path": str(mapped_frame),
                        }
                    ],
                    "keep_intermediate_centerlines": True,
                    "intermediate_dir": str(intermediate_dir),
                }
            ),
            encoding="utf-8",
        )
        return {
            "resistance_map": str(vtp),
            "summary_csv": str(summary),
            "ranked_csv": str(ranked),
            "metadata_json": str(metadata),
            "intermediate_dir": str(intermediate_dir),
        }

    def fake_compute_selected_frames(**kwargs):
        assert kwargs["metric_suffix"] == "systolic"
        assert kwargs["selection_policy"] == "max_mpa_pressure_last_cycle"
        assert kwargs["selected_frames"]["timestep_id"].tolist() == [2]
        assert kwargs["selected_frames"]["source_frame_path"].tolist() == [str(sim_dir / "result_0002.vtu")]
        assert kwargs["selected_frames"]["mapped_path"].iloc[0].endswith("0000_result_0002_centerline.vtp")
        resistance_dir = Path(kwargs["output_dir"])
        resistance_dir.mkdir(parents=True, exist_ok=True)
        summary = resistance_dir / "branch_resistance_summary_systolic.csv"
        ranked = resistance_dir / "ranked_stent_candidates_systolic.csv"
        vtp = resistance_dir / "resistance_map_systolic.vtp"
        metadata = resistance_dir / "resistance_map_systolic_metadata.json"
        pd.DataFrame([{"branch_id": 1, "resistance_systolic": 5.0}]).to_csv(summary, index=False)
        pd.DataFrame([{"branch_id": 1, "rank": 1}]).to_csv(ranked, index=False)
        line = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        line.point_data["branch_resistance_systolic"] = np.array([5.0, 5.0], dtype=float)
        line.save(vtp)
        metadata.write_text(json.dumps({"selected_frames": []}), encoding="utf-8")
        return {
            "resistance_map": str(vtp),
            "summary_csv": str(summary),
            "ranked_csv": str(ranked),
            "metadata_json": str(metadata),
        }

    render_calls: list[dict[str, object]] = []

    def fake_render_png(**kwargs):
        render_calls.append(kwargs)
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
        "svzerodtrees.post_processing.pulmonary_threed_suite._compute_pulmonary_resistance_map_for_selected_frames",
        fake_compute_selected_frames,
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
    assert (output_dir / "resistance_map_systolic.vtp").exists()
    assert (output_dir / "branch_resistance_summary_systolic.csv").exists()
    assert (output_dir / "ranked_stent_candidates_systolic.csv").exists()
    assert (output_dir / "resistance_map_systolic.png").exists()
    assert Path(result["metadata_json"]).exists()
    assert result["steps"]["resistance_map_systolic"]["status"] == "completed"
    assert "resistance_map_systolic" in result
    assert result["resistance_map"]["intermediate_dir"] is None
    assert not (output_dir / "resistance_map" / "intermediate_centerlines").exists()

    mean_metadata = json.loads((output_dir / "resistance_map" / "resistance_map_metadata.json").read_text(encoding="utf-8"))
    assert mean_metadata["keep_intermediate_centerlines"] is False
    assert mean_metadata["intermediate_dir"] is None
    top_level_mean_metadata = json.loads((output_dir / "resistance_map_metadata.json").read_text(encoding="utf-8"))
    assert top_level_mean_metadata["keep_intermediate_centerlines"] is False
    assert top_level_mean_metadata["intermediate_dir"] is None

    frames = pd.read_csv(output_dir / "frames.csv")
    assert list(frames.columns) == ["timestep_id", "path", "time_s"]
    assert render_calls[0].get("scalar_name", "branch_resistance_mean") == "branch_resistance_mean"
    assert render_calls[1]["scalar_name"] == "branch_resistance_systolic"

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

    def fake_compute_selected_frames(**kwargs):
        assert kwargs["metric_suffix"] == "systolic"
        resistance_dir = Path(kwargs["output_dir"])
        resistance_dir.mkdir(parents=True, exist_ok=True)
        summary = resistance_dir / "branch_resistance_summary_systolic.csv"
        ranked = resistance_dir / "ranked_stent_candidates_systolic.csv"
        vtp = resistance_dir / "resistance_map_systolic.vtp"
        metadata = resistance_dir / "resistance_map_systolic_metadata.json"
        pd.DataFrame([{"branch_id": 2, "resistance_systolic": 6.0}]).to_csv(summary, index=False)
        pd.DataFrame([{"branch_id": 2, "rank": 1}]).to_csv(ranked, index=False)
        line = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        line.point_data["branch_resistance_systolic"] = np.array([6.0, 6.0], dtype=float)
        line.save(vtp)
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
        "svzerodtrees.post_processing.pulmonary_threed_suite._compute_pulmonary_resistance_map_for_selected_frames",
        fake_compute_selected_frames,
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
    assert (output_dir / "resistance_map_systolic.png").exists()
    assert result["steps"]["pressure"]["status"] == "completed"
    assert result["steps"]["resistance_map"]["status"] == "completed"
    assert result["steps"]["resistance_map_systolic"]["status"] == "completed"
