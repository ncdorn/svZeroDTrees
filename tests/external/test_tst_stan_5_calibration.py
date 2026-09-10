from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from svzerodtrees._pysvzerod import require_calibration_capabilities
from svzerodtrees.calibration.workflow import (
    assemble_calibration_payload,
    calibrate_0d_from_mapped_centerline,
)
from svzerodtrees.config import (
    CalibrationConfig,
    CalibrationDataSourceConfig,
    CalibrationParameterSelectionConfig,
    CalibrationParametersConfig,
)


def _require_tst_stan_5_artifacts() -> dict[str, Path]:
    if os.environ.get("SVZERODTREES_RUN_REAL_CASE") != "1":
        pytest.skip(
            "opt-in real-case regression; set SVZERODTREES_RUN_REAL_CASE=1 to run"
        )

    artifact_dir = (
        Path(__file__).parents[2] / "tmp/tst-stan-5-iter03-centerline-timeseries"
    )
    paths = {
        "baseline": artifact_dir / "baseline_0d_c0.json",
        "centerline": artifact_dir / "centerlines.vtp",
        "mapped": artifact_dir / "centerline_timeseries_last_cycle.vtp",
        "metadata": artifact_dir / "centerline_timeseries_last_cycle_metadata.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip(
            "local tst-stan-5 calibration artifacts are absent: " + ", ".join(missing)
        )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not metadata.get("frame_indices") or not metadata.get("timestamps_s"):
        pytest.skip(
            "local tst-stan-5 metadata predates the timeseries contract; "
            "regenerate centerline_timeseries_last_cycle_metadata.json"
        )
    return paths


def _tst_stan_5_calibration(paths: dict[str, Path]) -> CalibrationConfig:
    return CalibrationConfig(
        data_source=CalibrationDataSourceConfig(
            mapped_centerline_result=str(paths["mapped"]),
            metadata_json=str(paths["metadata"]),
            centerline=str(paths["centerline"]),
            flow_array="velocity",
            # svSlicer writes integrated flow under this legacy array name.
            flow_observation_type="flow",
        ),
        parameters=CalibrationParametersConfig(
            vessels=CalibrationParameterSelectionConfig(
                default=["R_poiseuille"],
                overrides={
                    "branch15_seg0": [],
                    "branch27_seg0": [],
                    "branch28_seg0": [],
                    "branch31_seg0": [],
                },
            ),
            junctions=CalibrationParameterSelectionConfig(default=[]),
        ),
    )


@pytest.mark.external
def test_tst_stan_5_sampling_uses_interior_endpoints_and_explicit_exclusions():
    required = _require_tst_stan_5_artifacts()

    assembly = assemble_calibration_payload(
        zerod_config_path=str(required["baseline"]),
        calibration=_tst_stan_5_calibration(required),
    )

    assert assembly.observation_count == 100
    root = assembly.interface_sampling["branch0_seg0:upstream"]
    assert root["interface_kind"] == "external_upstream"
    assert root["quality_status"] == "qualified_interior"
    assert root["selected_path"] > root["requested_path"]
    assert root["usable_sample_count"] == 4
    assert "branch27_seg0" in assembly.excluded_blocks

    for sample in assembly.interface_sampling.values():
        if sample["quality_status"] == "qualified_interior":
            assert sample["selected_path"] != sample["requested_path"]


@pytest.mark.external
def test_tst_stan_5_calibration_end_to_end(tmp_path):
    paths = _require_tst_stan_5_artifacts()
    try:
        require_calibration_capabilities()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"compatible pysvzerod is unavailable for the real-case check: {exc}")

    calibration = _tst_stan_5_calibration(paths)
    assembly = assemble_calibration_payload(
        zerod_config_path=str(paths["baseline"]),
        calibration=calibration,
    )
    repeat = assemble_calibration_payload(
        zerod_config_path=str(paths["baseline"]),
        calibration=calibration,
    )

    assert assembly.observation_count == 100
    assert assembly.observation_count == repeat.observation_count
    assert assembly.observation_qc == repeat.observation_qc
    assert calibration.data_source.flow_observation_type == "flow"
    assert calibration.data_source.area_array is None
    assert assembly.solver_payload["vessels"][0]["calibrate"] == ["R_poiseuille"]
    assert "branch27_seg0" in assembly.excluded_blocks

    output_path = tmp_path / "calibrated_0d.json"
    result = calibrate_0d_from_mapped_centerline(
        zerod_config_path=str(paths["baseline"]),
        output_config_path=str(output_path),
        calibration=calibration,
    )

    assert result["status"] == "ok"
    summary = json.loads(
        (tmp_path / "calibration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["observation_count"] == 100
    assert summary["calibration_confirmation"]["converged"] is True
    assert summary["replay_stability"]["status"] == "pass"
    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert all("calibrate" not in vessel for vessel in published["vessels"])
    assert all(
        junction["junction_type"] != "internal_junction"
        for junction in published["junctions"]
    )

    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    baseline_by_name = {
        vessel["vessel_name"]: vessel for vessel in baseline["vessels"]
    }
    for vessel in published["vessels"]:
        original_values = baseline_by_name[vessel["vessel_name"]][
            "zero_d_element_values"
        ]
        published_values = vessel["zero_d_element_values"]
        for parameter in ("C", "L", "stenosis_coefficient"):
            assert published_values[parameter] == original_values[parameter]
