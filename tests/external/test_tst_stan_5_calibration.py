from __future__ import annotations

import json
from pathlib import Path

import pytest

from svzerodtrees.calibration.workflow import assemble_calibration_payload
from svzerodtrees.config import (
    CalibrationConfig,
    CalibrationDataSourceConfig,
    CalibrationParameterSelectionConfig,
    CalibrationParametersConfig,
)


@pytest.mark.external
def test_tst_stan_5_sampling_uses_interior_endpoints_and_explicit_exclusions():
    artifact_dir = (
        Path(__file__).parents[2] / "tmp/tst-stan-5-iter03-centerline-timeseries"
    )
    required = {
        "baseline": artifact_dir / "baseline_0d_c0.json",
        "centerline": artifact_dir / "centerlines.vtp",
        "mapped": artifact_dir / "centerline_timeseries_last_cycle.vtp",
        "metadata": artifact_dir / "centerline_timeseries_last_cycle_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        pytest.skip("local tst-stan-5 calibration artifacts are absent: " + ", ".join(missing))
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    if not metadata.get("frame_indices") or not metadata.get("timestamps_s"):
        pytest.skip(
            "local tst-stan-5 metadata predates the Stage 1 timeseries contract; "
            "regenerate centerline_timeseries_last_cycle_metadata.json"
        )

    assembly = assemble_calibration_payload(
        zerod_config_path=str(required["baseline"]),
        calibration=CalibrationConfig(
            data_source=CalibrationDataSourceConfig(
                mapped_centerline_result=str(required["mapped"]),
                metadata_json=str(required["metadata"]),
                centerline=str(required["centerline"]),
                flow_array="velocity",
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
        ),
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
