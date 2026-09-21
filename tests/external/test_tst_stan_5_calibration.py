from __future__ import annotations

import json
import math
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
    CalibrationMPAPressureTargetConfig,
    CalibrationObservationQCConfig,
    CalibrationParameterSelectionConfig,
    CalibrationParametersConfig,
    CalibrationRPAFlowSplitTargetConfig,
    CalibrationTargetsConfig,
)


def _real_case_opted_in() -> bool:
    return os.environ.get("SVZERODTREES_RUN_REAL_CASE") == "1"


def _artifact_unavailable(message: str) -> None:
    if _real_case_opted_in():
        pytest.fail(
            "SVZERODTREES_RUN_REAL_CASE=1 was requested, but the external "
            f"tst-stan-5 regression cannot run: {message}"
        )
    pytest.skip(message)


def _require_tst_stan_5_artifacts() -> dict[str, Path]:
    if not _real_case_opted_in():
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
        _artifact_unavailable(
            "required artifacts are absent: " + ", ".join(missing)
        )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not metadata.get("frame_indices") or not metadata.get("timestamps_s"):
        _artifact_unavailable(
            "local tst-stan-5 metadata predates the timeseries contract; "
            "regenerate centerline_timeseries_last_cycle_metadata.json"
        )
    return paths


def _required_case_setting(environment_name: str) -> str:
    value = os.environ.get(environment_name, "").strip()
    if not value:
        pytest.fail(
            "SVZERODTREES_RUN_REAL_CASE=1 requires the case owner to provide "
            f"{environment_name}; do not infer tst-stan-5 anatomy or use an "
            "unapproved default"
        )
    return value


def _required_positive_float(environment_name: str) -> float:
    raw_value = _required_case_setting(environment_name)
    try:
        value = float(raw_value)
    except ValueError as exc:
        pytest.fail(f"{environment_name} must be a finite positive number: {exc}")
    if not math.isfinite(value) or value <= 0.0:
        pytest.fail(f"{environment_name} must be a finite positive number")
    return value


def _tst_stan_5_targets() -> CalibrationTargetsConfig:
    """Read case-owner target roles and tolerances without guessing them."""
    mpa_vessel = _required_case_setting("SVZERODTREES_TST_STAN_5_MPA_VESSEL")
    rpa_vessel = _required_case_setting("SVZERODTREES_TST_STAN_5_RPA_VESSEL")
    lpa_vessel = _required_case_setting("SVZERODTREES_TST_STAN_5_LPA_VESSEL")
    if len({mpa_vessel, rpa_vessel, lpa_vessel}) != 3:
        pytest.fail(
            "SVZERODTREES_TST_STAN_5_MPA_VESSEL, _RPA_VESSEL, and _LPA_VESSEL "
            "must identify three distinct explicit roles"
        )

    mpa_interface = _required_case_setting(
        "SVZERODTREES_TST_STAN_5_MPA_INTERFACE"
    ).lower()
    rpa_interface = _required_case_setting(
        "SVZERODTREES_TST_STAN_5_RPA_INTERFACE"
    ).lower()
    valid_interfaces = {
        "external_upstream",
        "external_downstream",
        "internal",
        "upstream",
        "downstream",
    }
    if mpa_interface not in valid_interfaces or rpa_interface not in valid_interfaces:
        pytest.fail(
            "SVZERODTREES_TST_STAN_5_MPA_INTERFACE and _RPA_INTERFACE must be "
            f"one of {sorted(valid_interfaces)}"
        )

    improvement_setting = _required_case_setting(
        "SVZERODTREES_TST_STAN_5_REQUIRE_IMPROVEMENT_OVER_BASELINE"
    ).lower()
    if improvement_setting not in {"true", "false"}:
        pytest.fail(
            "SVZERODTREES_TST_STAN_5_REQUIRE_IMPROVEMENT_OVER_BASELINE must be "
            "explicitly true or false"
        )

    return CalibrationTargetsConfig(
        mpa_pressure=CalibrationMPAPressureTargetConfig(
            vessel=mpa_vessel,
            interface=mpa_interface,
            weight=_required_positive_float("SVZERODTREES_TST_STAN_5_MPA_WEIGHT"),
            normalized_rms_tolerance=_required_positive_float(
                "SVZERODTREES_TST_STAN_5_MPA_NRMSE_TOLERANCE"
            ),
        ),
        rpa_flow_split=CalibrationRPAFlowSplitTargetConfig(
            rpa_vessel=rpa_vessel,
            lpa_vessel=lpa_vessel,
            interface=rpa_interface,
            weight=_required_positive_float("SVZERODTREES_TST_STAN_5_RPA_WEIGHT"),
            absolute_tolerance=_required_positive_float(
                "SVZERODTREES_TST_STAN_5_RPA_SPLIT_TOLERANCE"
            ),
        ),
        require_improvement_over_baseline=improvement_setting == "true",
    )


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
        observation_qc=CalibrationObservationQCConfig(enforcement="target_focused"),
        targets=_tst_stan_5_targets(),
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
    except Exception as exc:
        message = f"compatible pysvzerod is unavailable for the real-case check: {exc}"
        if _real_case_opted_in():
            pytest.fail(
                "SVZERODTREES_RUN_REAL_CASE=1 was requested, but the configured "
                f"solver is unavailable or incompatible: {message}"
            )
        pytest.skip(message)

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
    assert assembly.observation_qc["enforcement"] == "target_focused"
    assert assembly.observation_qc["status"] == "pass"
    assert all(assembly.observation_qc["fatal_checks"].values())
    assert all(
        assembly.observation_qc["severity"][name] == "advisory"
        for name in assembly.observation_qc["advisory_checks"]
    )

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
    assert summary["calibration_confirmation"]["inactive_parameters_preserved"] is True
    assert summary["replay_stability"]["status"] == "pass"
    replay = summary["replay_stability"]
    accepted_cycle = replay["accepted_final_cycle"]["cycle"]
    assert accepted_cycle >= replay["validation_settings"]["replay_minimum_cycles"]
    assert accepted_cycle <= replay["validation_settings"]["replay_maximum_cycles"]
    assert replay["validation_settings"]["replay_maximum_cycles"] == (
        calibration.solver.replay_maximum_cycles
    )
    assert summary["observation_qc"]["enforcement"] == "target_focused"
    assert summary["observation_qc"]["status"] == "pass"
    assert all(summary["observation_qc"]["fatal_checks"].values())
    assert all(
        summary["observation_qc"]["severity"][name] == "advisory"
        for name in summary["observation_qc"]["advisory_checks"]
    )
    target_quality = summary["calibration_targets"]
    assert target_quality["status"] == "pass"
    assert target_quality["candidate"]["status"] == "pass"
    assert math.isfinite(target_quality["candidate"]["pressure_nrmse"])
    assert math.isfinite(target_quality["candidate"]["absolute_split_error"])
    assert target_quality["candidate"]["gate_results"] == {
        "mpa_pressure": True,
        "rpa_flow_split": True,
    }
    target_configuration = target_quality["configuration"]
    assert (
        target_configuration["mpa_pressure"]["vessel"]
        == calibration.targets.mpa_pressure.vessel
    )
    assert (
        target_configuration["rpa_flow_split"]["rpa_vessel"]
        == calibration.targets.rpa_flow_split.rpa_vessel
    )
    assert (
        target_configuration["rpa_flow_split"]["lpa_vessel"]
        == calibration.targets.rpa_flow_split.lpa_vessel
    )
    assert (
        target_configuration["require_improvement_over_baseline"]
        is calibration.targets.require_improvement_over_baseline
    )
    report_paths = [
        tmp_path / "calibration_observation_qc.json",
        tmp_path / "calibration_confirmation.json",
        tmp_path / "calibration_replay.json",
        tmp_path / "calibration_targets.json",
        tmp_path / "calibration_summary.json",
    ]
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    assert all(report["run_id"] == result["run_id"] for report in reports)
    assert all(report["digests"] == result["digests"] for report in reports)
    assert result["output_config_digest"] == result["digests"]["output"]
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
