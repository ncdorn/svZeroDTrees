from __future__ import annotations

import json

import numpy as np
import pytest

from svzerodtrees.calibration.targets import (
    TargetSeries,
    evaluate_pulmonary_targets,
    extract_target_series,
)


PHASES = np.array([0.0, 0.25, 0.5, 0.75])
TARGETS = {
    "mpa_pressure": {
        "vessel": "mpa",
        "interface": "internal",
        "weight": 1.0,
        "normalized_rms_tolerance": 0.05,
    },
    "rpa_flow_split": {
        "rpa_vessel": "rpa",
        "lpa_vessel": "lpa",
        "interface": "external_downstream",
        "weight": 1.0,
        "absolute_tolerance": 0.02,
    },
}


def _series(phases, values, units, **extra):
    result = {"phases": list(phases), "values": list(values), "units": units}
    result.update(extra)
    return result


def _inputs(*, pressure_0d=None, pressure_0d_units="mmHg", split_0d=0.6):
    phases_0d = np.linspace(0.0, 1.0, 9)[:-1]
    pressure_3d = 50.0 + 10.0 * np.sin(2.0 * np.pi * PHASES)
    pressure_0d_values = (
        50.0 + 10.0 * np.sin(2.0 * np.pi * phases_0d)
        if pressure_0d is None
        else pressure_0d
    )
    observations = {
        "mpa_pressure": _series(PHASES, pressure_3d, "mmHg"),
        "rpa_flow": _series(PHASES, np.full(4, split_0d), "cm^3/s"),
        "lpa_flow": _series(PHASES, np.full(4, 1.0 - split_0d), "cm^3/s"),
    }
    settled = {
        "mpa_pressure": _series(phases_0d, pressure_0d_values, pressure_0d_units),
        "rpa_flow": _series(phases_0d, np.full(8, split_0d), "cm^3/s"),
        "lpa_flow": _series(phases_0d, np.full(8, 1.0 - split_0d), "cm^3/s"),
    }
    return observations, settled


def test_target_evaluation_returns_metrics_and_deterministic_json():
    observations, settled = _inputs()
    result = evaluate_pulmonary_targets(observations, settled, TARGETS)

    assert result.pressure_nrmse == pytest.approx(0.0, abs=1e-12)
    assert result.rpa_split_3d == pytest.approx(0.6)
    assert result.rpa_split_0d == pytest.approx(0.6)
    assert result.absolute_split_error == pytest.approx(0.0, abs=1e-12)
    assert result.composite_score == pytest.approx(0.0, abs=1e-12)
    assert result.passed
    assert result.component_gates == {"mpa_pressure": True, "rpa_flow_split": True}
    payload = result.as_dict()
    assert payload["units"] == {"pressure": "mmHg", "flow": "cm^3/s"}
    assert json.loads(result.to_json()) == payload


def test_pressure_unit_alias_and_phase_grid_are_invariant():
    observations, settled = _inputs()
    pressure_pa = np.asarray(settled["mpa_pressure"]["values"]) * 133.32236842105263
    settled["mpa_pressure"] = _series(
        settled["mpa_pressure"]["phases"], pressure_pa, "Pa"
    )
    result = evaluate_pulmonary_targets(observations, settled, TARGETS)
    assert result.pressure_nrmse == pytest.approx(0.0, abs=1e-12)
    assert result.passed


def test_dyn_per_cm2_pressure_is_normalized_for_both_sources():
    observations, settled = _inputs()
    observations["mpa_pressure"] = _series(
        observations["mpa_pressure"]["phases"],
        np.asarray(observations["mpa_pressure"]["values"]) * 1333.2236842105263,
        "dyn/cm^2",
    )
    settled["mpa_pressure"] = _series(
        settled["mpa_pressure"]["phases"],
        np.asarray(settled["mpa_pressure"]["values"]) * 1333.2236842105263,
        "dyn/cm^2",
    )

    result = evaluate_pulmonary_targets(observations, settled, TARGETS)

    assert result.pressure_nrmse == pytest.approx(0.0, abs=1e-12)
    assert result.passed


def test_periodic_interpolation_handles_a_phase_grid_without_zero():
    observations, settled = _inputs()
    observations["mpa_pressure"]["values"] = [50.0, 55.0, 50.0, 45.0]
    phases = np.array([0.125, 0.375, 0.625, 0.875])
    settled["mpa_pressure"] = _series(
        phases,
        [55.0, 55.0, 45.0, 45.0],
        "mmHg",
    )
    result = evaluate_pulmonary_targets(observations, settled, TARGETS)
    assert result.pressure_nrmse == pytest.approx(0.0, abs=1e-12)


def test_split_orientation_is_applied_before_cycle_integration():
    observations, settled = _inputs()
    observations["rpa_flow"]["values"] = [-value for value in observations["rpa_flow"]["values"]]
    observations["lpa_flow"]["values"] = [-value for value in observations["lpa_flow"]["values"]]
    settled["rpa_flow"]["values"] = [-value for value in settled["rpa_flow"]["values"]]
    settled["lpa_flow"]["values"] = [-value for value in settled["lpa_flow"]["values"]]
    observations["rpa_flow"]["orientation"] = "toward_mpa"
    observations["lpa_flow"]["orientation"] = "toward_mpa"
    settled["rpa_flow"]["orientation"] = "toward_mpa"
    settled["lpa_flow"]["orientation"] = "toward_mpa"
    result = evaluate_pulmonary_targets(observations, settled, TARGETS)
    assert result.rpa_split_3d == pytest.approx(0.6)
    assert result.rpa_split_0d == pytest.approx(0.6)


def test_independent_component_gates_cannot_be_masked_by_composite_score():
    observations, settled = _inputs()
    settled["mpa_pressure"]["values"] = [value + 2.0 for value in settled["mpa_pressure"]["values"]]
    targets = {
        **TARGETS,
        "mpa_pressure": {**TARGETS["mpa_pressure"], "weight": 0.01},
    }
    result = evaluate_pulmonary_targets(observations, settled, targets)
    assert not result.pressure_gate.passed
    assert result.split_gate.passed
    assert result.composite_score < 1.0
    assert not result.passed


def test_invalid_split_denominator_fails_clearly():
    observations, settled = _inputs()
    observations["rpa_flow"]["values"] = [-1.0] * 4
    observations["lpa_flow"]["values"] = [1.0] * 4
    with pytest.raises(ValueError, match="3D RPA/LPA split denominator"):
        evaluate_pulmonary_targets(observations, settled, TARGETS)


def test_target_series_and_extraction_require_explicit_units_and_names():
    series = TargetSeries.from_values(PHASES, [1.0, 2.0, 1.0, 0.0], units="mmHg")
    extracted = extract_target_series(
        {"pressure:mpa:internal": series},
        vessel="mpa",
        interface="internal",
        quantity="pressure",
    )
    assert extracted == series
    with pytest.raises(ValueError, match="explicit units"):
        evaluate_pulmonary_targets(
            {
                "mpa_pressure": {"phases": PHASES, "values": [1, 2, 1, 0]},
                "rpa_flow": _series(PHASES, [1, 1, 1, 1], "cm^3/s"),
                "lpa_flow": _series(PHASES, [1, 1, 1, 1], "cm^3/s"),
            },
            _inputs()[1],
            TARGETS,
        )
