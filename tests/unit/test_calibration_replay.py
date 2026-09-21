from __future__ import annotations

import copy

from svzerodtrees.calibration.replay import build_replay_payload, validate_replay


def _config(*, requested_cycles: int = 1) -> dict:
    return {
        "simulation_parameters": {
            "number_of_cardiac_cycles": requested_cycles,
            "number_of_time_pts_per_cardiac_cycle": 4,
            "output_all_cycles": False,
        },
        "y": {
            "pressure:MPA": [100.0, 105.0, 100.0, 100.0],
            "flow:MPA": [1.0, 1.2, 1.0, 1.0],
        },
    }


def _rows(
    *, transient: bool = True, nonfinite: bool = False
) -> list[dict[str, object]]:
    # Three cycles with four points each and one shared endpoint per boundary.
    pressure_cycle = [100.0, 105.0, 100.0, 100.0]
    flow_cycle = [1.0, 1.2, 1.0, 1.0]
    # The last point is shared with the first point of cycle two and therefore
    # already has the settled endpoint value.
    transient_pressure = [200.0, 205.0, 200.0, 100.0]
    transient_flow = [2.0, 2.2, 2.0, 1.0]
    pressure = (
        (transient_pressure if transient else pressure_cycle)
        + pressure_cycle[1:]
        + pressure_cycle[1:]
    )
    flow = (
        (transient_flow if transient else flow_cycle) + flow_cycle[1:] + flow_cycle[1:]
    )
    if nonfinite:
        pressure[4] = float("nan")
    rows = []
    for index, (pressure_value, flow_value) in enumerate(zip(pressure, flow)):
        rows.append(
            {
                "name": "MPA",
                "time": float(index),
                "pressure_out": pressure_value,
                "flow_out": flow_value,
            }
        )
    return rows


def _settings(config: dict, *, maximum_cycles: int = 3) -> tuple[dict, dict]:
    return build_replay_payload(
        config,
        minimum_cycles=3,
        maximum_cycles=maximum_cycles,
        required_consecutive_stable_pairs=1,
    )


def test_three_cycle_startup_transient_returns_only_accepted_final_cycle() -> None:
    config = _config()
    original = copy.deepcopy(config)
    replay_config, settings = _settings(config)

    summary = validate_replay(
        _rows(),
        payload=config,
        validation_settings=settings,
        pressure_bound_multiplier=2.0,
        flow_bound_multiplier=2.0,
        cycle_stability_tolerance=1.0e-12,
    )

    assert summary["status"] == "pass"
    assert summary["cycle_stability"]["convergence_cycle"] == 3
    assert summary["accepted_final_cycle"]["cycle"] == 3
    accepted = summary["accepted_final_cycle"]["series"]
    pressure = next(item for item in accepted if item["kind"] == "pressure")
    assert pressure["values"] == [100.0, 105.0, 100.0, 100.0]
    assert summary["transient_maxima"]["pressure"]["maximum_absolute_value"] == 205.0
    assert config == original
    assert replay_config["simulation_parameters"]["number_of_cardiac_cycles"] == 3
    assert replay_config["simulation_parameters"]["output_all_cycles"] is True
    assert settings["requested_number_of_cardiac_cycles"] == 1


def test_nonfinite_values_fail_with_diagnostic_evidence() -> None:
    summary = validate_replay(
        _rows(nonfinite=True),
        payload=_config(),
        validation_settings={
            "number_of_time_pts_per_cardiac_cycle": 4,
            "replay_minimum_cycles": 3,
            "replay_maximum_cycles": 3,
        },
    )

    assert summary["status"] == "fail"
    assert summary["checks"]["finite_pressure_and_flow"] is False
    assert "non-finite" in summary["error"]


def test_malformed_shared_endpoint_boundaries_fail_diagnostically() -> None:
    rows = _rows()
    rows.pop()  # 9 rows cannot form complete three cycles with shared endpoints.

    summary = validate_replay(
        rows,
        payload=_config(),
        validation_settings={
            "number_of_time_pts_per_cardiac_cycle": 4,
            "replay_minimum_cycles": 3,
            "replay_maximum_cycles": 3,
        },
    )

    assert summary["status"] == "fail"
    assert summary["checks"]["cycle_boundaries"] is False
    assert "complete shared-endpoint cycles" in summary["error"]


def test_final_cycle_bounds_use_settled_cycle_not_startup_transient() -> None:
    summary = validate_replay(
        _rows(),
        payload=_config(),
        validation_settings={
            "number_of_time_pts_per_cardiac_cycle": 4,
            "replay_minimum_cycles": 3,
            "replay_maximum_cycles": 3,
        },
        pressure_bound_multiplier=1.1,
        flow_bound_multiplier=1.1,
        cycle_stability_tolerance=1.0e-12,
    )

    assert summary["status"] == "pass"
    assert summary["checks"]["bounded_pressure"] is True
    assert summary["transient_maxima"]["pressure"]["maximum_absolute_value"] == 205.0
    assert summary["final_cycle_bounds"]["pressure"]["maximum_absolute_value"] == 105.0


def test_final_cycle_bound_failure_is_reported_after_stability() -> None:
    rows = _rows(transient=False)
    # Preserve periodicity while making the accepted pressure cycle exceed its
    # observation-relative bound.
    for row in rows:
        row["pressure_out"] = float(row["pressure_out"]) + 100.0

    summary = validate_replay(
        rows,
        payload=_config(),
        validation_settings={
            "number_of_time_pts_per_cardiac_cycle": 4,
            "replay_minimum_cycles": 3,
            "replay_maximum_cycles": 3,
        },
        pressure_bound_multiplier=1.1,
        flow_bound_multiplier=2.0,
        cycle_stability_tolerance=1.0e-12,
    )

    assert summary["status"] == "fail"
    assert summary["checks"]["cycle_stability"] is True
    assert summary["checks"]["bounded_pressure"] is False
    assert summary["final_cycle_bounds"]["pressure"]["passed"] is False


def test_nonsettling_replay_at_maximum_cycles_reports_pair_trajectory() -> None:
    rows = _rows(transient=False)
    # Make each cycle distinct, so no consecutive pair can pass at zero tolerance.
    for index, row in enumerate(rows):
        if index >= 4:
            row["pressure_out"] = float(row["pressure_out"]) + index

    summary = validate_replay(
        rows,
        payload=_config(),
        validation_settings={
            "number_of_time_pts_per_cardiac_cycle": 4,
            "replay_minimum_cycles": 3,
            "replay_maximum_cycles": 3,
            "required_consecutive_stable_pairs": 1,
        },
        cycle_stability_tolerance=0.0,
    )

    assert summary["status"] == "fail"
    assert summary["checks"]["cycle_stability"] is False
    assert summary["cycle_stability"]["convergence_cycle"] is None
    assert len(summary["cycle_stability"]["pair_metrics"]) == 2
    assert summary["accepted_final_cycle"] is None


def test_required_consecutive_stable_pairs_delay_convergence_cycle() -> None:
    config = _config()
    replay_config, settings = build_replay_payload(
        config,
        minimum_cycles=3,
        maximum_cycles=4,
        required_consecutive_stable_pairs=2,
    )
    rows = _rows(transient=True)
    # Add one settled cycle by repeating the last shared-endpoint cycle.
    last_cycle = rows[-3:]
    rows.extend(
        {
            "name": row["name"],
            "time": float(row["time"]) + 3.0,
            "pressure_out": row["pressure_out"],
            "flow_out": row["flow_out"],
        }
        for row in last_cycle
    )

    summary = validate_replay(
        rows,
        payload=config,
        validation_settings=settings,
        pressure_bound_multiplier=2.0,
        flow_bound_multiplier=2.0,
        cycle_stability_tolerance=1.0e-12,
    )

    assert replay_config["simulation_parameters"]["number_of_cardiac_cycles"] == 4
    assert summary["status"] == "pass"
    assert summary["cycle_stability"]["convergence_cycle"] == 4
    assert summary["accepted_final_cycle"]["cycle"] == 4
