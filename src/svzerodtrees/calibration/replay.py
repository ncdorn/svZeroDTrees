"""Pure parsing and validation helpers for settled zero-dimensional replay.

The solver's replay table contains one row at each time point.  A cycle with
``N`` time points has ``N - 1`` new points because the endpoint is shared with
the following cycle.  This module keeps that boundary convention explicit and
does not modify the configuration passed to it.  The workflow is responsible
for invoking ``pysvzerod.simulate``; this module only builds a validation copy,
parses its result, and evaluates the emitted traces.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np


_REPLAY_SCALE_FLOOR = 1.0e-12
_DEFAULT_MINIMUM_CYCLES = 3
_DEFAULT_MAXIMUM_CYCLES = 10
_DEFAULT_REQUIRED_STABLE_PAIRS = 1
_DEFAULT_BOUND_MULTIPLIER = 10.0
_DEFAULT_STABILITY_TOLERANCE = 1.0e-3


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    """Return a validated integer setting, rejecting booleans and fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer at least {minimum}") from exc
    if integer != value or integer < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return integer


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _table_from_result(result: Any) -> dict[str, np.ndarray]:
    """Convert pandas-like, row-oriented, or column-oriented output to arrays."""
    columns = getattr(result, "columns", None)
    if columns is not None:
        column_names = [str(column) for column in columns]
        table: dict[str, np.ndarray] = {}
        for column_name, source_name in zip(column_names, columns):
            try:
                table[column_name] = np.asarray(result[source_name])
            except Exception as exc:  # pragma: no cover - pandas-specific errors vary
                raise ValueError(
                    f"solver replay result column '{column_name}' is unavailable"
                ) from exc
        return table

    if isinstance(result, list) and all(isinstance(row, Mapping) for row in result):
        if not result:
            raise ValueError("solver replay returned no tabular result")
        column_names = sorted({str(key) for row in result for key in row})
        return {
            column_name: np.asarray([row.get(column_name) for row in result])
            for column_name in column_names
        }

    if isinstance(result, Mapping) and result:
        table: dict[str, np.ndarray] = {}
        for key, values in result.items():
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(
                    "solver replay result must be a table of numeric columns"
                )
            table[str(key)] = np.asarray(values)
        return table

    raise ValueError("solver replay returned no tabular result")


def _as_numeric_array(values: Any, *, label: str, row_count: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"solver replay result {label} values must be numeric"
        ) from exc
    if array.ndim != 1 or len(array) != row_count:
        raise ValueError(f"solver replay result {label} values must match result rows")
    if array.size == 0:
        raise ValueError(f"solver replay result {label} values are empty")
    if not np.isfinite(array).all():
        raise ValueError(f"solver replay result contains non-finite {label} values")
    return array


def _series_item(
    *,
    name: str,
    kind: str,
    column: str,
    indices: list[int],
    times: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    index_array = np.asarray(indices, dtype=np.int64)
    series_times = times[index_array]
    series_values = values[index_array]
    if series_times.size == 0:
        raise ValueError(f"solver replay {kind} series '{name}' is empty")
    if np.any(np.diff(series_times) <= 0.0):
        raise ValueError(
            f"solver replay times for {kind} series '{name}' must be strictly increasing; "
            "shared cycle endpoints must appear once"
        )
    return {
        "name": name,
        "kind": kind,
        "column": column,
        "times": series_times,
        "values": series_values,
    }


def parse_replay_result(result: Any) -> list[dict[str, Any]]:
    """Extract finite pressure and flow series from a solver replay table.

    Both the usual vessel-column result (``pressure_in``, ``flow_out``, ...)
    and the variable result (``name`` identifies ``pressure:...`` or
    ``flow:...`` and values are stored in ``y``) are supported.  Arrays in the
    returned records intentionally remain NumPy arrays for numerical consumers;
    :func:`validate_replay` converts its public diagnostics to JSON-safe lists.
    """
    table = _table_from_result(result)
    if "name" not in table or "time" not in table:
        raise ValueError("solver replay result must include name and time columns")
    row_count = len(table["name"])
    if len(table["time"]) != row_count:
        raise ValueError(
            "solver replay result name and time columns have different lengths"
        )
    try:
        times = np.asarray(table["time"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("solver replay result time column must be numeric") from exc
    if times.ndim != 1 or len(times) != row_count or not times.size:
        raise ValueError("solver replay result time column must be a non-empty vector")
    if not np.isfinite(times).all():
        raise ValueError("solver replay result contains non-finite time values")

    names = table["name"]
    if np.asarray(names).ndim != 1:
        raise ValueError("solver replay result name column must be one-dimensional")
    series: list[dict[str, Any]] = []

    # Variable-based output stores each semantic variable in ``y``.  Grouping
    # by the complete variable name retains the configured vessel/interface.
    if "y" in table:
        variable_values = _as_numeric_array(table["y"], label="y", row_count=row_count)
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, raw_name in enumerate(names):
            variable_name = str(raw_name)
            kind = variable_name.split(":", 1)[0].lower()
            if kind in {"flow", "pressure"}:
                grouped.setdefault((variable_name, kind), []).append(index)
        for (name, kind), indices in sorted(grouped.items()):
            series.append(
                _series_item(
                    name=name,
                    kind=kind,
                    column="y",
                    indices=indices,
                    times=times,
                    values=variable_values,
                )
            )

    # Column-based output has one or more pressure/flow columns, grouped by
    # vessel name.  Keep the source column so interfaces remain distinguishable
    # to downstream target evaluators.
    for column_name, raw_values in table.items():
        lower_name = column_name.lower()
        if lower_name in {"name", "time", "y", "ydot"}:
            continue
        if (
            lower_name.startswith("d_")
            or lower_name.startswith("dflow")
            or lower_name.startswith("dpressure")
        ):
            continue
        kind = (
            "flow"
            if "flow" in lower_name
            else "pressure"
            if "pressure" in lower_name
            else None
        )
        if kind is None:
            continue
        values = _as_numeric_array(
            raw_values,
            label=f"{kind} values in '{column_name}'",
            row_count=row_count,
        )
        grouped_indices: dict[str, list[int]] = {}
        for index, raw_name in enumerate(names):
            grouped_indices.setdefault(str(raw_name), []).append(index)
        for name, indices in sorted(grouped_indices.items()):
            series.append(
                _series_item(
                    name=name,
                    kind=kind,
                    column=column_name,
                    indices=indices,
                    times=times,
                    values=values,
                )
            )

    if not series:
        raise ValueError("solver replay result contains no pressure or flow values")
    if not any(item["kind"] == "pressure" for item in series):
        raise ValueError("solver replay result contains no pressure values")
    if not any(item["kind"] == "flow" for item in series):
        raise ValueError("solver replay result contains no flow values")
    return series


def build_replay_payload(
    published_config: Mapping[str, Any],
    *,
    minimum_cycles: int = _DEFAULT_MINIMUM_CYCLES,
    maximum_cycles: int = _DEFAULT_MAXIMUM_CYCLES,
    required_consecutive_stable_pairs: int = _DEFAULT_REQUIRED_STABLE_PAIRS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validation-only copy and its bounded replay settings.

    The validation copy emits the configured maximum number of cycles.  This
    allows a slowly settling model to reach the minimum accepted horizon while
    keeping the simulation bounded.  The original requested cycle count and
    every other field in ``published_config`` are left untouched.
    """
    if not isinstance(published_config, Mapping):
        raise ValueError("published solver configuration must be a mapping")
    minimum = _positive_integer(minimum_cycles, name="replay_minimum_cycles", minimum=3)
    maximum = _positive_integer(
        maximum_cycles, name="replay_maximum_cycles", minimum=minimum
    )
    required = _positive_integer(
        required_consecutive_stable_pairs,
        name="required_consecutive_stable_pairs",
    )
    if required > maximum - 1:
        raise ValueError(
            "required_consecutive_stable_pairs must leave at least one cycle pair "
            "within replay_maximum_cycles"
        )

    replay_payload = copy.deepcopy(dict(published_config))
    simulation_parameters = replay_payload.get("simulation_parameters")
    if not isinstance(simulation_parameters, dict):
        raise ValueError(
            "published solver configuration requires simulation_parameters"
        )
    requested_cycles = _positive_integer(
        simulation_parameters.get("number_of_cardiac_cycles"),
        name="simulation_parameters.number_of_cardiac_cycles",
    )
    points_per_cycle = _positive_integer(
        simulation_parameters.get("number_of_time_pts_per_cardiac_cycle"),
        name="simulation_parameters.number_of_time_pts_per_cardiac_cycle",
        minimum=2,
    )
    if bool(simulation_parameters.get("coupled_simulation", False)):
        raise ValueError(
            "calibrated replay stability requires a non-coupled cardiac-cycle configuration"
        )

    simulation_parameters["number_of_cardiac_cycles"] = maximum
    simulation_parameters["output_all_cycles"] = True
    simulation_parameters["output_mean_only"] = False
    simulation_parameters["output_derivative"] = False
    simulation_parameters["output_interval"] = 1
    settings = {
        "requested_number_of_cardiac_cycles": requested_cycles,
        "validation_number_of_cardiac_cycles": maximum,
        "number_of_time_pts_per_cardiac_cycle": points_per_cycle,
        "replay_minimum_cycles": minimum,
        "replay_maximum_cycles": maximum,
        "required_consecutive_stable_pairs": required,
        "output_all_cycles": True,
        "output_mean_only": False,
        "output_derivative": False,
        "output_interval": 1,
    }
    return replay_payload, settings


def _observation_scales(payload: Mapping[str, Any] | None) -> dict[str, float]:
    scales = {"pressure": 0.0, "flow": 0.0}
    observations = payload.get("y") if isinstance(payload, Mapping) else None
    if observations is None:
        return {kind: _REPLAY_SCALE_FLOOR for kind in scales}
    if not isinstance(observations, Mapping):
        raise ValueError("calibration payload observations must be a mapping")
    for variable_name, raw_values in observations.items():
        kind = str(variable_name).split(":", 1)[0].lower()
        if kind not in scales:
            continue
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"calibration observation '{variable_name}' must be numeric"
            ) from exc
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"calibration observation '{variable_name}' is empty or non-finite"
            )
        scales[kind] = max(scales[kind], float(np.max(np.abs(values))))
    return {kind: max(scale, _REPLAY_SCALE_FLOOR) for kind, scale in scales.items()}


def _resolved_setting(
    name: str,
    explicit: Any,
    validation_settings: Mapping[str, Any] | None,
    replay_settings: Mapping[str, Any] | None,
    default: Any,
) -> Any:
    if explicit is not None:
        return explicit
    for settings in (validation_settings, replay_settings):
        if isinstance(settings, Mapping) and name in settings:
            return settings[name]
    return default


def _failure_result(
    message: str, *, validation_settings: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "finite_pressure_and_flow": False,
        "cycle_boundaries": False,
        "cycle_count": False,
        "bounded_pressure": False,
        "bounded_flow": False,
        "cycle_stability": False,
    }
    return {
        "status": "fail",
        "error": message,
        "checks": checks,
        "validation_settings": dict(validation_settings),
        "bounds": {},
        "final_cycle_bounds": {},
        "transient_maxima": {},
        "cycle_stability": {
            "convergence_cycle": None,
            "pair_metrics": [],
        },
        "series": [],
        "accepted_final_cycle": None,
    }


def validate_replay(
    result: Any,
    *,
    payload: Mapping[str, Any] | None = None,
    validation_settings: Mapping[str, Any] | None = None,
    replay_settings: Mapping[str, Any] | None = None,
    minimum_cycles: int | None = None,
    maximum_cycles: int | None = None,
    required_consecutive_stable_pairs: int | None = None,
    pressure_bound_multiplier: float | None = None,
    flow_bound_multiplier: float | None = None,
    cycle_stability_tolerance: float | None = None,
    observation_scales: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Validate finite, bounded, periodically settling replay output.

    The returned object is deterministic and JSON serializable.  A successful
    result places only the accepted settled cycle in
    ``accepted_final_cycle``; the full replay remains represented by compact
    transient maxima and per-cycle-pair trajectories.  Parsing failures are
    returned as diagnostic failures so callers can persist them without a
    second exception-handling path.
    """
    try:
        points_per_cycle = _resolved_setting(
            "number_of_time_pts_per_cardiac_cycle",
            None,
            validation_settings,
            replay_settings,
            (payload or {})
            .get("simulation_parameters", {})
            .get("number_of_time_pts_per_cardiac_cycle")
            if isinstance(payload, Mapping)
            else None,
        )
        points_per_cycle = _positive_integer(
            points_per_cycle,
            name="number_of_time_pts_per_cardiac_cycle",
            minimum=2,
        )
        minimum = _positive_integer(
            _resolved_setting(
                "replay_minimum_cycles",
                minimum_cycles,
                validation_settings,
                replay_settings,
                _DEFAULT_MINIMUM_CYCLES,
            ),
            name="replay_minimum_cycles",
            minimum=3,
        )
        maximum = _positive_integer(
            _resolved_setting(
                "replay_maximum_cycles",
                maximum_cycles,
                validation_settings,
                replay_settings,
                _DEFAULT_MAXIMUM_CYCLES,
            ),
            name="replay_maximum_cycles",
            minimum=minimum,
        )
        required = _positive_integer(
            _resolved_setting(
                "required_consecutive_stable_pairs",
                required_consecutive_stable_pairs,
                validation_settings,
                replay_settings,
                _DEFAULT_REQUIRED_STABLE_PAIRS,
            ),
            name="required_consecutive_stable_pairs",
        )
        if required > maximum - 1:
            raise ValueError(
                "required_consecutive_stable_pairs must leave at least one cycle pair "
                "within replay_maximum_cycles"
            )
        pressure_multiplier = _finite_positive(
            _resolved_setting(
                "pressure_bound_multiplier",
                pressure_bound_multiplier,
                validation_settings,
                replay_settings,
                _DEFAULT_BOUND_MULTIPLIER,
            ),
            name="pressure_bound_multiplier",
        )
        flow_multiplier = _finite_positive(
            _resolved_setting(
                "flow_bound_multiplier",
                flow_bound_multiplier,
                validation_settings,
                replay_settings,
                _DEFAULT_BOUND_MULTIPLIER,
            ),
            name="flow_bound_multiplier",
        )
        tolerance = _finite_nonnegative(
            _resolved_setting(
                "cycle_stability_tolerance",
                cycle_stability_tolerance,
                validation_settings,
                replay_settings,
                _DEFAULT_STABILITY_TOLERANCE,
            ),
            name="cycle_stability_tolerance",
        )
        scales = {
            kind: float(value)
            for kind, value in (
                observation_scales or _observation_scales(payload)
            ).items()
        }
        for kind in ("pressure", "flow"):
            scales[kind] = _finite_positive(
                max(scales.get(kind, _REPLAY_SCALE_FLOOR), _REPLAY_SCALE_FLOOR),
                name=f"{kind} observation scale",
            )
    except (TypeError, ValueError) as exc:
        return _failure_result(str(exc), validation_settings=validation_settings or {})

    resolved_validation = {
        "number_of_time_pts_per_cardiac_cycle": points_per_cycle,
        "replay_minimum_cycles": minimum,
        "replay_maximum_cycles": maximum,
        "required_consecutive_stable_pairs": required,
        "pressure_bound_multiplier": pressure_multiplier,
        "flow_bound_multiplier": flow_multiplier,
        "cycle_stability_tolerance": tolerance,
    }
    try:
        series = parse_replay_result(result)
    except (TypeError, ValueError) as exc:
        return _failure_result(str(exc), validation_settings=resolved_validation)

    cycle_span = points_per_cycle - 1
    cycle_counts: list[int] = []
    reference_times: np.ndarray | None = None
    boundary_errors: list[str] = []
    for item in series:
        values = item["values"]
        times = item["times"]
        if len(values) < points_per_cycle:
            boundary_errors.append(
                f"series '{item['name']}' has {len(values)} samples; "
                f"one cycle requires {points_per_cycle}"
            )
            continue
        remainder = (len(values) - 1) % cycle_span
        if remainder:
            boundary_errors.append(
                f"series '{item['name']}' has {len(values)} samples, which does not "
                f"form complete shared-endpoint cycles for {points_per_cycle} points per cycle"
            )
            continue
        intervals = np.diff(times)
        if intervals.size and not np.allclose(
            intervals,
            intervals[0],
            rtol=0.0,
            atol=max(abs(float(intervals[0])) * 1.0e-9, 1.0e-12),
        ):
            boundary_errors.append(
                f"series '{item['name']}' has non-uniform time intervals; "
                "shared cycle boundaries cannot be identified deterministically"
            )
            continue
        cycle_counts.append((len(values) - 1) // cycle_span)
        if reference_times is None:
            reference_times = times
        elif len(times) != len(reference_times) or not np.allclose(
            times, reference_times, rtol=0.0, atol=1.0e-12
        ):
            boundary_errors.append(
                f"series '{item['name']}' does not share the replay cycle time boundaries"
            )

    cycle_count = (
        cycle_counts[0] if cycle_counts and len(set(cycle_counts)) == 1 else None
    )
    if cycle_count is None:
        boundary_errors.append(
            "pressure and flow series do not share complete cycle boundaries"
        )
    if boundary_errors:
        failure = _failure_result(
            "; ".join(boundary_errors),
            validation_settings=resolved_validation,
        )
        failure["checks"]["finite_pressure_and_flow"] = True
        failure["diagnostics"] = {"boundary_errors": boundary_errors}
        return failure
    assert cycle_count is not None  # narrowed by the boundary checks above

    cycle_count_check = minimum <= cycle_count <= maximum
    transient_maxima: dict[str, dict[str, Any]] = {}
    for kind in ("pressure", "flow"):
        kind_series = [item for item in series if item["kind"] == kind]
        maximum_value = max(
            (float(np.max(np.abs(item["values"]))) for item in kind_series),
            default=0.0,
        )
        transient_maxima[kind] = {
            "maximum_absolute_value": maximum_value,
            "series_count": len(kind_series),
            "includes_startup_transient": True,
        }

    pair_metrics: list[dict[str, Any]] = []
    stable_pair_flags: list[bool] = []
    for pair_index in range(max(cycle_count - 1, 0)):
        pair_series: list[dict[str, Any]] = []
        maxima = {"pressure": None, "flow": None}
        for item in series:
            first_start = pair_index * cycle_span
            second_start = (pair_index + 1) * cycle_span
            first = item["values"][first_start : first_start + points_per_cycle]
            second = item["values"][second_start : second_start + points_per_cycle]
            normalized_rms = float(
                np.sqrt(np.mean(np.square(second - first)))
                / max(scales[item["kind"]], _REPLAY_SCALE_FLOOR)
            )
            passed = bool(normalized_rms <= tolerance)
            pair_series.append(
                {
                    "name": str(item["name"]),
                    "kind": str(item["kind"]),
                    "column": str(item.get("column", "")),
                    "normalized_rms": normalized_rms,
                    "passed": passed,
                }
            )
            if maxima[item["kind"]] is None or normalized_rms > maxima[item["kind"]]:
                maxima[item["kind"]] = normalized_rms
        pair_passed = bool(
            pair_series
            and all(item["passed"] for item in pair_series)
            and any(item["kind"] == "pressure" for item in pair_series)
            and any(item["kind"] == "flow" for item in pair_series)
        )
        stable_pair_flags.append(pair_passed)
        pair_metrics.append(
            {
                "left_cycle": pair_index + 1,
                "right_cycle": pair_index + 2,
                "maximum_pressure_normalized_rms": maxima["pressure"],
                "maximum_flow_normalized_rms": maxima["flow"],
                "passed": pair_passed,
                "series": sorted(
                    pair_series,
                    key=lambda item: (item["kind"], item["name"], item["column"]),
                ),
            }
        )

    convergence_cycle: int | None = None
    consecutive = 0
    for pair_index, pair_passed in enumerate(stable_pair_flags):
        consecutive = consecutive + 1 if pair_passed else 0
        right_cycle = pair_index + 2
        if right_cycle >= minimum and consecutive >= required:
            convergence_cycle = right_cycle
            break

    candidate_cycle = (
        convergence_cycle if convergence_cycle is not None else cycle_count
    )
    candidate_start = (candidate_cycle - 1) * cycle_span
    candidate_bounds: dict[str, dict[str, Any]] = {}
    for kind in ("pressure", "flow"):
        kind_items = [item for item in series if item["kind"] == kind]
        series_bounds: list[dict[str, Any]] = []
        bound_limit = (
            pressure_multiplier if kind == "pressure" else flow_multiplier
        ) * scales[kind]
        for item in kind_items:
            values = item["values"][
                candidate_start : candidate_start + points_per_cycle
            ]
            maximum_value = (
                float(np.max(np.abs(values))) if values.size else float("nan")
            )
            series_bounds.append(
                {
                    "name": str(item["name"]),
                    "column": str(item.get("column", "")),
                    "maximum_absolute_value": maximum_value,
                    "limit": float(bound_limit),
                    "passed": bool(values.size and maximum_value <= bound_limit),
                }
            )
        maximum_value = max(
            (entry["maximum_absolute_value"] for entry in series_bounds),
            default=0.0,
        )
        candidate_bounds[kind] = {
            "observation_scale": float(scales[kind]),
            "multiplier": float(
                pressure_multiplier if kind == "pressure" else flow_multiplier
            ),
            "limit": float(bound_limit),
            "maximum_absolute_value": float(maximum_value),
            "passed": bool(
                series_bounds and all(entry["passed"] for entry in series_bounds)
            ),
            "series": series_bounds,
        }

    if convergence_cycle is not None:
        accepted_series: list[dict[str, Any]] = []
        for item in series:
            values = item["values"][
                candidate_start : candidate_start + points_per_cycle
            ]
            times = item["times"][candidate_start : candidate_start + points_per_cycle]
            accepted_series.append(
                {
                    "name": str(item["name"]),
                    "kind": str(item["kind"]),
                    "column": str(item.get("column", "")),
                    "times": [float(value) for value in times],
                    "values": [float(value) for value in values],
                }
            )
        accepted_final_cycle: dict[str, Any] | None = {
            "cycle": convergence_cycle,
            "start_time": float(reference_times[candidate_start]),
            "end_time": float(reference_times[candidate_start + points_per_cycle - 1]),
            "series": sorted(
                accepted_series,
                key=lambda item: (item["kind"], item["name"], item["column"]),
            ),
        }
    else:
        accepted_final_cycle = None

    stability = {
        "tolerance": float(tolerance),
        "required_consecutive_stable_pairs": required,
        "convergence_cycle": convergence_cycle,
        "maximum_pressure_relative_rms": max(
            (
                pair["maximum_pressure_normalized_rms"]
                for pair in pair_metrics
                if pair["maximum_pressure_normalized_rms"] is not None
            ),
            default=None,
        ),
        "maximum_flow_relative_rms": max(
            (
                pair["maximum_flow_normalized_rms"]
                for pair in pair_metrics
                if pair["maximum_flow_normalized_rms"] is not None
            ),
            default=None,
        ),
        "pair_metrics": pair_metrics,
        "passed": convergence_cycle is not None,
    }
    checks = {
        "finite_pressure_and_flow": True,
        "cycle_boundaries": True,
        "cycle_count": cycle_count_check,
        "bounded_pressure": candidate_bounds["pressure"]["passed"],
        "bounded_flow": candidate_bounds["flow"]["passed"],
        "cycle_stability": stability["passed"],
    }
    diagnostics = {
        "cycle_count": cycle_count,
        "transient_maxima": transient_maxima,
        "final_cycle_bounds": candidate_bounds,
        "accepted_final_cycle": accepted_final_cycle,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "validation_settings": resolved_validation,
        "bounds": candidate_bounds,
        "final_cycle_bounds": candidate_bounds,
        "transient_maxima": transient_maxima,
        "cycle_stability": stability,
        "series": [
            {
                "name": str(item["name"]),
                "kind": str(item["kind"]),
                "column": str(item.get("column", "")),
                "sample_count": int(len(item["values"])),
                "maximum_absolute_value": float(np.max(np.abs(item["values"]))),
            }
            for item in sorted(
                series,
                key=lambda item: (item["kind"], item["name"], item["column"]),
            )
        ],
        "accepted_final_cycle": accepted_final_cycle,
        "diagnostics": diagnostics,
    }


# Compatibility names used by the current workflow and convenient for callers
# migrating to the extracted module.  The coordinator task can adopt the public
# names without carrying any private parsing implementation with it.
_build_replay_payload = build_replay_payload
_evaluate_replay = validate_replay
_replay_series = parse_replay_result
evaluate_replay = validate_replay


__all__ = [
    "build_replay_payload",
    "evaluate_replay",
    "parse_replay_result",
    "validate_replay",
]
