"""Pure pulmonary target extraction and scoring for calibration.

This module deliberately has no dependency on the solver, replay code, or the
pulmonary tuning helpers.  The coordinator supplies already-qualified target
series and configuration; this module only normalizes, aligns, scores, and
serializes those series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

from ..numerics import trapezoid


_MMHG_TO_PA = 133.32236842105263
_MMHG_TO_BARYE = 1333.2236842105263
_DEFAULT_PRESSURE_SCALE_FLOOR = 1.0e-6
_DEFAULT_SPLIT_DENOMINATOR_FLOOR = 1.0e-12
_PHASE_TOLERANCE = 1.0e-10

_PRESSURE_UNITS_TO_MMHG = {
    "mmhg": 1.0,
    "mm[ hg ]": 1.0,
    "mm[hg]": 1.0,
    "mm of mercury": 1.0,
    "mmhg_abs": 1.0,
    "pa": 1.0 / _MMHG_TO_PA,
    "pascal": 1.0 / _MMHG_TO_PA,
    "pascals": 1.0 / _MMHG_TO_PA,
    "kpa": 1000.0 / _MMHG_TO_PA,
    "kilopascal": 1000.0 / _MMHG_TO_PA,
    "kilopascals": 1000.0 / _MMHG_TO_PA,
    "barye": 1.0 / _MMHG_TO_BARYE,
    "dyn/cm^2": 1.0 / _MMHG_TO_BARYE,
    "dyn/cm2": 1.0 / _MMHG_TO_BARYE,
    "dynes/cm^2": 1.0 / _MMHG_TO_BARYE,
    "dynes/cm2": 1.0 / _MMHG_TO_BARYE,
    "dyne/cm^2": 1.0 / _MMHG_TO_BARYE,
    "dyne/cm2": 1.0 / _MMHG_TO_BARYE,
}

_FLOW_UNITS_TO_CM3_PER_S = {
    "cm^3/s": 1.0,
    "cm3/s": 1.0,
    "cm³/s": 1.0,
    "cc/s": 1.0,
    "ml/s": 1.0,
    "milliliter/s": 1.0,
    "milliliters/s": 1.0,
    "l/s": 1000.0,
    "liter/s": 1000.0,
    "liters/s": 1000.0,
    "m^3/s": 1.0e6,
    "m3/s": 1.0e6,
    "m³/s": 1.0e6,
}


def _unit_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target series units must be an explicit non-empty string")
    normalized = " ".join(value.strip().lower().replace("²", "^2").split())
    return normalized.replace(" / ", "/").replace(" ^ ", "^")


def _pressure_factor(units: str) -> tuple[str, float]:
    key = _unit_key(units)
    try:
        return "mmHg", _PRESSURE_UNITS_TO_MMHG[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_PRESSURE_UNITS_TO_MMHG))
        raise ValueError(
            f"unsupported pressure units {units!r}; supported units are {supported}"
        ) from exc


def _flow_factor(units: str) -> tuple[str, float]:
    key = _unit_key(units)
    try:
        return "cm^3/s", _FLOW_UNITS_TO_CM3_PER_S[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_FLOW_UNITS_TO_CM3_PER_S))
        raise ValueError(
            f"unsupported volumetric-flow units {units!r}; supported units are {supported}"
        ) from exc


def _as_finite_vector(values: Any, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric one-dimensional sequence") from exc
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{label} must contain at least two values")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _as_phases(values: Any, *, label: str) -> np.ndarray:
    phases = _as_finite_vector(values, label=label)
    if np.any(phases < -_PHASE_TOLERANCE) or np.any(
        phases > 1.0 + _PHASE_TOLERANCE
    ):
        raise ValueError(f"{label} must lie in the normalized periodic interval [0, 1]")
    return np.clip(phases, 0.0, 1.0)


def _orientation_sign(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        sign = float(value)
        if not np.isfinite(sign) or sign == 0.0:
            raise ValueError("flow orientation sign must be finite and non-zero")
        return 1.0 if sign > 0.0 else -1.0
    if not isinstance(value, str):
        raise ValueError(
            "flow orientation must be 'away_from_mpa', 'toward_mpa', or a non-zero sign"
        )
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "away_from_mpa",
        "away",
        "positive",
        "forward",
        "outward",
    }:
        return 1.0
    if normalized in {
        "toward_mpa",
        "toward",
        "negative",
        "reverse",
        "inward",
    }:
        return -1.0
    raise ValueError(
        "flow orientation must be 'away_from_mpa', 'toward_mpa', or a non-zero sign"
    )


def _read_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


@dataclass(frozen=True)
class TargetSeries:
    """A validated periodic target series in its source units.

    ``phases`` are normalized to one cardiac cycle, with zero at the start of
    the cycle.  Flow ``orientation`` identifies the sign convention relative
    to the MPA; it is intentionally explicit rather than inferred from a
    vessel name or branch number.
    """

    phases: tuple[float, ...]
    values: tuple[float, ...]
    units: str
    orientation: str | float = "away_from_mpa"

    @classmethod
    def from_values(
        cls,
        phases: Sequence[float],
        values: Sequence[float],
        *,
        units: str,
        orientation: str | float = "away_from_mpa",
    ) -> "TargetSeries":
        phase_array = _as_phases(phases, label="target phases")
        value_array = _as_finite_vector(values, label="target values")
        if phase_array.size != value_array.size:
            raise ValueError("target phases and values must have the same length")
        _unit_key(units)
        _orientation_sign(orientation)
        return cls(
            phases=tuple(float(item) for item in phase_array),
            values=tuple(float(item) for item in value_array),
            units=str(units),
            orientation=orientation,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phases": list(self.phases),
            "values": list(self.values),
            "units": self.units,
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class TargetGateResult:
    """One independently evaluated target component gate."""

    name: str
    error: float
    tolerance: float
    normalized_error: float
    weight: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "error": float(self.error),
            "tolerance": float(self.tolerance),
            "normalized_error": float(self.normalized_error),
            "weight": float(self.weight),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class PulmonaryTargetEvaluation:
    """Deterministic target metrics and independent gate results."""

    pressure_nrmse: float
    rpa_split_3d: float
    rpa_split_0d: float
    absolute_split_error: float
    pressure_normalized_error: float
    split_normalized_error: float
    composite_score: float
    pressure_gate: TargetGateResult
    split_gate: TargetGateResult
    pressure_units: str
    flow_units: str
    pressure_rms_error: float
    pressure_scale: float
    rpa_flow_3d: float
    lpa_flow_3d: float
    rpa_flow_0d: float
    lpa_flow_0d: float

    @property
    def component_gates(self) -> dict[str, bool]:
        return {
            "mpa_pressure": bool(self.pressure_gate.passed),
            "rpa_flow_split": bool(self.split_gate.passed),
        }

    @property
    def gate_results(self) -> dict[str, bool]:
        return self.component_gates

    @property
    def pressure_passed(self) -> bool:
        return bool(self.pressure_gate.passed)

    @property
    def split_passed(self) -> bool:
        return bool(self.split_gate.passed)

    @property
    def normalized_component_errors(self) -> dict[str, float]:
        return {
            "mpa_pressure": float(self.pressure_normalized_error),
            "rpa_flow_split": float(self.split_normalized_error),
        }

    @property
    def units(self) -> dict[str, str]:
        return {"pressure": self.pressure_units, "flow": self.flow_units}

    @property
    def passed(self) -> bool:
        return bool(self.pressure_gate.passed and self.split_gate.passed)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping with stable key and gate ordering."""

        return {
            "mpa_pressure": {
                "pressure_nrmse": float(self.pressure_nrmse),
                "pressure_rms_error": float(self.pressure_rms_error),
                "pressure_scale": float(self.pressure_scale),
                "normalized_error": float(self.pressure_normalized_error),
                "tolerance": float(self.pressure_gate.tolerance),
                "weight": float(self.pressure_gate.weight),
                "units": self.pressure_units,
                "passed": bool(self.pressure_gate.passed),
            },
            "rpa_flow_split": {
                "rpa_split_3d": float(self.rpa_split_3d),
                "rpa_split_0d": float(self.rpa_split_0d),
                "absolute_split_error": float(self.absolute_split_error),
                "normalized_error": float(self.split_normalized_error),
                "tolerance": float(self.split_gate.tolerance),
                "weight": float(self.split_gate.weight),
                "units": self.flow_units,
                "rpa_flow_3d": float(self.rpa_flow_3d),
                "lpa_flow_3d": float(self.lpa_flow_3d),
                "rpa_flow_0d": float(self.rpa_flow_0d),
                "lpa_flow_0d": float(self.lpa_flow_0d),
                "passed": bool(self.split_gate.passed),
            },
            "pressure_nrmse": float(self.pressure_nrmse),
            "rpa_split_3d": float(self.rpa_split_3d),
            "rpa_split_0d": float(self.rpa_split_0d),
            "absolute_split_error": float(self.absolute_split_error),
            "normalized_component_errors": {
                "mpa_pressure": float(self.pressure_normalized_error),
                "rpa_flow_split": float(self.split_normalized_error),
            },
            "composite_score": float(self.composite_score),
            "units": {"pressure": self.pressure_units, "flow": self.flow_units},
            "gate_results": self.component_gates,
            "passed": self.passed,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def _coerce_series(
    source: Any,
    *,
    phases: Sequence[float] | None,
    units: str | None,
    label: str,
) -> TargetSeries:
    if isinstance(source, TargetSeries):
        if phases is not None or units is not None:
            # Explicit call arguments are allowed only when they agree.  This
            # prevents a caller from silently relabeling an already-qualified
            # series.
            if phases is not None and tuple(float(x) for x in phases) != source.phases:
                raise ValueError(f"{label} supplies phases that disagree with TargetSeries")
            if units is not None and _unit_key(units) != _unit_key(source.units):
                raise ValueError(f"{label} supplies units that disagree with TargetSeries")
        return source

    orientation: str | float = "away_from_mpa"
    values = source
    local_phases = phases
    local_units = units
    if isinstance(source, Mapping):
        if "values" in source:
            values = source["values"]
        elif "series" in source:
            values = source["series"]
        else:
            raise ValueError(f"{label} mapping must contain a values field")
        local_phases = source.get("phases", source.get("phase", local_phases))
        local_units = source.get("units", local_units)
        orientation = source.get("orientation", source.get("flow_orientation", orientation))
        if local_phases is None and source.get("times") is not None:
            cycle_duration = source.get("cycle_duration", source.get("cycle_duration_s"))
            if cycle_duration is None:
                raise ValueError(f"{label} times require an explicit cycle_duration")
            cycle_duration = float(cycle_duration)
            if not np.isfinite(cycle_duration) or cycle_duration <= 0.0:
                raise ValueError(f"{label} cycle_duration must be finite and positive")
            local_phases = np.asarray(source["times"], dtype=np.float64) / cycle_duration

    if local_phases is None:
        raise ValueError(f"{label} requires an explicit normalized phase grid")
    if local_units is None:
        raise ValueError(f"{label} requires explicit units")
    return TargetSeries.from_values(
        local_phases,
        values,
        units=str(local_units),
        orientation=orientation,
    )


def _source_value(source: Any, names: Sequence[str], *, label: str) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        # A nested quantity mapping is convenient for coordinator payloads.
        for quantity in ("pressure", "flow"):
            nested = source.get(quantity)
            if isinstance(nested, Mapping):
                for name in names:
                    if name in nested:
                        return nested[name]
    else:
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
    raise ValueError(f"{label} target series is missing")


def extract_target_series(
    source: Mapping[str, Any],
    *,
    vessel: str,
    interface: str,
    quantity: str,
    phases: Sequence[float] | None = None,
    units: str | None = None,
    orientation: str | float | None = None,
) -> TargetSeries:
    """Extract one explicitly named interface series from coordinator data.

    The function accepts either flat keys (for example,
    ``pressure:branch0_seg0:J0``) or a nested ``pressure``/``flow`` mapping.
    It does not infer a vessel role from a branch number.
    """

    normalized_quantity = str(quantity).strip().lower()
    if normalized_quantity not in {"pressure", "flow", "volumetric_flow"}:
        raise ValueError("quantity must be pressure or flow")
    quantity_key = "flow" if normalized_quantity == "volumetric_flow" else normalized_quantity
    if not isinstance(vessel, str) or not vessel.strip():
        raise ValueError("vessel must be an explicit non-empty name")
    if not isinstance(interface, str) or not interface.strip():
        raise ValueError("interface must be an explicit non-empty name")
    names = (
        f"{quantity_key}:{vessel}:{interface}",
        f"{quantity_key}:{interface}:{vessel}",
        f"{quantity_key}_{vessel}_{interface}",
    )
    raw = _source_value(source, names, label=f"{quantity_key} {vessel} {interface}")
    if orientation is not None:
        if isinstance(raw, Mapping):
            if "orientation" not in raw:
                raw = dict(raw)
                raw["orientation"] = orientation
        else:
            raw = {"values": raw, "orientation": orientation}
    return _coerce_series(
        raw,
        phases=phases,
        units=units,
        label=f"{quantity_key} {vessel} {interface}",
    )


def _periodic_arrays(series: TargetSeries, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    phases = _as_phases(series.phases, label=f"{label} phases")
    values = _as_finite_vector(series.values, label=f"{label} values")
    if phases.size != values.size:
        raise ValueError(f"{label} phases and values must have the same length")
    order = np.argsort(phases, kind="mergesort")
    phases = phases[order]
    values = values[order]

    # A sampled cycle may include both 0 and 1.  They represent the same
    # periodic endpoint and are accepted only when their values agree.
    if phases[0] <= _PHASE_TOLERANCE and phases[-1] >= 1.0 - _PHASE_TOLERANCE:
        if not math.isclose(float(values[0]), float(values[-1]), rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"{label} periodic endpoints at phase 0 and 1 disagree")
        phases = phases[:-1]
        values = values[:-1]
    if phases.size < 2:
        raise ValueError(f"{label} requires at least two unique periodic phases")
    if np.any(np.diff(phases) <= _PHASE_TOLERANCE):
        raise ValueError(f"{label} phases must be strictly increasing")
    return phases, values


def _periodic_interpolate(
    source: TargetSeries,
    destination_phases: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    phases, values = _periodic_arrays(source, label=label)
    destination = np.mod(destination_phases, 1.0)
    # Shift destination phases into the source's periodic coordinate frame.
    # This makes a source grid that starts at (for example) phase 0.25 wrap
    # through its final sample rather than clamping early destination phases.
    destination = (destination - phases[0]) % 1.0 + phases[0]
    extended_phases = np.concatenate((phases, [phases[0] + 1.0]))
    extended_values = np.concatenate((values, [values[0]]))
    return np.interp(destination, extended_phases, extended_values)


def _periodic_integral(series: TargetSeries, *, label: str) -> float:
    phases, values = _periodic_arrays(series, label=label)
    oriented = values * _orientation_sign(series.orientation)
    extended_phases = np.concatenate((phases, [phases[0] + 1.0]))
    extended_values = np.concatenate((oriented, [oriented[0]]))
    return float(trapezoid(extended_values, extended_phases))


def _target_block(targets: Any, name: str) -> Any:
    block = _read_field(targets, name, None)
    if block is None:
        raise ValueError(f"target configuration is missing {name}")
    return block


def _positive_setting(block: Any, names: Sequence[str], *, label: str, default: float | None = None) -> float:
    value: Any = None
    for name in names:
        value = _read_field(block, name, None)
        if value is not None:
            break
    if value is None:
        if default is None:
            raise ValueError(f"target configuration is missing {label}")
        value = default
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def evaluate_pulmonary_targets(
    observations_3d: Mapping[str, Any],
    settled_0d: Mapping[str, Any],
    targets: Any,
    *,
    pressure_units_3d: str | None = None,
    pressure_units_0d: str | None = None,
    flow_units_3d: str | None = None,
    flow_units_0d: str | None = None,
    phases_3d: Sequence[float] | None = None,
    phases_0d: Sequence[float] | None = None,
    pressure_scale_floor: float = _DEFAULT_PRESSURE_SCALE_FLOOR,
    split_denominator_floor: float = _DEFAULT_SPLIT_DENOMINATOR_FLOOR,
) -> PulmonaryTargetEvaluation:
    """Evaluate MPA pressure and RPA flow split against 3D targets.

    ``observations_3d`` and ``settled_0d`` must provide ``mpa_pressure``,
    ``rpa_flow``, and ``lpa_flow`` entries.  Each entry is either a
    :class:`TargetSeries`, a mapping containing ``phases``, ``values``, and
    ``units``, or a values sequence paired with the corresponding global phase
    and unit arguments.  The target configuration is the validated
    ``CalibrationTargetsConfig`` (or an equivalent mapping).
    """

    try:
        pressure_floor = float(pressure_scale_floor)
        denominator_floor = float(split_denominator_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("pressure and split numerical floors must be finite and positive") from exc
    if not np.isfinite(pressure_floor) or pressure_floor <= 0.0:
        raise ValueError("pressure_scale_floor must be finite and positive")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("split_denominator_floor must be finite and positive")

    mpa_3d = _coerce_series(
        _source_value(observations_3d, ("mpa_pressure", "pressure_mpa", "mpa"), label="3D MPA pressure"),
        phases=phases_3d,
        units=pressure_units_3d,
        label="3D MPA pressure",
    )
    mpa_0d = _coerce_series(
        _source_value(settled_0d, ("mpa_pressure", "pressure_mpa", "mpa"), label="0D MPA pressure"),
        phases=phases_0d,
        units=pressure_units_0d,
        label="0D MPA pressure",
    )
    rpa_3d = _coerce_series(
        _source_value(observations_3d, ("rpa_flow", "flow_rpa", "rpa"), label="3D RPA flow"),
        phases=phases_3d,
        units=flow_units_3d,
        label="3D RPA flow",
    )
    lpa_3d = _coerce_series(
        _source_value(observations_3d, ("lpa_flow", "flow_lpa", "lpa"), label="3D LPA flow"),
        phases=phases_3d,
        units=flow_units_3d,
        label="3D LPA flow",
    )
    rpa_0d = _coerce_series(
        _source_value(settled_0d, ("rpa_flow", "flow_rpa", "rpa"), label="0D RPA flow"),
        phases=phases_0d,
        units=flow_units_0d,
        label="0D RPA flow",
    )
    lpa_0d = _coerce_series(
        _source_value(settled_0d, ("lpa_flow", "flow_lpa", "lpa"), label="0D LPA flow"),
        phases=phases_0d,
        units=flow_units_0d,
        label="0D LPA flow",
    )

    canonical_pressure_units_3d, pressure_factor_3d = _pressure_factor(mpa_3d.units)
    canonical_pressure_units_0d, pressure_factor_0d = _pressure_factor(mpa_0d.units)
    if pressure_units_3d is not None:
        canonical_pressure_units_3d, pressure_factor_3d = _pressure_factor(pressure_units_3d)
    if pressure_units_0d is not None:
        canonical_pressure_units_0d, pressure_factor_0d = _pressure_factor(pressure_units_0d)
    # All pressure metrics are reported in mmHg; differing source aliases are
    # expected, but a missing explicit unit is not.
    del canonical_pressure_units_3d, canonical_pressure_units_0d
    pressure_3d_phases, pressure_3d_values = _periodic_arrays(mpa_3d, label="3D MPA pressure")
    _, pressure_0d_values = _periodic_arrays(mpa_0d, label="0D MPA pressure")
    pressure_3d_values = pressure_3d_values * pressure_factor_3d
    pressure_0d_source = TargetSeries(
        phases=mpa_0d.phases,
        values=tuple(float(value * pressure_factor_0d) for value in mpa_0d.values),
        units="mmHg",
        orientation=mpa_0d.orientation,
    )
    pressure_0d_aligned = _periodic_interpolate(
        pressure_0d_source,
        pressure_3d_phases,
        label="0D MPA pressure",
    )
    pressure_difference = pressure_0d_aligned - pressure_3d_values
    pressure_rms_error = float(np.sqrt(np.mean(np.square(pressure_difference))))
    pressure_scale = max(float(np.ptp(pressure_3d_values)), pressure_floor)
    pressure_nrmse = pressure_rms_error / pressure_scale

    canonical_flow_units_3d, rpa_flow_factor_3d = _flow_factor(rpa_3d.units)
    _, lpa_flow_factor_3d = _flow_factor(lpa_3d.units)
    canonical_flow_units_0d, rpa_flow_factor_0d = _flow_factor(rpa_0d.units)
    _, lpa_flow_factor_0d = _flow_factor(lpa_0d.units)
    if flow_units_3d is not None:
        canonical_flow_units_3d, rpa_flow_factor_3d = _flow_factor(flow_units_3d)
        _, lpa_flow_factor_3d = _flow_factor(flow_units_3d)
    if flow_units_0d is not None:
        canonical_flow_units_0d, rpa_flow_factor_0d = _flow_factor(flow_units_0d)
        _, lpa_flow_factor_0d = _flow_factor(flow_units_0d)
    del canonical_flow_units_3d, canonical_flow_units_0d

    rpa_flow_3d = _periodic_integral(
        TargetSeries(rpa_3d.phases, rpa_3d.values, "cm^3/s", rpa_3d.orientation),
        label="3D RPA flow",
    ) * rpa_flow_factor_3d
    lpa_flow_3d = _periodic_integral(
        TargetSeries(lpa_3d.phases, lpa_3d.values, "cm^3/s", lpa_3d.orientation),
        label="3D LPA flow",
    ) * lpa_flow_factor_3d
    rpa_flow_0d = _periodic_integral(
        TargetSeries(rpa_0d.phases, rpa_0d.values, "cm^3/s", rpa_0d.orientation),
        label="0D RPA flow",
    ) * rpa_flow_factor_0d
    lpa_flow_0d = _periodic_integral(
        TargetSeries(lpa_0d.phases, lpa_0d.values, "cm^3/s", lpa_0d.orientation),
        label="0D LPA flow",
    ) * lpa_flow_factor_0d

    denominator_3d = rpa_flow_3d + lpa_flow_3d
    denominator_0d = rpa_flow_0d + lpa_flow_0d
    for label, denominator in (("3D", denominator_3d), ("0D", denominator_0d)):
        if not np.isfinite(denominator) or denominator <= denominator_floor:
            raise ValueError(
                f"{label} RPA/LPA split denominator must be finite, positive, and "
                f"above {denominator_floor:g}; got {denominator!r}"
            )
    rpa_split_3d = rpa_flow_3d / denominator_3d
    rpa_split_0d = rpa_flow_0d / denominator_0d
    absolute_split_error = abs(rpa_split_0d - rpa_split_3d)

    mpa_target = _target_block(targets, "mpa_pressure")
    split_target = _target_block(targets, "rpa_flow_split")
    pressure_tolerance = _positive_setting(
        mpa_target,
        ("normalized_rms_tolerance", "pressure_tolerance", "tolerance"),
        label="mpa_pressure normalized_rms_tolerance",
    )
    split_tolerance = _positive_setting(
        split_target,
        ("absolute_tolerance", "split_tolerance", "tolerance"),
        label="rpa_flow_split absolute_tolerance",
    )
    pressure_weight = _positive_setting(
        mpa_target,
        ("weight",),
        label="mpa_pressure weight",
        default=1.0,
    )
    split_weight = _positive_setting(
        split_target,
        ("weight",),
        label="rpa_flow_split weight",
        default=1.0,
    )

    pressure_normalized_error = pressure_nrmse / pressure_tolerance
    split_normalized_error = absolute_split_error / split_tolerance
    pressure_gate = TargetGateResult(
        name="mpa_pressure",
        error=float(pressure_nrmse),
        tolerance=pressure_tolerance,
        normalized_error=float(pressure_normalized_error),
        weight=pressure_weight,
        passed=bool(pressure_nrmse <= pressure_tolerance),
    )
    split_gate = TargetGateResult(
        name="rpa_flow_split",
        error=float(absolute_split_error),
        tolerance=split_tolerance,
        normalized_error=float(split_normalized_error),
        weight=split_weight,
        passed=bool(absolute_split_error <= split_tolerance),
    )
    composite_score = pressure_weight * pressure_normalized_error**2 + split_weight * split_normalized_error**2
    if not np.isfinite(composite_score):
        raise ValueError("target composite score is non-finite")

    return PulmonaryTargetEvaluation(
        pressure_nrmse=float(pressure_nrmse),
        rpa_split_3d=float(rpa_split_3d),
        rpa_split_0d=float(rpa_split_0d),
        absolute_split_error=float(absolute_split_error),
        pressure_normalized_error=float(pressure_normalized_error),
        split_normalized_error=float(split_normalized_error),
        composite_score=float(composite_score),
        pressure_gate=pressure_gate,
        split_gate=split_gate,
        pressure_units="mmHg",
        flow_units="cm^3/s",
        pressure_rms_error=float(pressure_rms_error),
        pressure_scale=float(pressure_scale),
        rpa_flow_3d=float(rpa_flow_3d),
        lpa_flow_3d=float(lpa_flow_3d),
        rpa_flow_0d=float(rpa_flow_0d),
        lpa_flow_0d=float(lpa_flow_0d),
    )


# Short alias for coordinators that use the generic target terminology.
evaluate_target_metrics = evaluate_pulmonary_targets


__all__ = [
    "PulmonaryTargetEvaluation",
    "TargetGateResult",
    "TargetSeries",
    "evaluate_pulmonary_targets",
    "extract_target_series",
]
