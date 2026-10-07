"""Validation of optional full-PA impedance tuning controls.

Shared by the YAML schema (``config.py``) and the iteration service
(``tuning/iteration.py``) so both accept exactly the same contract.  Every
control defaults to the historical behavior when omitted.

- ``proximal_compliance``: ``{wall_ehr: <dyn/cm^2>}`` gives each seed vessel a
  thin-wall linear compliance C = 3 A L / (2 Eh/r).
- ``polish``: ``{maxfev: <int>, initial_simplex_step: <float, optional>}`` re-tunes with the final (per-cap) trees,
  seeded from the shared-tree optimum, when an ``objective_tree_policy``
  differs from the final policy.
- ``tree_max_nodes``: node budget for every structured tree built during
  tuning and final assignment (None keeps the library default).
- ``precapillary_fraction`` / ``diastolic_offset_mmhg``: parameters of the
  corresponding ``wedge_pressure_policy`` values.
- ``leaf_resistance``: ``{downstream_fraction: f}`` gives every tree leaf a
  resistance to the outlet pressure carrying the fraction ``f`` of the tree's
  DC resistance (the capillary + venous bed).  Requires
  ``wedge_pressure_policy: measured`` (outlet at PCWP) so the capillary and
  venous drop is not counted twice.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import re

from .clinical_targets import DEFAULT_DIASTOLIC_OFFSET_MMHG, DEFAULT_PRECAPILLARY_FRACTION

# Keys of ``bcs.impedance`` (YAML) and of the iteration-service mapping.
IMPEDANCE_CONFIG_KEYS = (
    "tuning_model",
    "solver",
    "nm_iter",
    "n_procs",
    "grid_search_init",
    "d_min",
    "use_mean",
    "specify_diameter",
    "rescale_inflow",
    "convert_to_cm",
    "compliance_model",
    "diameter_scale",
    "diameter_std_cap",
    "outlet_mapping_mode",
    "outlet_mapping",
    "outlet_mapping_centerline",
    "objective_tree_policy",
    "wedge_pressure_policy",
    "precapillary_fraction",
    "diastolic_offset_mmhg",
    "keep_diastolic_target",
    "objective",
    "proximal_compliance",
    "tree_max_nodes",
    "polish",
    "leaf_resistance",
    "stopping",
    "tune_space",
)
# The iteration service also accepts the deprecated legacy mapping flag.
# Callers (svzt-agent) check their rendered keys against this set so a
# cluster install older than the caller fails loudly instead of silently
# ignoring new controls.
SUPPORTED_IMPEDANCE_KEYS = frozenset(IMPEDANCE_CONFIG_KEYS) | {"allow_ordered_outlet_mapping"}

# Tune-space parameter names the impedance tuner reads.
_TUNE_SPACE_NAME_PATTERNS = (
    re.compile(r"^(lpa|rpa)\.(xi|eta_sym|alpha|beta|diameter|inductance)$"),
    re.compile(r"^comp\.(lpa|rpa)\.(k1|k2|k3|C)$"),
    re.compile(r"^(lrr|d_min)$"),
)


def validate_tune_space_names(names, *, label: str = "tune_space") -> None:
    """Raise on parameter names the tuner would silently ignore."""
    unknown = sorted({str(n) for n in names if not any(p.match(str(n)) for p in _TUNE_SPACE_NAME_PATTERNS)})
    if unknown:
        raise ValueError(
            f"{label} has parameter names the impedance tuner does not use: {unknown}. "
            "Known: {lpa,rpa}.{xi,eta_sym,alpha,beta,diameter,inductance}, "
            "comp.{lpa,rpa}.{k1,k2,k3,C}, lrr, d_min"
        )


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def resolve_proximal_compliance(payload: Any, *, label: str = "proximal_compliance") -> dict | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    unknown = set(payload) - {"wall_ehr"}
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    if payload.get("wall_ehr") is None:
        raise ValueError(f"{label}.wall_ehr is required")
    wall_ehr = _finite(payload["wall_ehr"], f"{label}.wall_ehr")
    if wall_ehr <= 0.0:
        raise ValueError(f"{label}.wall_ehr must be > 0")
    return {"wall_ehr": wall_ehr}


def resolve_leaf_resistance(
    payload: Any,
    *,
    tuning_model: str,
    wedge_pressure_policy: str,
    label: str = "leaf_resistance",
) -> dict | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    unknown = set(payload) - {"downstream_fraction"}
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    if payload.get("downstream_fraction") is None:
        raise ValueError(f"{label}.downstream_fraction is required")
    fraction = _finite(payload["downstream_fraction"], f"{label}.downstream_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{label}.downstream_fraction must be in (0, 1)")
    if str(tuning_model).strip().lower() != "full_pa":
        raise ValueError(f"{label} is supported only for tuning_model='full_pa'")
    if str(wedge_pressure_policy).strip().lower() != "measured":
        raise ValueError(
            f"{label} requires wedge_pressure_policy 'measured': the leaf resistance "
            "carries the capillary and venous pressure drop down to the measured PCWP"
        )
    return {"downstream_fraction": fraction}


def resolve_polish(
    payload: Any,
    *,
    tuning_model: str,
    objective_tree_policy: Mapping[str, Any] | None,
    label: str = "polish",
) -> dict | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    unknown = set(payload) - {"maxfev", "initial_simplex_step"}
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    if str(tuning_model).strip().lower() != "full_pa":
        raise ValueError(f"{label} is supported only for tuning_model='full_pa'")
    if objective_tree_policy is None:
        raise ValueError(
            f"{label} requires an objective_tree_policy: without one the optimizer "
            "already uses the final trees and a polish would repeat the same fit"
        )
    maxfev = int(payload.get("maxfev") or 100)
    if maxfev <= 0:
        raise ValueError(f"{label}.maxfev must be > 0")
    resolved: dict[str, Any] = {"maxfev": maxfev}
    if payload.get("initial_simplex_step") is not None:
        step = _finite(payload["initial_simplex_step"], f"{label}.initial_simplex_step")
        if not 0.0 < step <= 0.5:
            raise ValueError(f"{label}.initial_simplex_step must be in (0, 0.5]")
        resolved["initial_simplex_step"] = step
    return resolved


def resolve_tree_max_nodes(value: Any, *, label: str = "tree_max_nodes") -> int | None:
    if value is None:
        return None
    nodes = int(value)
    if nodes <= 0:
        raise ValueError(f"{label} must be > 0")
    return nodes


def resolve_outlet_parameters(
    precapillary_fraction: Any,
    diastolic_offset_mmhg: Any,
    *,
    label: str = "impedance",
) -> tuple[float, float]:
    fraction = (
        DEFAULT_PRECAPILLARY_FRACTION
        if precapillary_fraction is None
        else _finite(precapillary_fraction, f"{label}.precapillary_fraction")
    )
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"{label}.precapillary_fraction must be in [0, 1)")
    offset = (
        DEFAULT_DIASTOLIC_OFFSET_MMHG
        if diastolic_offset_mmhg is None
        else _finite(diastolic_offset_mmhg, f"{label}.diastolic_offset_mmhg")
    )
    if offset < 0.0:
        raise ValueError(f"{label}.diastolic_offset_mmhg must be >= 0")
    return fraction, offset
