"""Validation of optional full-PA impedance tuning controls.

Shared by the YAML schema (``config.py``) and the iteration service
(``tuning/iteration.py``) so both accept exactly the same contract.  Every
control defaults to the historical behavior when omitted.

- ``proximal_compliance``: ``{wall_ehr: <dyn/cm^2>}`` gives each seed vessel a
  thin-wall linear compliance C = 3 A L / (2 Eh/r).  With
  ``tied_k3_ref: <dyn/cm^2>`` and a ``stiffness_scale`` tune-space parameter s,
  every evaluation uses Eh/r = s * wall_ehr on every seed vessel and tree
  k3 = s * tied_k3_ref on both sides (see ``tied_stiffness.py``).
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
- ``periodic_evaluation``: ``{min_cycles, max_cycles, mean_pressure_tol_mmhg}``
  or ``{enabled: false}``; how many cardiac cycles each full-PA evaluation
  runs (see ``periodicity.py``).  Omitted means the defaults (6 / 24 /
  0.1 mmHg), not the seed's cycle count.
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
    "periodic_evaluation",
    "tune_space",
)
# The iteration service also accepts the deprecated legacy mapping flag.
# Callers (svzt-agent) check their rendered keys against this set so a
# cluster install older than the caller fails loudly instead of silently
# ignoring new controls.
SUPPORTED_IMPEDANCE_KEYS = frozenset(IMPEDANCE_CONFIG_KEYS) | {"allow_ordered_outlet_mapping"}

# Tied stiffness multiplier s: proximal Eh/r = s * wall_ehr, tree k3 = s * tied_k3_ref.
STIFFNESS_SCALE_PARAM = "stiffness_scale"

# Tune-space parameter names the impedance tuner reads.
_TUNE_SPACE_NAME_PATTERNS = (
    re.compile(r"^(lpa|rpa)\.(xi|eta_sym|alpha|beta|diameter|inductance)$"),
    re.compile(r"^comp\.(lpa|rpa)\.(k1|k2|k3|C)$"),
    re.compile(r"^(lrr|d_min|stiffness_scale)$"),
)
_TIED_K3_NAMES = ("comp.lpa.k3", "comp.rpa.k3")


def validate_tune_space_names(names, *, label: str = "tune_space") -> None:
    """Raise on parameter names the tuner would silently ignore."""
    unknown = sorted({str(n) for n in names if not any(p.match(str(n)) for p in _TUNE_SPACE_NAME_PATTERNS)})
    if unknown:
        raise ValueError(
            f"{label} has parameter names the impedance tuner does not use: {unknown}. "
            "Known: {lpa,rpa}.{xi,eta_sym,alpha,beta,diameter,inductance}, "
            "comp.{lpa,rpa}.{k1,k2,k3,C}, lrr, d_min, stiffness_scale"
        )


def validate_stiffness_scale(
    proximal: Mapping[str, Any] | None,
    *,
    free: Mapping[str, float],
    fixed: Mapping[str, float],
    tied_names,
    compliance_model: str,
    label: str = "impedance",
) -> None:
    """Cross-check the tied stiffness multiplier against the tune space.

    ``free`` maps free names to their lower bound, ``fixed`` maps fixed names
    to their value.  ``stiffness_scale`` (free or fixed) and
    ``proximal_compliance.tied_k3_ref`` require each other; the tree k3 is
    then derived from s, so it must not also be a tune-space parameter.
    """
    k3_ref = None if proximal is None else proximal.get("tied_k3_ref")
    tied_names = {str(n) for n in tied_names}
    present = STIFFNESS_SCALE_PARAM in free or STIFFNESS_SCALE_PARAM in fixed
    if STIFFNESS_SCALE_PARAM in tied_names:
        raise ValueError(f"{label} tune_space: {STIFFNESS_SCALE_PARAM} must be free or fixed, not tied")
    if not present and k3_ref is None:
        return
    if not present:
        raise ValueError(
            f"{label}.proximal_compliance.tied_k3_ref requires a '{STIFFNESS_SCALE_PARAM}' "
            "tune_space parameter (free or fixed)"
        )
    if k3_ref is None:
        raise ValueError(
            f"{label} tune_space parameter '{STIFFNESS_SCALE_PARAM}' requires "
            "proximal_compliance.tied_k3_ref"
        )
    derived = sorted((set(free) | set(fixed) | tied_names) & set(_TIED_K3_NAMES))
    if derived:
        raise ValueError(
            f"{label} tune_space must not set {derived}: with '{STIFFNESS_SCALE_PARAM}' "
            "the tree k3 is s * proximal_compliance.tied_k3_ref"
        )
    if str(compliance_model).strip().lower() != "olufsen":
        raise ValueError(f"{label} '{STIFFNESS_SCALE_PARAM}' requires compliance_model='olufsen'")
    lowest = free.get(STIFFNESS_SCALE_PARAM, fixed.get(STIFFNESS_SCALE_PARAM))
    if not float(lowest) > 0.0:
        raise ValueError(f"{label} '{STIFFNESS_SCALE_PARAM}' must be > 0 (lower bound or fixed value)")


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
    unknown = set(payload) - {"wall_ehr", "tied_k3_ref"}
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    if payload.get("wall_ehr") is None:
        raise ValueError(f"{label}.wall_ehr is required")
    wall_ehr = _finite(payload["wall_ehr"], f"{label}.wall_ehr")
    if wall_ehr <= 0.0:
        raise ValueError(f"{label}.wall_ehr must be > 0")
    resolved: dict[str, Any] = {"wall_ehr": wall_ehr}
    if payload.get("tied_k3_ref") is not None:
        k3_ref = _finite(payload["tied_k3_ref"], f"{label}.tied_k3_ref")
        if k3_ref <= 0.0:
            raise ValueError(f"{label}.tied_k3_ref must be > 0")
        resolved["tied_k3_ref"] = k3_ref
    return resolved


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
