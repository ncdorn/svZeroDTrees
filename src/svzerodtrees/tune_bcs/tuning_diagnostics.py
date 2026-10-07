"""Proximal-compliance seed preprocessing and post-tuning diagnostics.

Proximal compliance: learned / calibrated full-PA 0D seeds carry the 3D
domain's vessel geometry but no compliance (C = 0).  With
``proximal_compliance: {wall_ehr: Eh/r}`` every seed vessel gets the thin-wall
linear compliance used by the structured trees, C = 3 A L / (2 Eh/r), with
A = mean(inlet_area, outlet_area) from ``geometric_params`` and
L = ``vessel_length`` (cm-g-s units).  Only effectively rigid vessels are
filled (C below RIGID_COMPLIANCE_THRESHOLD); compliance calibrated in a later
iteration is kept, and applying the step twice gives the same seed.

Raw learnedZeroD seeds fold each branch's R and L into its junction outlet and
leave a zero-length ``branch{N}_seg{K}_connectorEL`` vessel that keeps the
branch area.  Given centerline ``branch_lengths`` (cm, by ``BranchId``), such a
vessel takes the branch length minus the lengths of the other vessels on that
branch (the split connectors of multi-outlet junctions), which reproduces the
geometry of the calibrated seed built from the same centerline.  The branch's
R, L, and stenosis coefficient move from the junction outlet onto that vessel,
so the compliance sits on a vessel with its own resistance as in the calibrated
seed (without C the two forms are the same model).  A compliant vessel with
R = L = 0 directly upstream of an IMPEDANCE outlet makes the solver's Newton
iteration fail.

learnedZeroD can also fit a negative junction-outlet inductance (a surrogate
pressure-loss term, not a physical inertance).  Without compliance it is
harmless, but on a compliant vessel C and L < 0 form an unstable element and
the solver's Newton iteration fails on every step.  Vessels that receive
compliance here therefore have negative L set to 0 (``n_negative_l_clamped``).

Diagnostics (``tuning_diagnostics.json``): the outlet pressure and how it was
derived, per-tree size/truncation/resistance/compliance of the exported trees,
total model compliance (trees + proximal), and the stroke-volume /
pulse-pressure bracket for total compliance: regurgitant volume / PP (lower)
to forward stroke volume / PP (upper; ignores systolic runoff).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from ..io.inflow_handler import Inflow
from ..numerics import trapezoid

DYN_PER_MMHG = 1333.2
SEED_WITH_PROXIMAL_COMPLIANCE_FILENAME = "seed_with_proximal_compliance.json"
TUNING_DIAGNOSTICS_FILENAME = "tuning_diagnostics.json"
_BRANCH_ID = re.compile(r"^branch(\d+)_")
_FOLDED_LENGTH_SUFFIX = "_connectorEL"


# A seed vessel counts as rigid when its C is below this value [cm^5/dyn]
# (~1.3e-5 mL/mmHg).  Learned seeds carry ~1e-10 per vessel; physiological
# vessel compliance is ~1e-6 to 1e-4, so calibrated values are never overwritten.
RIGID_COMPLIANCE_THRESHOLD = 1.0e-8


def apply_proximal_compliance(
    seed_payload: Mapping[str, Any],
    wall_ehr: float,
    branch_lengths: Mapping[int, float] | None = None,
) -> tuple[dict, dict]:
    """Fill rigid seed vessels with C = 3 A L / (2 wall_ehr).

    Vessels whose C already exceeds RIGID_COMPLIANCE_THRESHOLD (e.g. compliance
    calibrated against a 3D run in a later iteration) keep their value.  A
    rigid vessel without ``geometric_params`` / ``vessel_length`` is an error,
    except that learnedZeroD connector vessels take their length from
    ``branch_lengths`` (cm by centerline ``BranchId``) when given.
    Returns the updated payload and a summary with per-category counts.
    """
    wall_ehr = float(wall_ehr)
    if not math.isfinite(wall_ehr) or wall_ehr <= 0.0:
        raise ValueError("proximal_compliance.wall_ehr must be > 0")
    payload = json.loads(json.dumps(seed_payload))
    vessels = payload.get("vessels") or []
    if not vessels:
        raise ValueError("proximal_compliance requires a seed with vessels")
    folded_lengths = _folded_branch_lengths(vessels, branch_lengths) if branch_lengths else {}
    missing, applied, kept = [], [], []
    applied_total = kept_total = 0.0
    volume = volume_radius = 0.0
    for vessel in vessels:
        values = vessel.setdefault("zero_d_element_values", {})
        name = str(vessel.get("vessel_name", vessel.get("vessel_id")))
        existing = float(values.get("C") or 0.0)
        geometry = _vessel_area_length(vessel, folded_lengths.get(name))
        if geometry is not None:
            area, length = geometry
            volume += area * length
            volume_radius += area * length * math.sqrt(area / math.pi)
        if existing > RIGID_COMPLIANCE_THRESHOLD:
            kept.append(name)
            kept_total += existing
            continue
        if geometry is None:
            missing.append(name)
            continue
        compliance = 3.0 * area * length / (2.0 * wall_ehr)
        values["C"] = compliance
        applied.append(name)
        applied_total += compliance
    moved = _unfold_junction_values(payload, set(folded_lengths) & set(applied))
    clamped = _clamp_negative_inductance(vessels, set(applied))
    if missing:
        raise ValueError(
            "proximal_compliance needs geometric_params.inlet_area/outlet_area and "
            "vessel_length for every rigid seed vessel; missing for: " + ", ".join(missing[:10])
            + (
                ""
                if branch_lengths
                else " (zero-length learnedZeroD connector vessels take their branch "
                "length from outlet_mapping_centerline)"
            )
        )
    summary = {
        "wall_ehr": wall_ehr,
        "n_vessels": len(vessels),
        "n_applied": len(applied),
        "n_kept_existing": len(kept),
        "n_branch_length_from_centerline": len(folded_lengths),
        "n_junction_values_moved_to_vessel": moved,
        "n_negative_l_clamped": len(clamped),
        "negative_l_clamped": clamped,
        "applied_compliance_ml_per_mmhg": applied_total * DYN_PER_MMHG,
        "kept_compliance_ml_per_mmhg": kept_total * DYN_PER_MMHG,
        "total_compliance_ml_per_mmhg": (applied_total + kept_total) * DYN_PER_MMHG,
        "rigid_threshold_cm5_per_dyn": RIGID_COMPLIANCE_THRESHOLD,
        # A uniform deformable 3D wall over the same vessels has the same total
        # compliance (sum 3 A L r / (2 E h)) when E*h = wall_ehr * r_eff.
        "seed_volume_ml": volume,
        "volume_weighted_radius_cm": (volume_radius / volume) if volume > 0.0 else None,
        "matched_uniform_wall_eh": (wall_ehr * volume_radius / volume) if volume > 0.0 else None,
    }
    return payload, summary


def _folded_branch_lengths(
    vessels: list[Mapping[str, Any]], branch_lengths: Mapping[int, float]
) -> dict[str, float]:
    """Lengths of zero-length learnedZeroD connector vessels, by vessel name.

    Each ``branch{N}_..._connectorEL`` vessel stands for branch N, whose R and
    L learnedZeroD folded into the junction outlet.  It gets the centerline
    branch length minus the lengths of the other vessels on branch N.
    """
    on_branch: dict[int, float] = {}
    folded: dict[str, int] = {}
    for vessel in vessels:
        name = str(vessel.get("vessel_name", ""))
        match = _BRANCH_ID.match(name)
        if match is None:
            continue
        branch_id = int(match.group(1))
        length = float(vessel.get("vessel_length") or 0.0)
        if name.endswith(_FOLDED_LENGTH_SUFFIX) and length == 0.0:
            folded[name] = branch_id
        else:
            on_branch[branch_id] = on_branch.get(branch_id, 0.0) + length
    lengths: dict[str, float] = {}
    for name, branch_id in folded.items():
        if branch_id not in branch_lengths:
            continue
        remaining = float(branch_lengths[branch_id]) - on_branch.get(branch_id, 0.0)
        if math.isfinite(remaining) and remaining > 0.0:
            lengths[name] = remaining
    return lengths


def _unfold_junction_values(payload: dict, vessel_names: set[str]) -> int:
    """Move junction-outlet R, L, stenosis onto the named downstream vessels.

    Only outlets whose vessel still has R = L = 0 move, so repeating the step
    changes nothing.  Returns the number of outlets moved.
    """
    if not vessel_names:
        return 0
    by_id = {
        vessel.get("vessel_id"): vessel
        for vessel in payload.get("vessels") or []
        if str(vessel.get("vessel_name", "")) in vessel_names
    }
    moved = 0
    for junction in payload.get("junctions") or []:
        junction_values = junction.get("junction_values")
        if junction.get("junction_type") != "BloodVesselJunction" or not junction_values:
            continue
        for index, outlet in enumerate(junction.get("outlet_vessels") or []):
            vessel = by_id.get(outlet)
            if vessel is None:
                continue
            values = vessel["zero_d_element_values"]
            if float(values.get("R_poiseuille") or 0.0) != 0.0 or float(values.get("L") or 0.0) != 0.0:
                continue
            for key in ("R_poiseuille", "L", "stenosis_coefficient"):
                series = junction_values.get(key)
                if isinstance(series, list) and index < len(series):
                    values[key] = float(series[index])
                    series[index] = 0.0
            moved += 1
    return moved


def _clamp_negative_inductance(
    vessels: list[dict], vessel_names: set[str]
) -> dict[str, float]:
    """Set L < 0 to 0 on the named (newly compliant) vessels.

    Returns the original negative L by vessel name.
    """
    clamped: dict[str, float] = {}
    for vessel in vessels:
        name = str(vessel.get("vessel_name", vessel.get("vessel_id")))
        if name not in vessel_names:
            continue
        values = vessel["zero_d_element_values"]
        inductance = float(values.get("L") or 0.0)
        if inductance < 0.0:
            clamped[name] = inductance
            values["L"] = 0.0
    return clamped


def _vessel_area_length(
    vessel: Mapping[str, Any], length_override: float | None = None
) -> tuple[float, float] | None:
    """Mean lumen area and length of a seed vessel, or None without geometry."""
    geom = vessel.get("geometric_params") or {}
    try:
        area = 0.5 * (float(geom["inlet_area"]) + float(geom["outlet_area"]))
        length = float(vessel.get("vessel_length") if length_override is None else length_override)
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(area) and area > 0.0 and math.isfinite(length) and length > 0.0):
        return None
    return area, length


def write_seed_with_proximal_compliance(
    seed_path: str | Path,
    output_dir: str | Path,
    wall_ehr: float,
    branch_lengths: Mapping[int, float] | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    updated, summary = apply_proximal_compliance(payload, wall_ehr, branch_lengths)
    out = (Path(output_dir) / SEED_WITH_PROXIMAL_COMPLIANCE_FILENAME).resolve()
    out.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_seed": str(seed_path),
        "seed_with_proximal_compliance": str(out),
        **summary,
    }
    return out, summary


def svpp_bracket(inflow_path: str | Path | None, mpa_p) -> dict[str, Any] | None:
    """Total-compliance bracket [mL/mmHg] from one inflow period and the pulse pressure."""
    if inflow_path is None or not Path(inflow_path).exists():
        return None
    inflow = Inflow.periodic(path=str(inflow_path))
    q, t = inflow.period()
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float)
    forward = float(trapezoid(np.clip(q, 0.0, None), t))
    backflow = float(-trapezoid(np.clip(q, None, 0.0), t))
    pulse = float(mpa_p[0]) - float(mpa_p[1])
    if pulse <= 0.0:
        return None
    return {
        "forward_volume_ml": forward,
        "regurgitant_volume_ml": backflow,
        "regurgitant_fraction": backflow / forward if forward > 0.0 else None,
        "pulse_pressure_mmhg": pulse,
        "lower_ml_per_mmhg": backflow / pulse,
        "upper_ml_per_mmhg": forward / pulse,
    }


def summarize_trees(tree_diagnostics: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    trees = [dict(v) for v in (tree_diagnostics or {}).values() if "error" not in v]
    if not trees:
        return {"n_trees": 0}
    conductance = sum(1.0 / t["dc_resistance"] for t in trees if t.get("dc_resistance"))
    compliance = sum(float(t["static_compliance"]) for t in trees)
    return {
        "n_trees": len(trees),
        "n_truncated": sum(1 for t in trees if t.get("truncated")),
        "truncated_trees": [t.get("cap", "?") for t in trees if t.get("truncated")],
        "parallel_dc_resistance": (1.0 / conductance) if conductance > 0.0 else None,
        "static_compliance_ml_per_mmhg": compliance * DYN_PER_MMHG,
        "ehr_min": min(t["ehr_min"] for t in trees),
        "ehr_max": max(t["ehr_max"] for t in trees),
        "max_nodes": max(int(t.get("max_nodes") or 0) for t in trees),
    }


def build_tuning_diagnostics(
    *,
    targets,
    tuning: Mapping[str, Any],
    tree_diagnostics: Mapping[str, Mapping[str, Any]] | None,
    proximal_summary: Mapping[str, Any] | None,
    inflow_path: str | Path | None,
    polish_summary: Mapping[str, Any] | None = None,
    published_fit: Mapping[str, Any] | None = None,
    optimizer_fit: Mapping[str, Any] | None = None,
    bound_flags: list | None = None,
) -> dict[str, Any]:
    trees = summarize_trees(tree_diagnostics)
    proximal_ml = float(proximal_summary["total_compliance_ml_per_mmhg"]) if proximal_summary else 0.0
    model_total = None
    if trees.get("n_trees"):
        model_total = trees["static_compliance_ml_per_mmhg"] + proximal_ml
    bracket = svpp_bracket(inflow_path, targets.mpa_p)
    in_bracket = None
    if bracket is not None and model_total is not None:
        in_bracket = bool(bracket["lower_ml_per_mmhg"] <= model_total <= bracket["upper_ml_per_mmhg"])
    return {
        "outlet_pressure": {
            "pd_mmhg": float(targets.wedge_p),
            "policy": getattr(targets, "wedge_pressure_policy", tuning.get("wedge_pressure_policy")),
            "measured_wedge_mmhg": _finite_or_none(getattr(targets, "measured_wedge_p", None)),
            "precapillary_fraction": getattr(targets, "precapillary_fraction", None),
            "diastolic_offset_mmhg": getattr(targets, "diastolic_offset_mmhg", None),
            "mpa_targets_mmhg": [float(v) for v in targets.mpa_p],
        },
        "objective": tuning.get("objective") or {"type": "relative"},
        "keep_diastolic_target": bool(tuning.get("keep_diastolic_target", False)),
        "trees": trees,
        "per_tree": dict(tree_diagnostics or {}),
        "proximal_compliance": dict(proximal_summary) if proximal_summary else None,
        "model_total_compliance_ml_per_mmhg": model_total,
        "svpp_bracket": bracket,
        "compliance_in_svpp_bracket": in_bracket,
        "polish": dict(polish_summary) if polish_summary else None,
        "optimizer_fit": dict(optimizer_fit) if optimizer_fit else None,
        "published_fit": _with_chi2(published_fit, targets, tuning),
        "inflow_consistency": _inflow_consistency(published_fit, inflow_path),
        "leaf_resistance": _leaf_resistance_summary(tuning, tree_diagnostics, published_fit, targets),
        "parameters_at_bounds": list(bound_flags or []),
        "svzerodtrees_version": _package_version(),
        "peak_rss_gb": _peak_rss_gb(),
    }


def _with_chi2(fit, targets, tuning):
    """Attach the sigma-scaled chi-square of a fit (objective sigmas or 2 mmHg / 0.02)."""
    if not fit or "mpa_pressure_mmhg" not in fit:
        return dict(fit) if fit else None
    objective = tuning.get("objective") or {}
    sigma_p = float(objective.get("pressure_sigma_mmhg") or 2.0)
    sigma_s = float(objective.get("split_sigma") or 0.02)
    errors = [float(m) - float(t) for m, t in zip(fit["mpa_pressure_mmhg"], targets.mpa_p)]
    split_error = float(fit["rpa_split"]) - float(targets.rpa_split)
    chi2 = sum((e / sigma_p) ** 2 for e in errors) + (split_error / sigma_s) ** 2
    return {**dict(fit), "errors_mmhg": errors, "split_error": split_error, "chi2": chi2,
            "sigmas": {"pressure_mmhg": sigma_p, "split": sigma_s}}


def _leaf_resistance_summary(tuning, tree_diagnostics, published_fit, targets):
    """Per-leaf capillary + venous resistance and the model's PVR partition.

    The capillary + venous share of the whole MPA-to-outlet mean pressure drop
    is the flow-weighted share carried by the leaf resistances:
    sum_i Q_i f_i (P_i - Pd) / (Q (P_mpa - Pd)), with P_i, Q_i the published
    model's mean pressure and flow at outlet i and f_i that tree's leaf share.
    Dong et al. 2021 (after Raj & Chen 1986) assign 33.2% of PVR to the
    arteries (MPA to pre-capillary arterioles), i.e. 66.8% capillary + venous.
    """
    leaf = tuning.get("leaf_resistance")
    if not leaf:
        return None
    trees = {k: v for k, v in (tree_diagnostics or {}).items() if isinstance(v, Mapping) and v.get("leaf_resistance")}
    resistances = [float(v["terminal_resistance"]) for v in trees.values()]
    summary = {
        "downstream_fraction": float(leaf["downstream_fraction"]),
        "terminal_resistance_min": min(resistances) if resistances else None,
        "terminal_resistance_max": max(resistances) if resistances else None,
        "model_capillary_venous_share_of_pvr": None,
        "model_arterial_share_of_pvr": None,
    }
    outlets = (published_fit or {}).get("outlet_means") or {}
    pressures = (published_fit or {}).get("mpa_pressure_mmhg")
    if not outlets or not pressures:
        return summary
    pd = float(targets.wedge_p)
    drop = float(pressures[2]) - pd
    total_flow = sum(float(o["flow"]) for o in outlets.values())
    if drop <= 0.0 or total_flow <= 0.0:
        return summary
    weighted = 0.0
    for bc_name, outlet in outlets.items():
        tree = trees.get(bc_name)
        if tree is None:
            return summary  # shared trees or missing diagnostics: no per-outlet share
        weighted += float(outlet["flow"]) * float(tree["leaf_resistance"]["downstream_fraction"]) * (
            float(outlet["pressure_mmhg"]) - pd
        )
    share = weighted / (total_flow * drop)
    summary["model_capillary_venous_share_of_pvr"] = share
    summary["model_arterial_share_of_pvr"] = 1.0 - share
    return summary


# Relative mean-flow difference above which the published model and the
# optimizer are judged to have used different inflows.
INFLOW_MISMATCH_TOLERANCE = 0.01


def _inflow_consistency(published_fit, inflow_path):
    """Mean flow of the exported (published) model vs the inflow.csv mean.

    The optimizer runs at the inflow.csv mean flow; the exported model keeps
    the seed's inflow.  A mismatch mixes an inflow difference into the
    published-vs-optimizer gap and into the published fit itself.
    """
    published = (published_fit or {}).get("inflow_mean_flow")
    if published is None or inflow_path is None or not Path(inflow_path).exists():
        return None
    from ..io.inflow_handler import mean_flow_from_path

    reference = float(mean_flow_from_path(str(inflow_path)))
    rel = abs(float(published) - reference) / abs(reference) if reference else None
    return {
        "published_model_mean_flow": float(published),
        "inflow_csv_mean_flow": reference,
        "rel_difference": rel,
        "consistent": None if rel is None else bool(rel <= INFLOW_MISMATCH_TOLERANCE),
    }


def _package_version():
    try:
        from importlib.metadata import version

        return version("svzerodtrees")
    except Exception:
        return None


def _peak_rss_gb():
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)
    except Exception:
        return None


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
