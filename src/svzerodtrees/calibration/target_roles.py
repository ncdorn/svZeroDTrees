"""MPA, LPA, and RPA target roles for target-focused full-PA calibration.

``target_focused`` calibration needs explicit MPA, LPA, and RPA vessel roles.
For a tuned full-PA model they follow from its topology plus the outlet-to-cap
pairing that tuning already validated (``outlet_cap_mapping.json``):

- the MPA is the vessel fed by the inflow boundary condition;
- the LPA and RPA are the two outlets of the junction at the MPA's distal end;
- each of those subtrees must end only in outlets whose caps lie on one side
  (``pairs[].side``), and the two sides must be ``lpa`` and ``rpa``.

Any other topology or a mixed subtree is an error, never a guess.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

MPA_INTERFACE = "external_upstream"
# LPA/RPA flow is read where each branch leaves the MPA junction.
SPLIT_INTERFACE = "upstream"


def pa_target_roles(
    zerod_config: Mapping[str, Any],
    outlet_cap_mapping: Mapping[str, Any],
) -> dict[str, str]:
    """Return ``{"mpa_vessel", "lpa_vessel", "rpa_vessel"}`` vessel names."""
    vessels = list(zerod_config.get("vessels") or [])
    junctions = list(zerod_config.get("junctions") or [])
    by_id = {int(v["vessel_id"]): v for v in vessels}
    inlet_junction: dict[int, Mapping[str, Any]] = {}
    for junction in junctions:
        for vessel_id in junction.get("inlet_vessels") or []:
            inlet_junction[int(vessel_id)] = junction

    inflow = [
        v for v in vessels if (v.get("boundary_conditions") or {}).get("inlet")
    ]
    if len(inflow) != 1:
        raise ValueError(f"expected one inflow vessel, found {len(inflow)}")
    mpa = inflow[0]
    mpa_junction = inlet_junction.get(int(mpa["vessel_id"]))
    if mpa_junction is None:
        raise ValueError("the MPA vessel does not end in a junction")
    branches = [int(i) for i in mpa_junction.get("outlet_vessels") or []]
    if len(branches) != 2:
        raise ValueError(
            f"the MPA junction {mpa_junction.get('junction_name')} must have two "
            f"outlets (LPA and RPA); found {len(branches)}"
        )

    sides_by_bc: dict[str, str] = {}
    for pair in outlet_cap_mapping.get("pairs") or []:
        side = str(pair.get("side") or "").strip().lower()
        if side not in {"lpa", "rpa"}:
            raise ValueError(
                f"outlet mapping pair for {pair.get('bc_name')} has no lpa/rpa side"
            )
        sides_by_bc[str(pair["bc_name"])] = side

    roles: dict[str, str] = {"mpa_vessel": str(mpa["vessel_name"])}
    for branch_id in branches:
        sides = set()
        stack, seen = [branch_id], set()
        while stack:
            vessel_id = stack.pop()
            if vessel_id in seen:
                continue
            seen.add(vessel_id)
            outlet_bc = (by_id[vessel_id].get("boundary_conditions") or {}).get("outlet")
            if outlet_bc:
                if outlet_bc not in sides_by_bc:
                    raise ValueError(f"outlet {outlet_bc} is missing from the outlet mapping")
                sides.add(sides_by_bc[outlet_bc])
            junction = inlet_junction.get(vessel_id)
            if junction is not None:
                stack.extend(int(i) for i in junction.get("outlet_vessels") or [])
        name = str(by_id[branch_id]["vessel_name"])
        if len(sides) != 1:
            raise ValueError(
                f"branch {name} drains outlets on sides {sorted(sides)}; expected one side"
            )
        side = sides.pop()
        key = f"{side}_vessel"
        if key in roles:
            raise ValueError(f"both MPA junction outlets drain the {side.upper()}")
        roles[key] = name
    return roles


def full_pa_calibration_targets(
    zerod_config_path: str | Path,
    outlet_cap_mapping_path: str | Path,
    *,
    mpa_normalized_rms_tolerance: float = 0.05,
    rpa_split_absolute_tolerance: float = 0.02,
    require_improvement_over_baseline: bool = True,
    gate_policy: str = "absolute",
) -> dict[str, Any]:
    """``calibration.targets`` block for a tuned full-PA model."""
    zerod_config = json.loads(Path(zerod_config_path).read_text(encoding="utf-8"))
    mapping = json.loads(Path(outlet_cap_mapping_path).read_text(encoding="utf-8"))
    roles = pa_target_roles(zerod_config, mapping)
    return {
        "mpa_pressure": {
            "vessel": roles["mpa_vessel"],
            "interface": MPA_INTERFACE,
            "weight": 1.0,
            "normalized_rms_tolerance": float(mpa_normalized_rms_tolerance),
        },
        "rpa_flow_split": {
            "rpa_vessel": roles["rpa_vessel"],
            "lpa_vessel": roles["lpa_vessel"],
            "interface": SPLIT_INTERFACE,
            "weight": 1.0,
            "absolute_tolerance": float(rpa_split_absolute_tolerance),
        },
        "require_improvement_over_baseline": bool(require_improvement_over_baseline),
        "gate_policy": str(gate_policy),
    }
