"""Tied stiffness multiplier s for full-PA impedance tuning.

One tune-space parameter, ``stiffness_scale`` (s), stiffens the proximal wall
and the structured trees' large vessels together:

    proximal Eh/r = s * proximal_compliance.wall_ehr   (every seed vessel)
    comp.lpa.k3 = comp.rpa.k3 = s * proximal_compliance.tied_k3_ref

The TST-STAN-3 identifiability study
(``examples/adaptation/tst3-stiffness-identifiability/FINDINGS.md``) found that
under the full-PA objective only the combined stiffness (total compliance) is
identified, not how it splits between the proximal vessels and the trees, so
the two are tuned along one direction instead of as two free parameters.

Because the proximal C depends on s, it is re-applied on every objective
evaluation: ``apply_proximal_compliance`` with ``overwrite=True`` on the
reference seed (the seed written at s = 1) gives each vessel's C at the
candidate's Eh/r, and those values replace the vessel C of the evaluation's
model copy.  Overwriting also replaces C kept from an earlier iteration's
seed (R-only calibration preserves it), so every iteration re-tunes s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .pipeline_options import STIFFNESS_SCALE_PARAM
from .tuning_diagnostics import apply_proximal_compliance

TIED_K3_NAMES = ("comp.lpa.k3", "comp.rpa.k3")
# Native-parameter key that records the proximal Eh/r of a candidate.
PROXIMAL_WALL_EHR_PARAM = "proximal.wall_ehr"


@dataclass(frozen=True)
class TiedStiffness:
    """Maps s to the proximal wall Eh/r, the tree k3, and per-vessel proximal C."""

    wall_ehr_ref: float
    k3_ref: float
    # Seed the tuning model was built from (proximal C applied at s = 1).
    seed_payload: Mapping[str, Any] = field(repr=False)
    # Centerline branch lengths [cm] for zero-length learnedZeroD connectors.
    branch_lengths: Mapping[int, float] | None = field(default=None, repr=False)

    def __post_init__(self):
        for label, value in (("wall_ehr_ref", self.wall_ehr_ref), ("k3_ref", self.k3_ref)):
            if not float(value) > 0.0:
                raise ValueError(f"TiedStiffness.{label} must be > 0")

    @staticmethod
    def scale(params: Mapping[str, float]) -> float:
        if STIFFNESS_SCALE_PARAM not in params:
            raise ValueError(f"tied stiffness needs the '{STIFFNESS_SCALE_PARAM}' parameter")
        s = float(params[STIFFNESS_SCALE_PARAM])
        if not s > 0.0:
            raise ValueError(f"{STIFFNESS_SCALE_PARAM} must be > 0, got {s}")
        return s

    def wall_ehr(self, s: float) -> float:
        return float(s) * float(self.wall_ehr_ref)

    def k3(self, s: float) -> float:
        return float(s) * float(self.k3_ref)

    def resolve_params(self, params: Mapping[str, float]) -> dict[str, float]:
        """Native parameters with the tree k3 and proximal Eh/r derived from s."""
        s = self.scale(params)
        resolved = dict(params)
        for name in TIED_K3_NAMES:
            resolved[name] = self.k3(s)
        resolved[PROXIMAL_WALL_EHR_PARAM] = self.wall_ehr(s)
        return resolved

    def proximal_compliance(self, s: float) -> dict[str, float]:
        """C [cm^5/dyn] of every seed vessel at Eh/r = s * wall_ehr_ref, by name."""
        updated, _ = apply_proximal_compliance(
            self.seed_payload, self.wall_ehr(s), self.branch_lengths, overwrite=True
        )
        return {
            str(vessel.get("vessel_name", vessel.get("vessel_id"))): float(
                vessel["zero_d_element_values"]["C"]
            )
            for vessel in updated.get("vessels") or []
        }

    def apply(self, config_handler, s: float) -> dict[str, float]:
        """Set the proximal C of ``config_handler``'s seed vessels for this s."""
        compliance = self.proximal_compliance(s)
        vessels = {vessel.name: vessel for vessel in config_handler.vessel_map.values()}
        missing = sorted(set(compliance) - set(vessels))
        if missing:
            raise ValueError(
                "tied stiffness: tuning model lacks seed vessels " + ", ".join(missing[:10])
            )
        for name, value in compliance.items():
            vessels[name].C = value
        return compliance
