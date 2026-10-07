"""Impedance tuning objective definitions.

``relative`` (default, historical): weighted relative squared error of MPA
systolic/diastolic/mean pressure and RPA split, x100, with sys/dia/mean
weights 1.5/1.0/1.2.  Each error is divided by its own target, so a small
target (e.g. a 3 mmHg diastolic) dominates the loss.

``likelihood``: Gaussian negative log-likelihood up to a constant,
sum(((model - target) / sigma)^2), with one measurement standard deviation for
the catheter pressures and one for the flow split.  Every target is weighted
by its measurement error, so the loss is a chi-square statistic and the
optimizer does not trade large errors on one target for small ones on another
merely because of the targets' magnitudes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

OBJECTIVE_TYPES = ("relative", "likelihood")
DEFAULT_PRESSURE_SIGMA_MMHG = 2.0
DEFAULT_SPLIT_SIGMA = 0.02
DEFAULT_TARGET_SIGMA = 1.0


@dataclass(frozen=True)
class TuningObjective:
    type: str = "relative"
    pressure_sigma_mmhg: float = DEFAULT_PRESSURE_SIGMA_MMHG
    split_sigma: float = DEFAULT_SPLIT_SIGMA
    # Likelihood only: the Nelder-Mead target stop fires when every weighted
    # target is within target_sigma standard deviations (None disables it).
    # The relative stopping.target_tolerance is not used with the likelihood.
    target_sigma: float | None = DEFAULT_TARGET_SIGMA

    @property
    def is_likelihood(self) -> bool:
        return self.type == "likelihood"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "pressure_sigma_mmhg": self.pressure_sigma_mmhg,
            "split_sigma": self.split_sigma,
            "target_sigma": self.target_sigma,
        }


def resolve_tuning_objective(
    payload: Mapping[str, Any] | TuningObjective | None,
    *,
    label: str = "objective",
) -> TuningObjective:
    """Validate an objective mapping; None means the historical relative loss."""
    if payload is None:
        return TuningObjective()
    if isinstance(payload, TuningObjective):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(payload) - {"type", "pressure_sigma_mmhg", "split_sigma", "target_sigma"}
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    kind = str(payload.get("type") or "relative").strip().lower()
    if kind not in OBJECTIVE_TYPES:
        raise ValueError(f"{label}.type must be one of " + "|".join(OBJECTIVE_TYPES))

    def _sigma(key: str, default: float) -> float:
        value = payload.get(key)
        value = default if value is None else float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}.{key} must be > 0")
        return value

    if "target_sigma" in payload and payload["target_sigma"] is None:
        target_sigma = None
    else:
        target_sigma = _sigma("target_sigma", DEFAULT_TARGET_SIGMA)
    return TuningObjective(
        type=kind,
        pressure_sigma_mmhg=_sigma("pressure_sigma_mmhg", DEFAULT_PRESSURE_SIGMA_MMHG),
        split_sigma=_sigma("split_sigma", DEFAULT_SPLIT_SIGMA),
        target_sigma=target_sigma,
    )
