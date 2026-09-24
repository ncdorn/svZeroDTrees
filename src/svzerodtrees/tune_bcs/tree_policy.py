"""Tree-generation policy used inside the full_pa tuning objective.

A full_pa run has two tree-generation policies:

* the *final* policy (top-level ``use_mean``, ``diameter_scale``,
  ``diameter_std_cap``) builds the published ``tuned_zerod_config`` and
  therefore the 3D outlet BCs;
* the *objective* policy (``objective_tree_policy``) builds candidate trees
  for every optimizer evaluation.

When ``objective_tree_policy`` is omitted the objective uses the final policy
(historical behavior).  Supplying it lets the tuning loop use cheap shared
trees while the published config keeps per-outlet trees.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "OBJECTIVE_TREE_POLICY_KEYS",
    "REFERENCE_DIAMETER_MODES",
    "resolve_objective_tree_policy",
]

REFERENCE_DIAMETER_MODES = ("arithmetic_mean", "conductance_matched")
OBJECTIVE_TREE_POLICY_KEYS = (
    "use_mean",
    "diameter_scale",
    "diameter_std_cap",
    "reference_diameter",
)


def resolve_objective_tree_policy(
    raw: Mapping[str, Any] | None,
    *,
    tuning_model: str,
    use_mean: bool,
    diameter_scale: float,
    diameter_std_cap: float | None,
    free_param_names: Iterable[str] = (),
    label: str = "objective_tree_policy",
) -> dict[str, Any] | None:
    """Validate ``objective_tree_policy`` and fill omitted fields.

    ``use_mean``, ``diameter_scale`` and ``diameter_std_cap`` default to the
    final policy.  ``reference_diameter`` defaults to ``arithmetic_mean``.

    ``reference_diameter='conductance_matched'`` builds one shared tree per
    side at the diameter whose DC conductance, repeated once per outlet,
    equals that of the per-outlet trees defined by this policy's
    ``diameter_scale``/``diameter_std_cap``.  It requires ``use_mean=True``
    and a final policy that builds per-outlet trees, and it cannot be
    combined with a free ``lpa.diameter``/``rpa.diameter`` tune parameter
    because it replaces that diameter.

    Returns ``None`` when ``raw`` is ``None``.
    """

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = sorted(set(raw) - set(OBJECTIVE_TREE_POLICY_KEYS))
    if unknown:
        raise ValueError(
            f"{label} has unknown keys {unknown}; allowed keys are "
            + ", ".join(OBJECTIVE_TREE_POLICY_KEYS)
        )
    if str(tuning_model).strip().lower() != "full_pa":
        raise ValueError(f"{label} is supported only for tuning_model='full_pa'")

    def _field(key: str, default: Any) -> Any:
        value = raw.get(key)
        return default if value is None else value

    policy_diameter_scale = float(_field("diameter_scale", diameter_scale))
    if not np.isfinite(policy_diameter_scale) or policy_diameter_scale < 0.0:
        raise ValueError(f"{label}.diameter_scale must be finite and >= 0")
    std_cap_value = raw["diameter_std_cap"] if "diameter_std_cap" in raw else diameter_std_cap
    policy_std_cap = None if std_cap_value is None else float(std_cap_value)
    if policy_std_cap is not None and (not np.isfinite(policy_std_cap) or policy_std_cap < 0.0):
        raise ValueError(f"{label}.diameter_std_cap must be finite and >= 0")

    reference = str(_field("reference_diameter", "arithmetic_mean")).strip().lower()
    if reference not in REFERENCE_DIAMETER_MODES:
        raise ValueError(
            f"{label}.reference_diameter must be one of "
            + "|".join(REFERENCE_DIAMETER_MODES)
        )
    policy_use_mean = bool(_field("use_mean", use_mean))

    if reference == "conductance_matched":
        if not policy_use_mean:
            raise ValueError(
                f"{label}.reference_diameter='conductance_matched' requires "
                f"{label}.use_mean=true"
            )
        if bool(use_mean):
            raise ValueError(
                f"{label}.reference_diameter='conductance_matched' requires the "
                "final policy to build per-outlet trees (use_mean=false)"
            )
        free_diameters = sorted(
            name for name in free_param_names if name in {"lpa.diameter", "rpa.diameter"}
        )
        if free_diameters:
            raise ValueError(
                f"{label}.reference_diameter='conductance_matched' replaces the "
                f"shared tree diameter and cannot be combined with free {free_diameters}"
            )

    return {
        "use_mean": policy_use_mean,
        "diameter_scale": policy_diameter_scale,
        "diameter_std_cap": policy_std_cap,
        "reference_diameter": reference,
    }
