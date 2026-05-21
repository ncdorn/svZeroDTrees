"""Boundary-condition tuning interfaces.

Keep package exports lazy so narrow imports such as ``svzerodtrees.config`` do
not eagerly load simulation modules and trigger circular imports.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PAConfig",
    "ClinicalTargets",
    "FreeParam",
    "FixedParam",
    "TiedParam",
    "TuneSpace",
    "identity",
    "positive",
    "unit_interval",
    "construct_impedance_trees",
    "assign_rcr_bcs",
    "validate_cap_to_bc_mapping",
    "ImpedanceTuner",
    "RCRTuner",
]

_LAZY_EXPORTS = {
    "PAConfig": ("svzerodtrees.tune_bcs.pa_config", "PAConfig"),
    "ClinicalTargets": (
        "svzerodtrees.tune_bcs.clinical_targets",
        "ClinicalTargets",
    ),
    "FreeParam": ("svzerodtrees.tune_bcs.tune_space", "FreeParam"),
    "FixedParam": ("svzerodtrees.tune_bcs.tune_space", "FixedParam"),
    "TiedParam": ("svzerodtrees.tune_bcs.tune_space", "TiedParam"),
    "TuneSpace": ("svzerodtrees.tune_bcs.tune_space", "TuneSpace"),
    "identity": ("svzerodtrees.tune_bcs.tune_space", "identity"),
    "positive": ("svzerodtrees.tune_bcs.tune_space", "positive"),
    "unit_interval": ("svzerodtrees.tune_bcs.tune_space", "unit_interval"),
    "construct_impedance_trees": (
        "svzerodtrees.tune_bcs.assign_bcs",
        "construct_impedance_trees",
    ),
    "assign_rcr_bcs": ("svzerodtrees.tune_bcs.assign_bcs", "assign_rcr_bcs"),
    "validate_cap_to_bc_mapping": (
        "svzerodtrees.tune_bcs.assign_bcs",
        "validate_cap_to_bc_mapping",
    ),
    "ImpedanceTuner": (
        "svzerodtrees.tune_bcs.impedance_tuner",
        "ImpedanceTuner",
    ),
    "RCRTuner": ("svzerodtrees.tune_bcs.rcr_tuner", "RCRTuner"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
