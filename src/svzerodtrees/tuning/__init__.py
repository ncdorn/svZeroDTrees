"""Iteration tuning helpers for 3D-0D workflows."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_CONVERGENCE_TOLERANCE",
    "compute_centerline_mpa_metrics",
    "compute_flow_split_metrics",
    "evaluate_iteration_gate",
    "generate_reduced_pa_from_iteration",
    "prepare_reduced_rri_seed_from_learned",
    "run_impedance_tuning_for_iteration",
    "write_iteration_decision",
    "write_iteration_metrics",
]

_LAZY_EXPORTS = {
    "DEFAULT_CONVERGENCE_TOLERANCE": (
        "svzerodtrees.tuning.iteration",
        "DEFAULT_CONVERGENCE_TOLERANCE",
    ),
    "compute_centerline_mpa_metrics": (
        "svzerodtrees.tuning.iteration",
        "compute_centerline_mpa_metrics",
    ),
    "compute_flow_split_metrics": (
        "svzerodtrees.tuning.iteration",
        "compute_flow_split_metrics",
    ),
    "evaluate_iteration_gate": (
        "svzerodtrees.tuning.iteration",
        "evaluate_iteration_gate",
    ),
    "generate_reduced_pa_from_iteration": (
        "svzerodtrees.tuning.iteration",
        "generate_reduced_pa_from_iteration",
    ),
    "prepare_reduced_rri_seed_from_learned": (
        "svzerodtrees.tuning.learned_seed",
        "prepare_reduced_rri_seed_from_learned",
    ),
    "run_impedance_tuning_for_iteration": (
        "svzerodtrees.tuning.iteration",
        "run_impedance_tuning_for_iteration",
    ),
    "write_iteration_decision": (
        "svzerodtrees.tuning.iteration",
        "write_iteration_decision",
    ),
    "write_iteration_metrics": (
        "svzerodtrees.tuning.iteration",
        "write_iteration_metrics",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
