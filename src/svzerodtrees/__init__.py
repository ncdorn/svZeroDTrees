"""Top-level package for svzerodtrees.

Keep top-level imports lazy so package discovery and test collection do not
eagerly import the full scientific stack.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "StructuredTree",
    "ConfigHandler",
    "Inflow",
    "Simulation",
    "SimulationDirectory",
    "PipelineWorkflow",
    "TuneBCsWorkflow",
    "ConstructTreesWorkflow",
    "AdaptationWorkflow",
    "PostprocessWorkflow",
    "run_structured_tree_adaptation",
    "load_config",
    "compute_pulmonary_resistance_map",
    "DEFAULT_CONVERGENCE_TOLERANCE",
    "compute_centerline_mpa_metrics",
    "compute_flow_split_metrics",
    "evaluate_iteration_gate",
    "generate_reduced_pa_from_iteration",
    "run_impedance_tuning_for_iteration",
    "write_iteration_decision",
    "write_iteration_metrics",
]

_LAZY_EXPORTS = {
    "StructuredTree": ("svzerodtrees.microvasculature", "StructuredTree"),
    "ConfigHandler": ("svzerodtrees.io", "ConfigHandler"),
    "Inflow": ("svzerodtrees.io", "Inflow"),
    "Simulation": ("svzerodtrees.simulation", "Simulation"),
    "SimulationDirectory": ("svzerodtrees.simulation", "SimulationDirectory"),
    "PipelineWorkflow": ("svzerodtrees.api", "PipelineWorkflow"),
    "TuneBCsWorkflow": ("svzerodtrees.api", "TuneBCsWorkflow"),
    "ConstructTreesWorkflow": ("svzerodtrees.api", "ConstructTreesWorkflow"),
    "AdaptationWorkflow": ("svzerodtrees.api", "AdaptationWorkflow"),
    "PostprocessWorkflow": ("svzerodtrees.api", "PostprocessWorkflow"),
    "run_structured_tree_adaptation": ("svzerodtrees.api", "run_structured_tree_adaptation"),
    "load_config": ("svzerodtrees.config", "load_config"),
    "compute_pulmonary_resistance_map": (
        "svzerodtrees.post_processing",
        "compute_pulmonary_resistance_map",
    ),
    "DEFAULT_CONVERGENCE_TOLERANCE": (
        "svzerodtrees.tuning",
        "DEFAULT_CONVERGENCE_TOLERANCE",
    ),
    "compute_centerline_mpa_metrics": (
        "svzerodtrees.tuning",
        "compute_centerline_mpa_metrics",
    ),
    "compute_flow_split_metrics": (
        "svzerodtrees.tuning",
        "compute_flow_split_metrics",
    ),
    "evaluate_iteration_gate": ("svzerodtrees.tuning", "evaluate_iteration_gate"),
    "generate_reduced_pa_from_iteration": (
        "svzerodtrees.tuning",
        "generate_reduced_pa_from_iteration",
    ),
    "run_impedance_tuning_for_iteration": (
        "svzerodtrees.tuning",
        "run_impedance_tuning_for_iteration",
    ),
    "write_iteration_decision": ("svzerodtrees.tuning", "write_iteration_decision"),
    "write_iteration_metrics": ("svzerodtrees.tuning", "write_iteration_metrics"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

