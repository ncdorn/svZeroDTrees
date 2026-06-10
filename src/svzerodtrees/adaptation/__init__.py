"""Structured-tree adaptation interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "MicrovascularAdaptor",
    "run_adaptation",
    "CWSSAdaptation",
    "CWSSIMSAdaptation",
    "run_structured_tree_adaptation",
    "run_reduced_pa_adaptation_case",
    "run_adaptation_benchmark_study",
]

_LAZY_EXPORTS = {
    "MicrovascularAdaptor": (
        "svzerodtrees.adaptation.microvascular_adaptor",
        "MicrovascularAdaptor",
    ),
    "run_adaptation": ("svzerodtrees.adaptation.integrator", "run_adaptation"),
    "CWSSAdaptation": ("svzerodtrees.adaptation.models.cwss", "CWSSAdaptation"),
    "CWSSIMSAdaptation": (
        "svzerodtrees.adaptation.models.cwss_ims",
        "CWSSIMSAdaptation",
    ),
    "run_structured_tree_adaptation": (
        "svzerodtrees.adaptation.workflow",
        "run_structured_tree_adaptation",
    ),
    "run_reduced_pa_adaptation_case": (
        "svzerodtrees.adaptation.benchmark",
        "run_reduced_pa_adaptation_case",
    ),
    "run_adaptation_benchmark_study": (
        "svzerodtrees.adaptation.benchmark",
        "run_adaptation_benchmark_study",
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
