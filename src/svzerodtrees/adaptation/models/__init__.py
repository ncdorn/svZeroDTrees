"""Structured-tree adaptation models."""

from __future__ import annotations

from importlib import import_module

__all__ = ["AdaptationModel", "CWSSAdaptation", "CWSSIMSAdaptation"]

_LAZY_EXPORTS = {
    "AdaptationModel": ("svzerodtrees.adaptation.models.base", "AdaptationModel"),
    "CWSSAdaptation": ("svzerodtrees.adaptation.models.cwss", "CWSSAdaptation"),
    "CWSSIMSAdaptation": (
        "svzerodtrees.adaptation.models.cwss_ims",
        "CWSSIMSAdaptation",
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
