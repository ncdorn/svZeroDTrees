"""Microvasculature model interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = ["StructuredTree", "TreeVessel", "TreeParameters", "compliance"]

_LAZY_EXPORTS = {
    "StructuredTree": (
        "svzerodtrees.microvasculature.structured_tree.structuredtree",
        "StructuredTree",
    ),
    "TreeVessel": ("svzerodtrees.microvasculature.treevessel", "TreeVessel"),
    "TreeParameters": (
        "svzerodtrees.microvasculature.treeparams",
        "TreeParameters",
    ),
}


def __getattr__(name: str):
    if name == "compliance":
        value = import_module("svzerodtrees.microvasculature.compliance")
        globals()[name] = value
        return value

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
