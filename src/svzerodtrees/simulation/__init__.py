"""Simulation interfaces for svzerodtrees."""

from __future__ import annotations

from importlib import import_module

__all__ = ["SimulationDirectory", "Simulation"]

_LAZY_EXPORTS = {
    "SimulationDirectory": (
        "svzerodtrees.simulation.simulation_directory",
        "SimulationDirectory",
    ),
    "Simulation": ("svzerodtrees.simulation.simulation", "Simulation"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
