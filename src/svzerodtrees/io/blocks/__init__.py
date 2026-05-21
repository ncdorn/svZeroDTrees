"""Block objects used to construct svZeroD solver payloads."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Vessel",
    "Junction",
    "SimParams",
    "BoundaryCondition",
    "CouplingBlock",
    "Chamber",
    "Valve",
]

_LAZY_EXPORTS = {
    "Vessel": ("svzerodtrees.io.blocks.vessel", "Vessel"),
    "Junction": ("svzerodtrees.io.blocks.junction", "Junction"),
    "SimParams": ("svzerodtrees.io.blocks.simulation_parameters", "SimParams"),
    "BoundaryCondition": (
        "svzerodtrees.io.blocks.boundary_condition",
        "BoundaryCondition",
    ),
    "CouplingBlock": ("svzerodtrees.io.blocks.coupling_block", "CouplingBlock"),
    "Chamber": ("svzerodtrees.io.blocks.chamber", "Chamber"),
    "Valve": ("svzerodtrees.io.blocks.valve", "Valve"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

