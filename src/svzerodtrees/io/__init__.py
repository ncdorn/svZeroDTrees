"""I/O interfaces for svzerodtrees.

Expose top-level names lazily so imports of narrower subpackages such as
``svzerodtrees.io.blocks`` do not pull in solver-backed configuration code.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["ConfigHandler", "Inflow"]

_LAZY_EXPORTS = {
    "ConfigHandler": ("svzerodtrees.io.config_handler", "ConfigHandler"),
    "Inflow": ("svzerodtrees.io.inflow_handler", "Inflow"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

