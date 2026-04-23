from __future__ import annotations

import importlib
from functools import lru_cache

_INSTALL_HINT = (
    "pysvzerod is required for solver-backed svZeroDTrees workflows. "
    "Install the sibling svZeroDSolver checkout first with "
    "`python -m pip install -e ../svZeroDSolver` "
    "(or `python -m pip install -e /home/users/ndorn/svZeroDSolver` on Sherlock)."
)


@lru_cache(maxsize=1)
def require_pysvzerod():
    try:
        return importlib.import_module("pysvzerod")
    except ModuleNotFoundError as exc:
        if exc.name != "pysvzerod":
            raise
        raise ModuleNotFoundError(_INSTALL_HINT) from exc


def simulate_pysvzerod(config):
    return require_pysvzerod().simulate(config)
