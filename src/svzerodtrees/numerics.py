"""Numerical helpers shared by version-sensitive scientific code."""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(
    y: Any,
    x: Any = None,
    dx: float = 1.0,
    axis: int = -1,
) -> Any:
    """Integrate *y* with the composite trapezoidal rule.

    NumPy renamed ``trapz`` to ``trapezoid`` in its 2.x API.  Resolve the
    implementation at call time so importing this module remains safe on
    both the oldest supported NumPy and current releases.
    """

    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = getattr(np, "trapz", None)
    if implementation is None:
        raise AttributeError(
            "the installed NumPy release provides neither trapezoid nor trapz"
        )
    return implementation(y, x=x, dx=dx, axis=axis)
