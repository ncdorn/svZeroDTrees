from __future__ import annotations

import numpy as np
import pytest

from svzerodtrees import numerics


def _available_numpy_trapezoid():
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = getattr(np, "trapz", None)
    if implementation is None:
        raise RuntimeError("the test environment has no NumPy trapezoid API")
    return implementation


def test_trapezoid_selects_new_numpy_api_when_available(monkeypatch):
    reference = _available_numpy_trapezoid()
    calls = []

    def selected(y, *, x=None, dx=1.0, axis=-1):
        calls.append((x, dx, axis))
        return reference(y, x=x, dx=dx, axis=axis)

    monkeypatch.setattr(numerics.np, "trapezoid", selected, raising=False)

    result = numerics.trapezoid([0.0, 2.0, 4.0], x=[0.0, 0.5, 1.0])

    assert result == pytest.approx(2.0)
    assert calls == [([0.0, 0.5, 1.0], 1.0, -1)]


def test_trapezoid_selects_legacy_numpy_api_when_new_api_is_unavailable(monkeypatch):
    reference = _available_numpy_trapezoid()
    calls = []

    def selected(y, *, x=None, dx=1.0, axis=-1):
        calls.append((x, dx, axis))
        return reference(y, x=x, dx=dx, axis=axis)

    monkeypatch.delattr(numerics.np, "trapezoid", raising=False)
    monkeypatch.setattr(numerics.np, "trapz", selected, raising=False)

    result = numerics.trapezoid([0.0, 2.0, 4.0], x=[0.0, 0.5, 1.0])

    assert result == pytest.approx(2.0)
    assert calls == [([0.0, 0.5, 1.0], 1.0, -1)]


@pytest.mark.parametrize("use_legacy_api", [False, True])
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_trapezoid_matches_numpy_for_multidimensional_inputs(
    axis, use_legacy_api, monkeypatch
):
    values = np.arange(24.0).reshape(2, 3, 4)
    reference = _available_numpy_trapezoid()

    if use_legacy_api:
        monkeypatch.delattr(numerics.np, "trapezoid", raising=False)
        monkeypatch.setattr(numerics.np, "trapz", reference, raising=False)
    else:
        monkeypatch.setattr(numerics.np, "trapezoid", reference, raising=False)

    if axis == -1:
        x = np.array([0.0, 0.25, 0.75, 1.5])
        expected = reference(values, x=x, axis=axis)
        actual = numerics.trapezoid(values, x=x, axis=axis)
    else:
        expected = reference(values, dx=0.5, axis=axis)
        actual = numerics.trapezoid(values, dx=0.5, axis=axis)

    np.testing.assert_allclose(actual, expected)


def test_trapezoid_fails_if_numpy_exposes_neither_api(monkeypatch):
    monkeypatch.delattr(numerics.np, "trapezoid", raising=False)
    monkeypatch.delattr(numerics.np, "trapz", raising=False)

    with pytest.raises(AttributeError, match="neither trapezoid nor trapz"):
        numerics.trapezoid([0.0, 1.0])
