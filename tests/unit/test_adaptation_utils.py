from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from svzerodtrees.adaptation.utils import (
    append_result_to_csv,
    pack_state,
    rel_change,
    time_to_95,
    unpack_state,
    wrap_event,
)


class FakeStore:
    def __init__(self, diameters):
        self.d = np.asarray(diameters, dtype=np.float64)

    def n_nodes(self):
        return self.d.size


class FakeTree:
    def __init__(self, diameters):
        self.store = FakeStore(diameters)


def test_pack_state_handles_empty_single_multiple_and_stale_thickness():
    assert pack_state().size == 0
    assert pack_state(None).size == 0

    tree = FakeTree([2.0, 4.0])
    packed = pack_state(tree)
    np.testing.assert_allclose(packed, [1.0, 0.1, 2.0, 0.2])
    np.testing.assert_allclose(tree._thickness_state, [0.1, 0.2])

    tree._thickness_state = np.array([9.0])
    repacked = pack_state(tree)
    np.testing.assert_allclose(repacked, [1.0, 0.1, 2.0, 0.2])

    second = FakeTree([6.0])
    np.testing.assert_allclose(pack_state([tree, second]), [1.0, 0.1, 2.0, 0.2, 3.0, 0.3])


def test_pack_state_requires_built_storage():
    with pytest.raises(AttributeError, match="missing built storage"):
        pack_state(SimpleNamespace(store=None))


def test_unpack_state_updates_diameters_and_thickness_for_multiple_trees():
    first = FakeTree([2.0, 4.0])
    second = FakeTree([6.0])

    unpack_state(np.array([1.5, 0.2, 2.5, -1.0, 3.5, 0.4]), first, second)

    np.testing.assert_allclose(first.store.d, [3.0, 5.0])
    np.testing.assert_allclose(second.store.d, [7.0])
    np.testing.assert_allclose(first._thickness_state, [0.2, 1e-9])
    np.testing.assert_allclose(second._thickness_state, [0.4])


def test_unpack_state_rejects_bad_length_and_missing_storage():
    tree = FakeTree([2.0])
    with pytest.raises(ValueError, match="State vector length"):
        unpack_state([1.0], tree)

    with pytest.raises(AttributeError, match="missing built storage"):
        unpack_state([1.0, 0.1], SimpleNamespace(store=None))


def test_rel_change_and_time_to_95():
    assert rel_change(np.array([2.0, 3.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)

    sol = SimpleNamespace(
        t=np.array([0.0, 1.0, 2.0]),
        y=np.array(
            [
                [1.0, 9.6, 10.0],
                [1.0, 19.5, 20.0],
            ]
        ),
    )
    assert time_to_95(sol) == pytest.approx(1.0)

    never = SimpleNamespace(t=np.array([0.0]), y=np.array([[1.0], [2.0]]))
    assert time_to_95(never) == pytest.approx(0.0)


def test_wrap_event_preserves_terminal_direction_and_binds_args():
    calls = []

    def event_func(t, y, scale):
        calls.append((t, tuple(y), scale))
        return scale * t

    event_func.terminal = True
    event_func.direction = -1

    wrapped = wrap_event(event_func, 3.0)

    assert wrapped(2.0, np.array([1.0])) == pytest.approx(6.0)
    assert wrapped.terminal is True
    assert wrapped.direction == -1
    assert calls == [(2.0, (1.0,), 3.0)]


def test_append_result_to_csv_appends_header_only_once(tmp_path):
    output_path = tmp_path / "results.csv"

    append_result_to_csv(pd.DataFrame([{"a": 1, "b": 2}]), str(output_path))
    append_result_to_csv(pd.DataFrame([{"a": 3, "b": 4}]), str(output_path))

    assert output_path.read_text() == "a,b\n1,2\n3,4\n"
