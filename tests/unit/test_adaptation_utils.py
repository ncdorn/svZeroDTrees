from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from svzerodtrees.adaptation.microvascular_adaptor import (
    _resolve_inflow_time_array,
    _resolve_target_pressure_csv,
    MicrovascularAdaptor,
)
from svzerodtrees.io.config_handler import ConfigHandler
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


def test_resolve_target_pressure_csv_finds_postprocess_output(tmp_path):
    simdir = tmp_path / "preop"
    simdir.mkdir()
    target = tmp_path / "results" / "postprocess" / "mpa_pressure_vs_time.csv"
    target.parent.mkdir(parents=True)
    target.write_text("time_s,mpa_pressure_mmhg\n0.0,10.0\n", encoding="utf-8")

    resolved = _resolve_target_pressure_csv(SimpleNamespace(path=str(simdir)))

    assert resolved == str(target)


def test_resolve_inflow_time_array_uses_rebuilt_primary_flow_boundary_condition():
    handler = ConfigHandler(
        {
            "boundary_conditions": [
                {
                    "bc_name": "QIN",
                    "bc_type": "FLOW",
                    "bc_values": {"Q": [1.0, 1.0, 1.0], "t": [0.0, 0.5, 1.0]},
                }
            ],
            "simulation_parameters": {
                "number_of_time_pts_per_cardiac_cycle": 3,
                "number_of_cardiac_cycles": 1,
            },
            "vessels": [],
            "junctions": [],
        }
    )
    handler.inflows.clear()

    assert _resolve_inflow_time_array(handler) == [0.0, 0.5, 1.0]


def test_resolve_inflow_time_array_falls_back_when_primary_config_has_no_flow_bc():
    primary = ConfigHandler(
        {
            "boundary_conditions": [],
            "simulation_parameters": {
                "coupled_simulation": True,
                "number_of_time_pts": 2,
                "output_all_cycles": True,
                "steady_initial": False,
                "density": 1.06,
                "viscosity": 0.04,
            },
            "external_solver_coupling_blocks": [],
            "vessels": [],
            "junctions": [],
        }
    )
    fallback = ConfigHandler(
        {
            "boundary_conditions": [
                {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {"Q": [1.0, 2.0, 1.0], "t": [0.0, 0.5, 1.0]},
                }
            ],
            "simulation_parameters": {
                "number_of_time_pts_per_cardiac_cycle": 3,
                "number_of_cardiac_cycles": 1,
            },
            "vessels": [],
            "junctions": [],
        }
    )

    assert _resolve_inflow_time_array(primary, fallback) == [0.0, 0.5, 1.0]


def test_tree_metadata_with_outlet_mapping_attaches_bc_names():
    adaptor = MicrovascularAdaptor.__new__(MicrovascularAdaptor)
    adaptor._adapted_tree_outlet_mapping = {"lpa_tree": ["LPA_A", "LPA_B"], "rpa_tree": []}

    tree = SimpleNamespace(name="lpa_tree", to_dict=lambda: {"name": "lpa_tree", "inductance": 0.0})

    metadata = adaptor._tree_metadata_with_outlet_mapping(tree)

    assert metadata["outlet_mapping"] == {"bc_names": ["LPA_A", "LPA_B"]}
