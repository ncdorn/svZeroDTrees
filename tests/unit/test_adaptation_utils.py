from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import svzerodtrees.adaptation.microvascular_adaptor as adaptor_module

from svzerodtrees.adaptation.microvascular_adaptor import (
    _resolve_inflow_time_array,
    _resolve_target_pressure_csv,
    MicrovascularAdaptor,
)
from svzerodtrees.io.config_handler import ConfigHandler
from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.treeparams import TreeParameters
from svzerodtrees.adaptation.utils import (
    append_result_to_csv,
    estimate_steady_tree_hemodynamics,
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


class SteadyStore:
    def __init__(self):
        self.d = np.asarray([2.0, 1.0, 1.0], dtype=np.float64)
        self.left = np.asarray([1, -1, -1], dtype=np.int32)
        self.right = np.asarray([2, -1, -1], dtype=np.int32)
        self.gen = np.asarray([0, 1, 1], dtype=np.int32)
        self.ids = np.asarray([0, 1, 2], dtype=np.int32)
        self.lrr = 10.0
        self.eta = 0.04

    def n_nodes(self):
        return self.d.size


class SteadyTree:
    def __init__(self):
        self.store = SteadyStore()
        self.Pd = 5.0


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


def test_estimate_steady_tree_hemodynamics_conserves_flow_and_matches_poiseuille_pressures():
    tree = SteadyTree()
    hemo = estimate_steady_tree_hemodynamics(tree, root_flow=6.0, distal_pressure=5.0)

    expected_r = np.array([1.0, 0.5, 0.5])
    expected_R_seg = 8.0 * tree.store.eta * (tree.store.lrr * expected_r) / (np.pi * expected_r ** 4)
    expected_R_eq = np.array(
        [
            expected_R_seg[0] + 1.0 / (1.0 / expected_R_seg[1] + 1.0 / expected_R_seg[2]),
            expected_R_seg[1],
            expected_R_seg[2],
        ]
    )

    np.testing.assert_allclose(hemo.equivalent_resistance, expected_R_eq)
    np.testing.assert_allclose(hemo.flow_in, [6.0, 3.0, 3.0])

    root_pin = 5.0 + 6.0 * expected_R_eq[0]
    root_pout = root_pin - 6.0 * expected_R_seg[0]
    child_pout = root_pout - 3.0 * expected_R_seg[1]
    np.testing.assert_allclose(hemo.pressure_in, [root_pin, root_pout, root_pout])
    np.testing.assert_allclose(hemo.pressure_out, [root_pout, child_pout, child_pout])
    np.testing.assert_allclose(
        hemo.wall_shear_stress,
        4.0 * tree.store.eta * np.array([6.0, 3.0, 3.0]) / (np.pi * expected_r ** 3),
    )


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


def test_create_impedance_bcs_tracks_runtime_tree_names(monkeypatch):
    class FakeTree:
        def __init__(self, name):
            self.name = name

        def compute_olufsen_impedance(self, n_procs=1, tsteps=None):
            return np.ones(int(tsteps or 1)), [0.0, 1.0]

        def create_impedance_bc(self, name, _outlet_id, Pd=0.0):
            return SimpleNamespace(name=name, Pd=Pd)

    monkeypatch.setattr(adaptor_module, "_impedance_kernel_steps_from_config", lambda _cfg: 2)
    monkeypatch.setattr(
        adaptor_module,
        "vtp_info",
        lambda *_args, **_kwargs: {"lpa_cap.vtp": 1.0, "rpa_cap.vtp": 1.0},
    )

    adaptor = MicrovascularAdaptor.__new__(MicrovascularAdaptor)
    adaptor.lpa_tree = FakeTree("LPA")
    adaptor.rpa_tree = FakeTree("RPA")
    adaptor.clinical_targets = SimpleNamespace(wedge_p=12.0)
    adaptor.convert_to_cm = False
    adaptor.postop_simdir = SimpleNamespace(
        svzerod_3Dcoupling=SimpleNamespace(
            bcs={
                "LPA_OUT": SimpleNamespace(name="LPA_OUT"),
                "RPA_OUT": SimpleNamespace(name="RPA_OUT"),
            }
        ),
        mesh_complete=SimpleNamespace(mesh_surfaces_dir="mesh-surfaces"),
    )

    adaptor.createImpedanceBCs()

    assert adaptor._adapted_tree_outlet_mapping == {
        "LPA": ["LPA_OUT"],
        "RPA": ["RPA_OUT"],
    }


def test_construct_impedance_trees_preserves_inductance(monkeypatch):
    created = []

    class FakeStructuredTree:
        def __init__(self, name, time, simparams=None, compliance_model=None):
            self.name = name
            self.time = time
            self.simparams = simparams
            self.compliance_model = compliance_model
            self.inductance = None
            created.append(self)

        def build(self, **kwargs):
            self.build_kwargs = kwargs

    monkeypatch.setattr(adaptor_module, "StructuredTree", FakeStructuredTree)
    monkeypatch.setattr(adaptor_module, "_resolve_inflow_time_array", lambda *_args: [0.0, 1.0])

    adaptor = MicrovascularAdaptor.__new__(MicrovascularAdaptor)
    adaptor.tree_params = {
        "lpa": TreeParameters(
            name="lpa",
            lrr=2.0,
            diameter=0.5,
            d_min=0.1,
            alpha=0.9,
            beta=0.6,
            compliance_model=ConstantCompliance(1.0),
            inductance=0.125,
        ),
        "rpa": TreeParameters(
            name="rpa",
            lrr=3.0,
            diameter=0.6,
            d_min=0.2,
            alpha=0.8,
            beta=0.5,
            compliance_model=ConstantCompliance(2.0),
            inductance=0.25,
        ),
    }
    adaptor.preop_simdir = SimpleNamespace(svzerod_3Dcoupling=None, zerod_config=None)
    adaptor.simple_pa = None

    lpa_tree, rpa_tree = adaptor.construct_impedance_trees()

    assert [tree.name for tree in created] == ["LPA", "RPA"]
    assert lpa_tree.inductance == pytest.approx(0.125)
    assert rpa_tree.inductance == pytest.approx(0.25)


def test_finalize_coupling_with_adapted_trees_cleans_inflow_and_replaces_tree_metadata(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.bcs = {
                "INFLOW": SimpleNamespace(name="INFLOW"),
                "LPA_A": SimpleNamespace(name="LPA_A"),
                "RPA_A": SimpleNamespace(name="RPA_A"),
            }
            self.inflows = {"INFLOW": object()}
            self.tree_params = {
                "old_lpa": {"name": "old_lpa"},
                "old_rpa": {"name": "old_rpa"},
            }
            self.path = "original.json"
            self.written_path = None

        def to_json(self, path):
            self.written_path = path

    adaptor = MicrovascularAdaptor.__new__(MicrovascularAdaptor)
    adaptor.postop_simdir = SimpleNamespace(svzerod_3Dcoupling=DummyCoupler())
    adaptor.adapted_simdir = SimpleNamespace(path=str(tmp_path), svzerod_3Dcoupling=None)
    adaptor._adapted_tree_outlet_mapping = {
        "lpa_tree": ["LPA_A"],
        "rpa_tree": ["RPA_A"],
    }
    adaptor.lpa_tree = SimpleNamespace(
        name="lpa_tree",
        to_dict=lambda: {
            "name": "lpa_tree",
            "inductance": 1.0,
            "compliance": {
                "model": "OlufsenCompliance",
                "params": {"k1": 11.0, "k2": -22.0, "k3": 33.0},
            },
        },
    )
    adaptor.rpa_tree = SimpleNamespace(
        name="rpa_tree",
        to_dict=lambda: {
            "name": "rpa_tree",
            "inductance": 2.0,
            "compliance": {
                "model": "OlufsenCompliance",
                "params": {"k1": 44.0, "k2": -55.0, "k3": 66.0},
            },
        },
    )

    def fake_create_impedance_bcs(*, target_coupler=None):
        assert target_coupler is not None
        target_coupler.bcs["LPA_A"] = SimpleNamespace(name="LPA_A")
        target_coupler.bcs["RPA_A"] = SimpleNamespace(name="RPA_A")

    adaptor.createImpedanceBCs = fake_create_impedance_bcs

    adaptor._finalize_coupling_with_adapted_trees()

    adapted = adaptor.adapted_simdir.svzerod_3Dcoupling
    assert "INFLOW" not in adapted.bcs
    assert "INFLOW" not in adapted.inflows
    assert sorted(adapted.tree_params) == ["lpa_tree", "rpa_tree"]
    assert adapted.tree_params["lpa_tree"]["compliance"]["model"] == "OlufsenCompliance"
    assert adapted.tree_params["lpa_tree"]["compliance"]["params"] == {"k1": 11.0, "k2": -22.0, "k3": 33.0}
    assert adapted.tree_params["rpa_tree"]["compliance"]["model"] == "OlufsenCompliance"
    assert adapted.tree_params["rpa_tree"]["compliance"]["params"] == {"k1": 44.0, "k2": -55.0, "k3": 66.0}
    assert adapted.written_path == adapted.path
