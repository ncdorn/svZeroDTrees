"""Full-PA 0D impedance kernels span exactly one cardiac period.

svZeroDSolver's IMPEDANCE BC reads ``z`` as one period sampled at the solver
step (``number_of_time_pts_per_cardiac_cycle - 1`` values).  The kernel must
therefore be built on ``linspace(0, T, number_of_time_pts_per_cardiac_cycle)``
whatever the inflow sampling is.  Before the fix, construct_impedance_trees
built it on the inflow samples, so its span was ``tsteps * T / (n_inflow - 1)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from svzerodtrees.io.blocks.boundary_condition import resolve_impedance_kernel_time_grid
from svzerodtrees.io.config_handler import ConfigHandler
from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree
from svzerodtrees.microvasculature.treeparams import TreeParameters
from svzerodtrees.tune_bcs.assign_bcs import construct_impedance_trees

PERIOD = 0.8
SOLVER_POINTS = 11  # number_of_time_pts_per_cardiac_cycle
INFLOW_POINTS = 31  # deliberately not the solver resolution

LPA_CAPS = {"/mesh/lpa_1.vtp": 1.0, "/mesh/lpa_2.vtp": 3.0}
RPA_CAPS = {"/mesh/rpa_1.vtp": 2.0, "/mesh/rpa_2.vtp": 6.0}
SEED = {"rpa_2": "/mesh/rpa_2.vtp", "lpa_1": "/mesh/lpa_1.vtp", "rpa_1": "/mesh/rpa_1.vtp", "lpa_2": "/mesh/lpa_2.vtp"}
WEDGE_P_MMHG = 8.0


def _params(side: str, diameter: float) -> TreeParameters:
    return TreeParameters(
        name=side,
        lrr=10.0,
        diameter=diameter,
        d_min=0.4,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(6.67e5),
        inductance=0.0,
    )


def _inflow(n_points: int) -> dict:
    t = np.linspace(0.0, PERIOD, n_points)
    q = 60.0 + 40.0 * np.sin(2.0 * np.pi * t / PERIOD)
    return {"Q": q.tolist(), "t": t.tolist()}


def _handler(n_inflow_points: int) -> ConfigHandler:
    names = list(SEED)
    return ConfigHandler(
        {
            "boundary_conditions": [
                {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": _inflow(n_inflow_points)}
            ]
            + [{"bc_name": name, "bc_type": "RESISTANCE", "bc_values": {"R": 1000.0, "Pd": 0.0}} for name in names],
            "simulation_parameters": {
                "number_of_time_pts_per_cardiac_cycle": SOLVER_POINTS,
                "number_of_cardiac_cycles": 1,
                "density": 1.06,
                "viscosity": 0.04,
            },
            "vessels": [
                {
                    "boundary_conditions": {"inlet": "INFLOW"} if vid == 0 else {"outlet": names[vid - 1]},
                    "vessel_id": vid,
                    "vessel_length": 10.0,
                    "vessel_name": f"branch{vid}_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {"C": 0.0, "L": 0.0, "R_poiseuille": 1.0, "stenosis_coefficient": 0.0},
                }
                for vid in range(len(names) + 1)
            ],
            "junctions": [
                {
                    "junction_name": "J0",
                    "junction_type": "NORMAL_JUNCTION",
                    "inlet_vessels": [0],
                    "outlet_vessels": list(range(1, len(names) + 1)),
                    "areas": [1.0] * len(names),
                }
            ],
        }
    )


@pytest.fixture
def pulmonary_caps(monkeypatch):
    def fake_vtp_info(_mesh_path, *, convert_to_cm, pulmonary):
        assert pulmonary is True
        return dict(RPA_CAPS), dict(LPA_CAPS), {"/mesh/inflow.vtp": 1.0}

    monkeypatch.setattr("svzerodtrees.tune_bcs.assign_bcs.vtp_info", fake_vtp_info)


def _kernels(n_inflow_points: int, *, use_mean: bool) -> dict[str, np.ndarray]:
    handler = _handler(n_inflow_points)
    construct_impedance_trees(
        handler,
        "/mesh",
        WEDGE_P_MMHG,
        _params("lpa", 0.9),
        _params("rpa", 1.1),
        d_min=0.4,
        n_procs=1,
        use_mean=use_mean,
        specify_diameter=use_mean,
        diameter_scale=1.0,
        verbose=False,
        plot_stiffness=False,
    )
    return {name: np.asarray(handler.bcs[name].values["z"], dtype=float) for name in SEED}


def test_time_grid_is_one_period_at_solver_resolution():
    simparams = {"number_of_time_pts_per_cardiac_cycle": SOLVER_POINTS}
    time_grid, kernel_steps = resolve_impedance_kernel_time_grid(simparams, _inflow(INFLOW_POINTS)["t"])

    assert kernel_steps == SOLVER_POINTS - 1
    assert time_grid == pytest.approx(np.linspace(0.0, PERIOD, SOLVER_POINTS).tolist())
    # what compute_olufsen_impedance spans: tsteps * mean(diff(time)) == one period
    assert kernel_steps * float(np.mean(np.diff(time_grid))) == pytest.approx(PERIOD)


def test_time_grid_period_is_the_inflow_span_like_the_solver():
    simparams = {"number_of_time_pts_per_cardiac_cycle": 5}
    time_grid, _ = resolve_impedance_kernel_time_grid(simparams, [0.2, 0.5, 1.0])
    assert time_grid == pytest.approx(np.linspace(0.0, 0.8, 5).tolist())


@pytest.mark.parametrize("times", [[0.0], [0.5, 0.5], [0.0, float("nan")]])
def test_time_grid_rejects_inflow_without_a_period(times):
    with pytest.raises(ValueError, match="cardiac_period"):
        resolve_impedance_kernel_time_grid({"number_of_time_pts_per_cardiac_cycle": 5}, times)


def test_time_grid_falls_back_to_declared_cardiac_period_for_steady_inflow():
    simparams = {"number_of_time_pts_per_cardiac_cycle": 5, "cardiac_period": 0.6}
    time_grid, _ = resolve_impedance_kernel_time_grid(simparams, [0.0])
    assert time_grid == pytest.approx(np.linspace(0.0, 0.6, 5).tolist())


def test_time_grid_accepts_matching_declared_cardiac_period():
    simparams = {"number_of_time_pts_per_cardiac_cycle": 5, "cardiac_period": PERIOD}
    time_grid, _ = resolve_impedance_kernel_time_grid(simparams, _inflow(INFLOW_POINTS)["t"])
    assert time_grid[-1] == pytest.approx(PERIOD)


@pytest.mark.parametrize("declared", [0.7, 0.0, -1.0])
def test_time_grid_rejects_declared_period_that_disagrees_with_the_solver(declared):
    simparams = {"number_of_time_pts_per_cardiac_cycle": 5, "cardiac_period": declared}
    with pytest.raises(ValueError, match="cardiac_period"):
        resolve_impedance_kernel_time_grid(simparams, _inflow(INFLOW_POINTS)["t"])


@pytest.mark.parametrize("use_mean", [False, True], ids=["per_cap_trees", "shared_trees"])
def test_full_pa_kernels_do_not_depend_on_inflow_sampling(pulmonary_caps, use_mean):
    on_solver_grid = _kernels(SOLVER_POINTS, use_mean=use_mean)
    off_solver_grid = _kernels(INFLOW_POINTS, use_mean=use_mean)

    for name in SEED:
        assert off_solver_grid[name].size == SOLVER_POINTS - 1
        np.testing.assert_allclose(off_solver_grid[name], on_solver_grid[name], rtol=1e-12, atol=0.0)


def test_shared_tree_kernel_equals_kernel_on_linspace_period(pulmonary_caps):
    kernels = _kernels(INFLOW_POINTS, use_mean=True)

    reference = StructuredTree(
        name="LPA",
        time=np.linspace(0.0, PERIOD, SOLVER_POINTS).tolist(),
        simparams=_handler(SOLVER_POINTS).simparams,
        compliance_model=ConstantCompliance(6.67e5),
    )
    params = _params("lpa", 0.9)
    reference.build(
        initial_d=params.diameter,
        d_min=params.d_min,
        lrr=params.lrr,
        alpha=params.alpha,
        beta=params.beta,
        xi=params.xi,
        eta_sym=params.eta_sym,
    )
    z_ref, _ = reference.compute_olufsen_impedance(tsteps=SOLVER_POINTS - 1)

    for name in ("lpa_1", "lpa_2"):
        np.testing.assert_allclose(kernels[name], z_ref, rtol=1e-12, atol=0.0)
