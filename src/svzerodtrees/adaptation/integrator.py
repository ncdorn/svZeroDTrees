import copy

import numpy as np
from scipy.integrate import solve_ivp

from .utils import pack_state, plot_adaptation_histories, rel_change, simulate_outlet_trees, time_to_95, unpack_state, wrap_event

"""
integrator for computing adaptation in a structured tree
"""


def _termination_reason(sol, model_instance):
    if sol.status == 1:
        return getattr(model_instance, "event_reason_label", "event_converged")
    if sol.status == 0:
        return "t_end_reached"
    return str(sol.message).strip() or "solver_failed"


def _first_event_time(sol):
    for event_group in getattr(sol, "t_events", []) or []:
        if len(event_group):
            return float(event_group[0])
    return None


def _radius_change_stats(y_initial, y_final, n_lpa):
    total_states = y_initial.size
    lpa_r_idx = np.arange(0, 2 * n_lpa, 2)
    lpa_h_idx = np.arange(1, 2 * n_lpa, 2)
    rpa_r_idx = np.arange(2 * n_lpa, total_states, 2)
    rpa_h_idx = np.arange(2 * n_lpa + 1, total_states, 2)

    def _stats(indices):
        if indices.size == 0:
            return {"mean_relative_change": 0.0, "max_abs_relative_change": 0.0}
        relative = (y_final[indices] - y_initial[indices]) / y_initial[indices]
        return {
            "mean_relative_change": float(np.mean(relative)),
            "max_abs_relative_change": float(np.max(np.abs(relative))),
        }

    return {
        "lpa_radius": _stats(lpa_r_idx),
        "lpa_thickness": _stats(lpa_h_idx),
        "rpa_radius": _stats(rpa_r_idx),
        "rpa_thickness": _stats(rpa_h_idx),
    }


def run_adaptation(
    preop_pa,
    postop_pa,
    model,
    K_arr,
    *,
    t_end=3600.0,
    rtol=1e-6,
    atol=1e-7,
    max_step=60.0,
):
    postop_pa.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    postop_pa.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)

    trees = (postop_pa.lpa_tree, postop_pa.rpa_tree)
    y0 = pack_state(*trees)
    y_initial = y0.copy()

    model_instance = model(K_arr)

    last_update_y = y0.copy()
    last_t_holder = [-float("inf")]
    flow_log = []
    solver_trace = []
    event_state = {"triggered": False, "was_positive": False}

    postop_pa.update_bcs()
    postop_pa.simulate()
    pre_adapted_split = postop_pa.rpa_split

    wrapped_event = wrap_event(
        model_instance.event,
        postop_pa,
        last_update_y,
        flow_log,
        event_state,
    )
    sol = solve_ivp(
        model_instance.compute_rhs,
        (0, float(t_end)),
        y0,
        args=(postop_pa, None, last_update_y, last_t_holder, flow_log, solver_trace),
        events=wrapped_event,
        method="BDF",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    y_final = sol.y[:, -1]
    unpack_state(y_final, *trees)
    postop_pa.update_bcs()
    postop_pa.simulate()

    final_rpa_split = float(postop_pa.rpa_split)
    stable = int(sol.status == 1)
    t95 = time_to_95(sol)
    geom_err = rel_change(y_final, y_initial)
    n_lpa = int(postop_pa.lpa_tree.store.n_nodes()) if hasattr(postop_pa.lpa_tree, "store") else 0

    hists = [plot_adaptation_histories(sol.y, n_lpa)]
    radius_stats = _radius_change_stats(y_initial, y_final, n_lpa)
    print(
        "Mean relative radius change "
        f"LPA: {radius_stats['lpa_radius']['mean_relative_change']:.3e}, "
        f"RPA: {radius_stats['rpa_radius']['mean_relative_change']:.3e}"
    )
    print(
        "Mean relative thickness change "
        f"LPA: {radius_stats['lpa_thickness']['mean_relative_change']:.3e}, "
        f"RPA: {radius_stats['rpa_thickness']['mean_relative_change']:.3e}"
    )

    rhs_l2_history = [float(entry["rhs_l2"]) for entry in solver_trace]
    solver_diagnostics = {
        "termination_reason": _termination_reason(sol, model_instance),
        "solver_message": str(sol.message).strip(),
        "event_time": _first_event_time(sol),
        "integration_time_points": [float(value) for value in sol.t.tolist()],
        "flow_split_history": [
            {"t": float(entry["t"]), "rpa_split": float(entry["rpa_split"])}
            for entry in flow_log
        ],
        "solver_trace": solver_trace,
        "rhs_l2_initial": rhs_l2_history[0] if rhs_l2_history else 0.0,
        "rhs_l2_final": rhs_l2_history[-1] if rhs_l2_history else 0.0,
        "radius_change": radius_stats,
    }

    result = dict(
        K_tau_r=K_arr[0],
        K_sig_r=K_arr[1],
        K_tau_h=K_arr[2],
        K_sig_h=K_arr[3],
        preop_rpa_split=float(preop_pa.rpa_split),
        postop_rpa_split=float(pre_adapted_split),
        final_rpa_split=final_rpa_split,
        geom_err=float(geom_err),
        t95=float(t95),
        stable=stable,
        n_rhs=sol.nfev,
        solver_diagnostics=solver_diagnostics,
    )

    return result, flow_log, sol, postop_pa, hists


def run_adaptation_outsidesim(
    preop_pa,
    postop_pa,
    model,
    K_arr,
    *,
    t_end=3600.0,
    rtol=1e-6,
    atol=1e-7,
    max_step=60.0,
):
    postop_pa.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    postop_pa.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)

    trees = (postop_pa.lpa_tree, postop_pa.rpa_tree)
    y0 = pack_state(*trees)

    model_instance = model(K_arr)

    last_update_y = y0.copy()
    last_t_holder = [-float("inf")]
    flow_log = []

    postop_pa.update_bcs()
    postop_pa.simulate()
    pre_adapted_split = postop_pa.rpa_split

    wrapped_event = wrap_event(model_instance.event_outsidesim, postop_pa, last_update_y, flow_log)

    last_rpa_split = 0.0
    relative_split_change = 1.0

    while relative_split_change - 1e-5 > 0.0:
        y0 = pack_state(*trees)

        print("\nsimulating postop_pa")
        postop_pa.update_bcs()
        postop_pa.simulate()

        print("integrating adaptation to steady state")
        sol = solve_ivp(
            model_instance.compute_rhs_nosim,
            (0, float(t_end)),
            y0,
            args=(postop_pa, None, last_update_y, last_t_holder, flow_log),
            events=wrapped_event,
            method="BDF",
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step),
        )

        geom_rel_change = (sol.y[:, -1] - last_update_y) / last_update_y
        geom_change = np.mean(np.abs(geom_rel_change))

        unpack_state(sol.y[:, -1], *trees)
        simulate_outlet_trees(postop_pa)
        postop_pa.update_bcs()

        relative_split_change = (
            abs(postop_pa.rpa_split - last_rpa_split) / last_rpa_split if last_rpa_split != 0 else 1.0
        )
        print(f"relative split change this iteration: {relative_split_change:.3e}")
        last_rpa_split = postop_pa.rpa_split

    final_rpa_split = postop_pa.rpa_split
    stable = int(sol.status == 1)
    t95 = time_to_95(sol)
    geom_err = rel_change(sol.y[:, -1], y0)

    result = dict(
        K_tau_r=K_arr[0],
        K_sig_r=K_arr[1],
        K_tau_h=K_arr[2],
        K_sig_h=K_arr[3],
        preop_rpa_split=preop_pa.rpa_split,
        postop_rpa_split=pre_adapted_split,
        final_rpa_split=final_rpa_split,
        geom_err=geom_err,
        t95=t95,
        stable=stable,
        n_rhs=sol.nfev,
    )

    return result, flow_log, sol, postop_pa
