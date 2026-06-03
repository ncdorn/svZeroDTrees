import copy
from types import SimpleNamespace

import numpy as np
from scipy.integrate import BDF, DOP853, RK23, RK45, Radau, solve_ivp

from .utils import pack_state, plot_adaptation_histories, rel_change, simulate_outlet_trees, time_to_95, unpack_state, wrap_event

"""
integrator for computing adaptation in a structured tree
"""


_SOLVER_CLASSES = {
    "RK23": RK23,
    "RK45": RK45,
    "DOP853": DOP853,
    "BDF": BDF,
    "Radau": Radau,
}


def _solver_class(method):
    try:
        return _SOLVER_CLASSES[str(method)]
    except KeyError as exc:
        supported = ", ".join(sorted(_SOLVER_CLASSES))
        raise ValueError(f"Unsupported adaptation solver method '{method}'. Supported methods: {supported}.") from exc


def _build_solver_result(t_history, y_history, *, status, message, nfev, event_time=None):
    t_arr = np.asarray(t_history, dtype=np.float64)
    y_arr = np.column_stack(y_history).astype(np.float64, copy=False)
    t_events = [np.asarray([float(event_time)], dtype=np.float64)] if event_time is not None else []
    return SimpleNamespace(
        t=t_arr,
        y=y_arr,
        status=int(status),
        message=str(message),
        nfev=int(nfev),
        t_events=t_events,
    )


def _accepted_step_max_step(model_instance, requested_max_step):
    effective_max_step = float(requested_max_step)
    window_duration = getattr(model_instance, "split_window_duration", None)
    if window_duration is None:
        return effective_max_step

    # Require at least five accepted samples across the convergence window.
    sampling_cap = max(float(window_duration) / 5.0, 1e-6)
    return min(effective_max_step, sampling_cap)


def _accepted_step_convergence_diagnostics(model_instance, t, flow_log):
    if hasattr(model_instance, "convergence_diagnostics"):
        return model_instance.convergence_diagnostics(t, flow_log)

    margin = float(model_instance.convergence_margin(t, flow_log))
    return {
        "t": float(t),
        "window_duration": None,
        "window_coverage": None,
        "window_span": None,
        "center_deviation": None,
        "margin": margin,
        "converged": margin <= 0.0,
    }


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


def _run_with_accepted_step_convergence(
    postop_pa,
    trees,
    y0,
    model_instance,
    K_arr,
    *,
    t_end,
    rtol,
    atol,
    max_step,
    method,
):
    rhs_flow_log = []
    accepted_flow_log = [{"t": 0.0, "rpa_split": float(postop_pa.rpa_split)}]
    accepted_convergence_history = []
    solver_trace = []
    last_update_y = y0.copy()
    last_t_holder = [-float("inf")]
    effective_max_step = _accepted_step_max_step(model_instance, max_step)

    def rhs(t, y):
        return model_instance.compute_rhs(
            t,
            y,
            postop_pa,
            None,
            last_update_y,
            last_t_holder,
            rhs_flow_log,
            solver_trace,
        )

    solver_cls = _solver_class(method)
    solver = solver_cls(
        rhs,
        0.0,
        y0,
        float(t_end),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(effective_max_step),
    )

    t_history = [0.0]
    y_history = [y0.copy()]
    event_time = None
    termination_reason = "t_end_reached"
    message = "The solver successfully reached the end of the integration interval."

    while solver.status == "running":
        step_message = solver.step()
        if solver.status == "failed":
            termination_reason = "solver_failed"
            message = str(step_message).strip() or "solver failed"
            break

        current_t = float(solver.t)
        current_y = np.asarray(solver.y, dtype=np.float64).copy()
        t_history.append(current_t)
        y_history.append(current_y)

        unpack_state(current_y, *trees)
        postop_pa.update_bcs()
        postop_pa.simulate()
        accepted_flow_log.append({"t": current_t, "rpa_split": float(postop_pa.rpa_split)})
        convergence_diag = _accepted_step_convergence_diagnostics(model_instance, current_t, accepted_flow_log)
        accepted_convergence_history.append(convergence_diag)

        if bool(convergence_diag["converged"]):
            event_time = current_t
            termination_reason = getattr(model_instance, "event_reason_label", "event_converged")
            message = f"Accepted-step convergence reached at t={current_t:.6f}"
            break

    status = 1 if event_time is not None else (0 if solver.status == "finished" else -1)
    sol = _build_solver_result(
        t_history,
        y_history,
        status=status,
        message=message,
        nfev=solver.nfev,
        event_time=event_time,
    )
    return (
        sol,
        accepted_flow_log,
        accepted_convergence_history,
        rhs_flow_log,
        solver_trace,
        termination_reason,
        effective_max_step,
    )


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
    method="RK23",
):
    postop_pa.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    postop_pa.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)
    # Force the first RHS evaluation to rebuild outlet-tree hemodynamics from the
    # postop reduced-order state rather than reusing copied preop results.
    postop_pa._tree_results_time = None
    if hasattr(postop_pa.lpa_tree, "results"):
        postop_pa.lpa_tree.results = None
    if hasattr(postop_pa.rpa_tree, "results"):
        postop_pa.rpa_tree.results = None

    trees = (postop_pa.lpa_tree, postop_pa.rpa_tree)
    y0 = pack_state(*trees)
    y_initial = y0.copy()

    model_instance = model(K_arr)

    postop_pa.update_bcs()
    postop_pa.simulate()
    pre_adapted_split = postop_pa.rpa_split

    uses_accepted_step_convergence = hasattr(model_instance, "convergence_margin")

    if uses_accepted_step_convergence:
        (
            sol,
            flow_log,
            accepted_convergence_history,
            rhs_flow_log,
            solver_trace,
            termination_reason,
            effective_max_step,
        ) = _run_with_accepted_step_convergence(
            postop_pa,
            trees,
            y0,
            model_instance,
            K_arr,
            t_end=t_end,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            method=method,
        )
    else:
        last_update_y = y0.copy()
        event_reference_y = y0.copy()
        last_t_holder = [-float("inf")]
        flow_log = []
        accepted_convergence_history = []
        rhs_flow_log = flow_log
        solver_trace = []
        event_state = {"triggered": False, "was_positive": False}
        effective_max_step = float(max_step)

        wrapped_event = wrap_event(
            model_instance.event,
            postop_pa,
            event_reference_y,
            flow_log,
            event_state,
        )
        sol = solve_ivp(
            model_instance.compute_rhs,
            (0, float(t_end)),
            y0,
            args=(postop_pa, None, last_update_y, last_t_holder, flow_log, solver_trace),
            events=wrapped_event,
            method=str(method),
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step),
        )
        termination_reason = _termination_reason(sol, model_instance)

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
        "termination_reason": termination_reason,
        "solver_message": str(sol.message).strip(),
        "event_time": _first_event_time(sol),
        "integration_time_points": [float(value) for value in sol.t.tolist()],
        "flow_split_history": [
            {"t": float(entry["t"]), "rpa_split": float(entry["rpa_split"])}
            for entry in flow_log
        ],
        "rhs_flow_split_history": [
            {"t": float(entry["t"]), "rpa_split": float(entry["rpa_split"])}
            for entry in rhs_flow_log
        ],
        "solver_trace": solver_trace,
        "rhs_l2_initial": rhs_l2_history[0] if rhs_l2_history else 0.0,
        "rhs_l2_final": rhs_l2_history[-1] if rhs_l2_history else 0.0,
        "radius_change": radius_stats,
        "integration_method": str(method),
        "requested_max_step": float(max_step),
        "effective_max_step": float(effective_max_step),
        "convergence_check_mode": ("accepted_step_window" if uses_accepted_step_convergence else "scipy_event"),
        "accepted_step_flow_split_history": [
            {"t": float(entry["t"]), "rpa_split": float(entry["rpa_split"])}
            for entry in flow_log
        ],
        "accepted_step_convergence_history": accepted_convergence_history,
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
