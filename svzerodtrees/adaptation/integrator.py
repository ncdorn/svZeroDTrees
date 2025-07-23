import copy
from .utils import *

'''
integrator for computing adaptation in a structured tree
'''

from scipy.integrate import solve_ivp
from .utils import pack_state, rel_change, time_to_95

def run_adaptation(preop_pa, postop_pa, model, K_arr):

    postop_pa.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    postop_pa.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)

    all_vessels = postop_pa.lpa_tree.enumerate_vessels() + postop_pa.rpa_tree.enumerate_vessels(start_idx=postop_pa.lpa_tree.count_vessels())
    y0 = pack_state(all_vessels)

    model_instance = model(K_arr)

    last_update_y = y0.copy()
    last_t_holder = [-float("inf")]
    flow_log = []

    postop_pa.update_bcs()
    postop_pa.simulate()

    pre_adapted_split = postop_pa.rpa_split

    t_end = 3600
    wrapped_event = wrap_event(model_instance.event, postop_pa, last_update_y, flow_log) # need to propagate event rules
    # wrapped_event = model_instance.make_event(postop_pa, last_update_y, last_t_holder)
    sol = solve_ivp(
        model_instance.compute_rhs, (0, t_end), y0,
        args=(postop_pa, all_vessels, last_update_y, last_t_holder, flow_log),
        events=wrapped_event, # events=lambda t, y, *args: model_instance.event(t, y, postop_pa, last_update_y),

        method='BDF', rtol=1e-6, atol=1e-7, max_step=60.0
    )

    final_rpa_split = postop_pa.rpa_split
    stable          = int(sol.status == 1)          # 1 = event fired / converged
    t95             = time_to_95(sol)               # first time |Δy| < 5 %
    geom_err        = rel_change(sol.y[:, -1], y0)  # overall geometry change

    y_final = sol.y[:, -1]
    y0 = np.array(y0)
    lpa_idx = len(postop_pa.lpa_tree.enumerate_vessels())

    # create histograms
    y_third = sol.y[:, len(sol.y[0]) // 3]
    y_two_third = sol.y[:, len(sol.y[0])  // 3 * 2]

    # hists = [0, 0, 0, 0]
    # hists[0] = r_h_histogram(y0, lpa_idx)
    # hists[1] = r_h_histogram(y_third, lpa_idx)
    # hists[2] = r_h_histogram(y_two_third, lpa_idx)
    # hists[3] = r_h_histogram(y_final, lpa_idx)

    hists = [plot_adaptation_histories(sol.y, lpa_idx)]

    # Indices
    lpa_r_idx = np.arange(0, 2 * lpa_idx, 2)     # even indices for LPA radii
    lpa_h_idx = np.arange(1, 2 * lpa_idx, 2)     # odd indices for LPA thicknesses
    rpa_r_idx = np.arange(2 * lpa_idx, len(y0), 2)
    rpa_h_idx = np.arange(2 * lpa_idx + 1, len(y0), 2)

    # Compute mean relative changes
    mean_r_change_lpa = np.mean((y_final[lpa_r_idx] - y0[lpa_r_idx]) / y0[lpa_r_idx])
    mean_h_change_lpa = np.mean((y_final[lpa_h_idx] - y0[lpa_h_idx]) / y0[lpa_h_idx])
    mean_r_change_rpa = np.mean((y_final[rpa_r_idx] - y0[rpa_r_idx]) / y0[rpa_r_idx])
    mean_h_change_rpa = np.mean((y_final[rpa_h_idx] - y0[rpa_h_idx]) / y0[rpa_h_idx])

    print(f"Mean relative radius change LPA: {mean_r_change_lpa:.3e}, RPA: {mean_r_change_rpa:.3e}")
    print(f"Mean relative thickness change LPA: {mean_h_change_lpa:.3e}, RPA: {mean_h_change_rpa:.3e}")

    result = dict(
        K_tau_r  = K_arr[0],
        K_sig_r  = K_arr[1],
        K_tau_h  = K_arr[2],
        K_sig_h  = K_arr[3],
        preop_rpa_split  = preop_pa.rpa_split,
        postop_rpa_split = pre_adapted_split,
        final_rpa_split  = final_rpa_split,
        geom_err = geom_err,
        t95      = t95,
        stable   = stable,
        n_rhs    = sol.nfev            # number of RHS evaluations
    )

    return result, flow_log, sol, postop_pa, hists


def run_adaptation_outsidesim(preop_pa, postop_pa, model, K_arr):

    postop_pa.lpa_tree = copy.deepcopy(preop_pa.lpa_tree)
    postop_pa.rpa_tree = copy.deepcopy(preop_pa.rpa_tree)

    all_vessels = postop_pa.lpa_tree.enumerate_vessels() + postop_pa.rpa_tree.enumerate_vessels(start_idx=postop_pa.lpa_tree.count_vessels())
    y0 = pack_state(all_vessels)

    model_instance = model(K_arr)

    last_update_y = y0.copy()
    last_t_holder = [-float("inf")]
    flow_log = []

    
    postop_pa.update_bcs()
    postop_pa.simulate()
    pre_adapted_split = postop_pa.rpa_split

    t_end = 3600
    
    # wrapped_event = model_instance.make_event(postop_pa, last_update_y, last_t_holder)
    wrapped_event = wrap_event(model_instance.event_outsidesim, postop_pa, last_update_y, flow_log) # need to propagate event rules

    last_rpa_split = 0.0
    relative_split_change = 1.0  # to enter the while loop

    while relative_split_change - 1e-5 > 0.0:

        y0 = pack_state(all_vessels)
        
        print(f"\nsimulating postop_pa")
        postop_pa.update_bcs()
        postop_pa.simulate()

        # integrate adaptation to steady state
        print(f"integrating adaptation to steady state")
        sol = solve_ivp(
            model_instance.compute_rhs_nosim, (0, t_end), y0,
            args=(postop_pa, all_vessels, last_update_y, last_t_holder, flow_log),
            events=wrapped_event, # events=lambda t, y, *args: model_instance.event(t, y, postop_pa, last_update_y),

            method='BDF', rtol=1e-6, atol=1e-7, max_step=60.0
        )

        geom_rel_change = (sol.y[:, -1] - last_update_y) / last_update_y
        geom_change = np.mean(np.abs(geom_rel_change))

        # resimulate postop_pa with updated geometry
        unpack_state(sol.y[:, -1], all_vessels)
        simulate_outlet_trees(postop_pa)

        relative_split_change = abs(postop_pa.rpa_split - last_rpa_split) / last_rpa_split if last_rpa_split != 0 else 1.0
        print(f"relative split change this iteration: {relative_split_change:.3e}")
        last_rpa_split = postop_pa.rpa_split

    final_rpa_split = postop_pa.rpa_split
    stable          = int(sol.status == 1)          # 1 = event fired / converged
    t95             = time_to_95(sol)               # first time |Δy| < 5 %
    geom_err        = rel_change(sol.y[:, -1], y0)  # overall geometry change

    result = dict(
        K_tau_r  = K_arr[0],
        K_sig_r  = K_arr[1],
        K_tau_h  = K_arr[2],
        K_sig_h  = K_arr[3],
        preop_rpa_split  = preop_pa.rpa_split,
        postop_rpa_split = pre_adapted_split,
        final_rpa_split  = final_rpa_split,
        geom_err = geom_err,
        t95      = t95,
        stable   = stable,
        n_rhs    = sol.nfev            # number of RHS evaluations
    )

    return result, flow_log, sol, postop_pa
