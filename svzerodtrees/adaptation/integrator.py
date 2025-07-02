import copy
from .utils import wrap_event

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
    wrapped_event = wrap_event(model_instance.event, postop_pa, last_update_y) # need to propagate event rules
    sol = solve_ivp(
        model_instance.compute_rhs, (0, t_end), y0,
        args=(postop_pa, all_vessels, last_update_y, last_t_holder, flow_log),
        events=wrapped_event, # events=lambda t, y, *args: model_instance.event(t, y, postop_pa, last_update_y),

        method='BDF', rtol=1e-6, atol=1e-9, max_step=60.0
    )

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

