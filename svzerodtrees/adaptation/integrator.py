import copy

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

    t_end = 3600
    sol = solve_ivp(
        model_instance.compute_rhs, (0, t_end), y0,
        args=(postop_pa, all_vessels, last_update_y, last_t_holder, flow_log),
        events=lambda t, y, *args: model_instance.event(t, y, postop_pa, last_update_y),
        method='BDF', rtol=1e-6, atol=1e-9, max_step=60.0
    )

    return sol, flow_log, postop_pa