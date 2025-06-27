import numpy as np
from ..microvasculature.utils import assign_flow_to_root

'''
adaptation utils
'''

# packing/unpacking helper functions
def pack_state(vessels):
    """Flatten radii and thicknesses into y0."""
    y0 = np.empty(2*len(vessels))
    for v in vessels:
        base = 2*v.idx
        if v.r < 0.0001 or v.h < 0.0001:
            raise Exception(f"vessel {v.name} has invalid radius or thickness: r={v.r}, h={v.h}")
        y0[base]   = v.r
        y0[base+1] = v.h
    return y0

def unpack_state(y, vessels):
    """Write y back into the vessel objects (fast loop, no recursion)."""
    for v in vessels:
        base = 2*v.idx
        v.r     = y[base]
        v.h  = y[base+1]

def write_resistances(config, resistances):
    '''
    write a list of resistances to the outlet bcs of a config dict

    :param config: svzerodplus config dict
    :param resistances: list of resistances, ordered by outlet in the config
    '''
    idx = 0
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':

            bc_config['bc_values']['R'] = resistances[idx]

            idx += 1


def simulate_outlet_trees(simple_pa):
    '''
    get lpa/rpa flow, simulate the micro trees and assign flow to tree vessels
    '''
    lpa_flow = np.mean(simple_pa.result[simple_pa.result.name=='branch2_seg0']['flow_out'])
    rpa_flow = np.mean(simple_pa.result[simple_pa.result.name=='branch4_seg0']['flow_out'])
    lpa_tree_result = simple_pa.lpa_tree.simulate([lpa_flow, lpa_flow])
    assign_flow_to_root(lpa_tree_result, simple_pa.lpa_tree.root)
    rpa_tree_result = simple_pa.rpa_tree.simulate([rpa_flow, rpa_flow])
    assign_flow_to_root(rpa_tree_result, simple_pa.rpa_tree.root)


def rel_change(y, y_ref):
        """largest element-wise fractional change |Δy|/|y_ref|"""
        return np.max(np.abs((y - y_ref) / y_ref))
    

def time_to_95(sol):
    """first time point where every state is within 5 % of final value"""
    y_end = sol.y[:, -1]
    for ti, yi in zip(sol.t, sol.y.T):
        if rel_change(yi, y_end) < 0.05:
            return ti
    return np.nan


def wrap_event(event_func, *extra_args):
    def wrapped_event(t, y, *args):
        return event_func(t, y, *extra_args)
    wrapped_event.terminal = getattr(event_func, "terminal", False)
    wrapped_event.direction = getattr(event_func, "direction", 0)
    return wrapped_event