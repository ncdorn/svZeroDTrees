import json
import sys
import os
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
# print(sys.path)
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from svzerodtrees.structured_tree_simulation import *
from scipy.optimize import minimize
import svzerodplus

def build_tree(config):
    simparams = config["simulation_parameters"]

    outlet_trees = []
    outlet_idx = 0 # need this when iterating through outlets 
    # get the outlet vessel
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, simparams)
                for bc in config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc["bc_name"]:
                        R = bc["bc_values"]["R"]
                outlet_stree.optimize_tree_radius(R)
                outlet_idx += 1 # track the outlet idx for more than one outlet
                outlet_trees.append(outlet_stree)
    
    return outlet_trees


def run_from_file(input_file, output_file):
    """Run the svZeroDSolver from file. 
    This is going to be a very reduced, simple model for the purposes of creating a tree with flow values
    and then testing pries and secomb

    Args:
        input_file: Input file with configuration.
        output_file: Output file with configuration.
    """
    with open(input_file) as ff:
        config = json.load(ff)

    solver = svzerodplus.Solver(input_file)
    result = solver.run()
    #     result = dict(result = result.to_dict())
    # with open(output_file, "w") as ff:
    #     json.dump(result, ff)

    # get the outlet flowrate
    q_outs = get_outlet_data(config, result, 'flow_out', steady=True)

    outlet_trees = build_tree(config)

    # ps_params = [k_p, k_m, k_c, k_s, S_0, tau_ref, Q_ref, L]
    ps_params = [1.24, .229, 2.20, .885, .219, 9.66 * 10 ** -7, 1.9974, 5.9764 * 10 ** -4]

    SSE = optimize_pries_secomb(ps_params, outlet_trees, q_outs)
    # result = minimize(optimize_pries_secomb,
    #                   ps_params,
    #                   args=(outlet_trees, config["simulation_parameters"], q_outs),
    #                   method='Nelder-Mead')
    # SSE = optimize_pries_secomb(ps_params, [outlet_tree], config["simulation_parameters"], [q_out])
    # print(result)
    # root.adapt_diameter()



    return result


if __name__ == '__main__':
    input_file = 'models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.in'
    output_file = 'models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.out'


    result = run_from_file(input_file, output_file)


