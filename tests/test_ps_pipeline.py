import json
import sys
import os
import numpy as np
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
# print(sys.path)
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from svzerodtrees.structured_tree_simulation import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import integrate_pries_secomb
import svzerodtrees.preop as preop
import pickle


def build_tree(config, result):
    simparams = config["simulation_parameters"]
    # get the outlet flowrate
    q_outs = get_outlet_data(config, result, "flow_out", steady=True)
    outlet_trees = []
    outlet_idx = 0 # need this when iterating through outlets 
    # get the outlet vessel
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, simparams, bc_config, Q_outlet=[np.mean(q_outs[outlet_idx])])
                        R = bc_config["bc_values"]["R"]
                # outlet_stree.optimize_tree_radius(R)
                outlet_stree.build_tree()
                outlet_idx += 1 # track the outlet idx for more than one outlet
                outlet_trees.append(outlet_stree)
    
    return outlet_trees

def test_preop():
    '''
    test the preop optimization scheme
    '''
    input_file = 'tests/cases/LPA_RPA_0d_steady/LPA_RPA_0d_steady.json'
    log_file = 'tests/cases/LPA_RPA_0d_steady/LPA_RPA_0d_steady.log'
    working_dir = 'tests/cases/LPA_RPA_0d_steady'

    preop_config, preop_result = preop.optimize_outlet_bcs(
        'tests/cases/LPA_RPA_0d_steady/clinical_targets.csv',
        input_file,
        working_dir,
        log_file=log_file,
        show_optimization=False
    )

    with open('tests/cases/LPA_RPA_0d_steady/preop_result.out', 'wb') as ff:
        pickle.dump(preop_result, ff)
        


def test_tree_construction():
    '''
    test the tree construction algorithm
    '''
    
    with open('tests/cases/LPA_RPA_0d_steady/preop_config.in') as ff:
        preop_config = json.load(ff)

    with open('tests/cases/LPA_RPA_0d_steady/preop_result.out', 'rb') as ff:
        preop_result = pickle.load(ff)
    
    log_file = 'tests/cases/LPA_RPA_0d_steady/LPA_RPA_0d_steady.log'

    write_to_log(log_file, 'testing tree construction', write=True)

    roots = preop.construct_trees(preop_config, preop_result, log_file)

    R_bc = []
    for bc_config in preop_config["boundary_conditions"]:
        if 'RESISTANCE' in bc_config["bc_type"]:
            R_bc.append(bc_config["bc_values"]["R"])
    
    np.array(R_bc)
    R_opt = np.array([root.R_eq for root in roots])

    SSE = sum((R_bc - R_opt) ** 2)
    print(SSE)

    assert SSE < 0.1


def test_build_tree():
    # build tree with radius below 0.15 and observe wtf is happening
    pass

def run_from_file(input_file, output_file):
    """Run svZeroDPlus from file. 
    This is going to be a very reduced, simple model for the purposes of creating a tree with flow values
    and then testing pries and secomb

    Args:
        input_file: Input file with configuration.
        output_file: Output file with configuration.
    """
    with open(input_file) as ff:
        config = json.load(ff)
    # print(config)
    result = run_svzerodplus(config)
    # 
    #     result = dict(result = result.to_dict())
    # with open(output_file, "w") as ff:
    #     json.dump(result, ff)


    outlet_trees = build_tree(config, result)
    print(outlet_trees[0].count_vessels())

    # tree_config = outlet_trees[0].create_solver_config(1333.2)
    # with open('tests/cases/ps_tree_example_1.json', "w") as ff:
    #     json.dump(tree_config, ff)

    # tree_result = svzerodplus.simulate(tree_config)
    outlet_trees[0].create_bcs()
    # ps_params = [k_p, k_m, k_c, k_s, S_0, tau_ref, Q_ref, L]
    ps_params = [1.24, .229, 2.20, .885, .219, 9.66 * 10 ** -7, 1.9974, 5.9764 * 10 ** -4]
    print('R = ' + str(outlet_trees[0].R()))
    SSE = integrate_pries_secomb(ps_params, outlet_trees)
    print('R = ' + str(outlet_trees[0].R()))
    # result = minimize(optimize_pries_secomb,
    #                   ps_params,
    #                   args=(outlet_trees, config["simulation_parameters"], q_outs),
    #                   method='Nelder-Mead')
    # SSE = optimize_pries_secomb(ps_params, [outlet_tree], config["simulation_parameters"], [q_out])
    # print(result)
    # root.adapt_diameter()


    return result


if __name__ == '__main__':
    # input_file = 'tests/cases/ps_tree_test.json'
    output_file = 'tests/cases/ps_tree_test.out'

    input_file = 'tests/cases/ps_tree_test.json'



    # result = run_from_file(input_file, output_file)
    test_tree_construction()


