import json
import sys
import os
import numpy as np
import time
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
from svzerodtrees.structuredtree import StructuredTree
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees.optimization import StentOptimization
from svzerodtrees import operation, preop, interface
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
import pickle
from deepdiff import DeepDiff
import pysvzerod


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
                        outlet_stree = StructuredTree.from_outlet_vessel(vessel_config, simparams, bc_config, Q_outlet=[np.mean(q_outs[outlet_idx])])
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
    clinical_targets = 'tests/cases/LPA_RPA_0d_steady/clinical_targets.csv'
    working_dir = 'tests/cases/LPA_RPA_0d_steady'

    config_handler, result_handler = preop.optimize_outlet_bcs(
        input_file,
        clinical_targets,
        log_file,
        show_optimization=False,
        
    )

    result_handler.to_file('tests/cases/LPA_RPA_0d_steady/result_handler.out')
    config_handler.to_file('tests/cases/LPA_RPA_0d_steady/preop_config.in')


def test_cwss_tree_construction():
    '''
    test the tree construction algorithm

    '''
    
    config_handler = ConfigHandler.from_file('tests/cases/LPA_RPA_0d_steady/preop_config.in')

    with open('tests/cases/LPA_RPA_0d_steady/result_handler.out', 'rb') as ff:
        result_handler = pickle.load(ff)
    
    log_file = 'tests/cases/LPA_RPA_0d_steady/LPA_RPA_0d_steady.log'

    write_to_log(log_file, 'testing tree construction', write=True)

    preop.construct_cwss_trees(config_handler, result_handler, log_file, d_min=0.02)

    # print the number of vessels in the tree
    print("n_vessels = " + str([tree.count_vessels() for tree in config_handler.trees]))

    R_bc = []
    for bc_config in config_handler.config["boundary_conditions"]:
        if 'RESISTANCE' in bc_config["bc_type"]:
            R_bc.append(bc_config["bc_values"]["R"])
    
    np.array(R_bc)
    R_opt = np.array([tree.root.R_eq for tree in config_handler.trees])

    SSE = sum((R_bc - R_opt) ** 2)


def test_pries_tree_construction():
    # test pries and secomb tree building
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    result_handler = ResultHandler.from_config_handler(config_handler)

    preop.construct_pries_trees(config_handler, result_handler, d_min=0.05, tol=0.1)


def test_repair_stenosis():
    '''
    test the virtual 0d stenosis repair algorithm for the proximal, extensive and custom cases
    '''
    preop_config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')
    
    with open('tests/cases/repair.json') as ff:
        repair_dict = json.load(ff)
    
    with open('tests/cases/LPA_RPA_0d_steady/result_handler.out', 'rb') as ff:
        result_handler = pickle.load(ff)

    operation.repair_stenosis(preop_config_handler, result_handler, repair_dict['custom'])


def test_no_repair():
    '''
    test the case in which no repair, and hence no adaptation, occurs
    '''

    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')
    
    with open('tests/cases/repair.json') as ff:
        repair_dict = json.load(ff)
    
    with open('tests/cases/LPA_RPA_0d_steady/result_handler.out', 'rb') as ff:
        result_handler = pickle.load(ff)

    repair_config = repair_dict['no repair']

    preop.construct_pries_trees(config_handler, result_handler, fig_dir='tests/cases/LPA_RPA_0d_steady/', d_min=0.49)


    operation.repair_stenosis_coefficient(config_handler, result_handler, repair_config)

    adapt_pries_secomb(config_handler, result_handler)

    result_handler.format_results()

    print(result_handler.clean_results)

    # assert result


def test_cwss_adaptation():
    '''
    test the constant wss tree adaptation algorithm
    '''
    
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    result_handler = ResultHandler.from_config_handler(config_handler)

    with open('tests/cases/repair.json') as ff:
        repair_dict = json.load(ff)
    
    repair_config = repair_dict['custom']

    preop.construct_cwss_trees(config_handler, result_handler, n_procs=12, d_min=0.03)

    operation.repair_stenosis(config_handler, result_handler, repair_config)

    adapt_constant_wss(config_handler, result_handler)

    result_handler.format_results()


def compare_parallel_tree_construction():
    '''
    test parallelized tree construction vs unparallelized
    '''

    config_handler_par = ConfigHandler.from_json('tests/cases/full_pa_test/preop_config.json')
    config_handler = ConfigHandler.from_json('tests/cases/full_pa_test/preop_config.json')

    result_handler_par = ResultHandler.from_config_handler(config_handler_par)
    result_handler = ResultHandler.from_config_handler(config_handler)

    # unparallelized
    unp_start_time = time.perf_counter()
    preop.construct_cwss_trees(config_handler, result_handler, d_min=0.02)
    unp_end_time = time.perf_counter()

    # parallelized
    par_start_time = time.perf_counter()
    preop.construct_cwss_trees_parallel(config_handler_par, result_handler_par, n_procs=24, d_min=0.02)
    par_end_time = time.perf_counter()


    print(f"unparallelized tree construction took {unp_end_time - unp_start_time} seconds")
    print(f"parallelized tree construction took {par_end_time - par_start_time} seconds")

    # compare the results
    # print(DeepDiff(result_handler_par.results, result_handler.results))


def test_pries_adaptation():
    '''
    test the constant wss tree adaptation algorithm
    '''
    
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    result_handler = ResultHandler.from_config_handler(config_handler)
    
    with open('tests/cases/repair.json') as ff:
        repair_dict = json.load(ff)
    
    repair_config = repair_dict['proximal']

    log_file = 'tests/cases/LPA_RPA_0d_steady/LPA_RPA_0d_steady.log'

    write_to_log(log_file, 'testing tree construction', write=True)

    preop.construct_pries_trees(config_handler, result_handler, n_procs=None, log_file=log_file, d_min=0.007)

    operation.repair_stenosis(config_handler, result_handler, repair_config, log_file)

    adapt_pries_secomb(config_handler, result_handler, log_file)

    result_handler.format_results()

    # print(result_handler.results)


def test_stent_optimization():
    '''
    test the stent diameter optimization method
    '''

    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    result_handler = ResultHandler.from_config_handler(config_handler)

    repair_config = {
        "type": "optimize stent",
        "location": "proximal",
        "value": [0.5, 0.5],
        "objective": "flow split",
    }

    os.chdir('tests/cases/LPA_RPA_0d_steady')

    stent_optimization = StentOptimization(config_handler,
                                               result_handler,
                                               repair_config,
                                               adapt='cwss',
                                               log_file=None,
                                               n_procs=12,
                                               trees_exist=False)
        
    stent_optimization.minimize_nm()

    print('stent optimization run')

    print(result_handler.flow_split)


def test_run_from_file():
    expfile = 'AS2_stent.json'
    os.chdir('tests/cases/AS2/experiments')

    interface.run_from_file(expfile, vis_trees=True)


def test_pa_optimizer():
    # test the pa optimizer pipeline

    os.chdir('tests/cases/AS2')
    input_file = 'AS2_prestent.json'
    log_file = 'AS2_test.log'
    clinical_targets = 'clinical_targets.csv'
    mesh_surfaces_path = '/home/ndorn/Documents/Stanford/PhD/Simvascular/threed_models/AS2_prestent/Meshes/1.6M_elements/mesh-surfaces'

    config_handler, result_handler, pa_config = preop.optimize_pa_bcs(
        input_file,
        mesh_surfaces_path,
        clinical_targets,
        log_file
    )

    pa_config.to_json('pa_reduced_config.json')

    with open('pa_config_result.json', 'w') as ff:
        json.dump(pa_config.simulate(), ff)

    # save the optimized pa config
    config_handler.to_json('pa_optimized_config.json')

    # save the preop_Result
    result_handler.to_file('pa_preop_result.out')


def test_simple_config():
    '''
    test the simplest config
    '''

    os.chdir('tests/cases/full_pa_test')
    input_file = 'pa_config.json'
    
    config_handler = ConfigHandler.from_json(input_file)

    result_handler = ResultHandler.from_config_handler(config_handler)

    config_handler.simulate(result_handler, 'preop')

    result_handler.results_to_dict()

    with open('pa_config_result.json', 'w') as ff:
        json.dump(result_handler.results['preop'], ff, indent=4)



if __name__ == '__main__':

    test_run_from_file()