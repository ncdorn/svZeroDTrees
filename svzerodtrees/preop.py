import copy
import csv
from pathlib import Path
import numpy as np
import json
from svzerodtrees.utils import *
from svzerodtrees.post_processing.stree_data_processing import *
from svzerodtrees.post_processing.stree_visualization import *
from scipy.optimize import minimize
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees.adaptation import *


def optimize_outlet_bcs(input_file,
                        clinical_targets: csv,
                        log_file=None,
                        make_steady=False,
                        unsteady=False,
                        change_to_R=False,
                        show_optimization=True):
    '''

    :param clinical_targets: clinical targets input csv
    :param input_file: 0d solver json input file name string
    :param output_file: 0d solver json output file name string
    :return preop_config: 0D config with optimized BCs
    :return preop_flow: flow result with optimized preop BCs
    '''
    # get the clinical target values
    write_to_log(log_file, "Getting clinical target values...")
    bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
    cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
    q = bsa * cardiac_index * 16.667 # cardiac output in L/min. convert to cm3/s
    mpa_pressures = get_value_from_csv(clinical_targets, 'mpa pressures') # mmHg
    mpa_sys_p_target = int(mpa_pressures[0:2])
    mpa_dia_p_target = int(mpa_pressures[3:5])
    mpa_mean_p_target = int(get_value_from_csv(clinical_targets, 'mpa mean pressure'))
    target_ps = np.array([
        mpa_sys_p_target,
        mpa_dia_p_target,
        mpa_mean_p_target
    ])
    target_ps = target_ps * 1333.22 # convert to barye

    # load input json as a config dict
    with open(input_file) as ff:
        preop_config = json.load(ff)

    if make_steady:
        make_inflow_steady(preop_config)
        write_to_log(log_file, "inlet BCs converted to steady")

    if change_to_R:
        Pd = convert_RCR_to_R(preop_config)
        write_to_log(log_file, "RCR BCs converted to R, Pd = " + str(Pd))
    # get resistances from the zerod input file
    resistance = get_resistances(preop_config)
    # get the LPA and RPA branch numbers
    lpa_rpa_branch = [1, 2] # in general, this should be the branch of LPA and RPA based on the simvascular 0d algorithm

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(r,
                                     input_config=preop_config,
                                     target_ps=None,
                                     unsteady=unsteady,
                                     lpa_rpa_branch=lpa_rpa_branch
                                     ):
        # r = abs(r)
        # r = [r, r]
        write_resistances(input_config, r)
        zerod_result = run_svzerodplus(input_config)
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_pressure(zerod_result, branch=0) # get mpa pressure

        # lpa_rpa_branch = ["V" + str(idx) for idx in input_config["junctions"][0]["outlet_vessels"]]

        q_MPA = get_branch_result(zerod_result, branch=0, data_name='flow_in')
        q_RPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[0], data_name='flow_in')
        q_LPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[1], data_name='flow_in')
        if unsteady:
            pred_p = np.array([
                mpa_sys_p,
                mpa_dia_p,
                mpa_mean_p
            ])
            p_diff = np.sum(np.square(np.subtract(pred_p, target_ps)))
        else:
            p_diff = abs(target_ps[2] - mpa_mean_p) ** 2
        # SSE = np.sum(np.square(np.subtract(pred_p, target_p)))
        # MSE = np.square(np.subtract(pred_p, target_p)).mean()

        RPA_diff = abs((q_RPA[-1] - (0.52 * q_MPA[-1]))) ** 2
        
        min_obj = p_diff + RPA_diff
        if show_optimization:
            obj_fun.append(min_obj)
            plot_optimization_progress(obj_fun)
        return min_obj

    # write to log file for debugging
    write_to_log(log_file, "Optimizing preop outlet resistance...")
    # run the optimization algorithm
    result = minimize(zerod_optimization_objective,
                      resistance,
                      args=(preop_config, target_ps),
                      method="CG",
                      options={"disp": False},
                      )
    log_optimization_results(log_file, result, '0D optimization')
    # write to log file for debugging
    write_to_log(log_file, "Outlet resistances optimized! " + str(result.x))

    R_final = result.x # get the array of optimized resistances
    write_resistances(preop_config, R_final)

    preop_flow = run_svzerodplus(preop_config)

    return preop_config, preop_flow


def construct_cwss_trees(config: dict, result, log_file=None, vis_trees=False, fig_dir=None):
    '''
    construct structured trees at every outlet of the 0d model optimized against the outflow BC resistance,
    for the constant wall shear stress assumption.

    :param config: 0D solver config
    :param result: 0D solver result corresponding to config
    :param log_file: optional path to a log file
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.

    :return roots: return the root TreeVessel objects of the outlet trees

    '''
    trees = []
    q_outs = get_outlet_data(config, result, 'flow_out', steady=True)
    p_outs = get_outlet_data(config, result, 'pressure_out', steady=True)
    outlet_idx = 0
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                        outlet_tree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                            config["simulation_parameters"],
                                                                            bc_config, 
                                                                            P_outlet=[p_outs[outlet_idx]],
                                                                            Q_outlet=[q_outs[outlet_idx]])
                        R = bc_config["bc_values"]["R"]
                # write to log file for debugging
                write_to_log(log_file, "** building tree for resistance: " + str(R) + " **")
                # outlet_tree.optimize_tree_radius(R)
                x, fun, R_final = outlet_tree.optimize_tree_radius(R, log_file)
                # write to log file for debugging
                write_to_log(log_file, "     the number of vessels is " + str(outlet_tree.count_vessels()))
                vessel_config["tree"] = outlet_tree.block_dict
                trees.append(outlet_tree)
                outlet_idx += 1

    # if vis_trees:
    #     visualize_trees(config, roots, fig_dir=fig_dir, fig_name='_preop')

    return trees


def construct_pries_trees(config: dict, result, log_file=None, vis_trees=False, fig_dir=None):
    '''
    construct trees for pries and secomb adaptation and perform initial integration
    :param config: 0D solver preop config
    :param result: 0D solver result corresponding to config
    :param ps_params: Pries and Secomb parameters in the form [k_p, k_m, k_c, k_s, L (cm), S_0, tau_ref, Q_ref], default are from Pries et al. 2001
        units:
            k_p, k_m, k_c, k_s [=] dimensionless
            L [=] cm
            S_0 [=] dimensionless
            tau_ref [=] dyn/cm2
            Q_ref [=] cm3/s
    :param log_file: optional path to a log file
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.
    '''
    simparams = config["simulation_parameters"]
    # get the outlet flowrate
    q_outs = get_outlet_data(config, result, "flow_out", steady=True)
    p_outs = get_outlet_data(config, result, "pressure_out", steady=True)
    trees = []
    outlet_idx = 0 # need this when iterating through outlets 
    # get the outlet vessel
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                               simparams,
                                                                               bc_config, 
                                                                               Q_outlet=[np.mean(q_outs[outlet_idx])],
                                                                               P_outlet=[np.mean(p_outs[outlet_idx])])
                        R = bc_config["bc_values"]["R"]
                # write to log file for debugging
                write_to_log(log_file, "** building tree for resistance: " + str(R) + " **")
                # outlet_tree.optimize_tree_radius(R)
                x, fun, R_final = outlet_stree.optimize_tree_radius(R, log_file)
                print('integrating pries and secomb')
                outlet_stree.integrate_pries_secomb()
                # write to log file for debugging
                write_to_log(log_file, "     the number of vessels is " + str(outlet_stree.count_vessels()))
                vessel_config["tree"] = outlet_stree.block_dict
                trees.append(outlet_stree)
                outlet_idx += 1
    
    # integrate_pries_secomb(ps_params, trees)

    return trees

    
    def optimize_ps_params():

        pass