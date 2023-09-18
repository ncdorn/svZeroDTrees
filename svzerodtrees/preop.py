import copy
import csv
from pathlib import Path
import numpy as np
import json
from svzerodtrees.utils import *
from svzerodtrees.post_processing.plotting import *
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
    optimize the outlet boundary conditions of a 0D model by conjugate gradient method

    :param input_file: 0d solver json input file name string
    :param clinical_targets: clinical targets input csv
    :param log_file: str with path to log_file
    :param make_steady: if the input file has an unsteady inflow, make the inflow steady
    :param unsteady: True if the input file has unsteady inflow
    :param change_to_R: True if you want to change the input config from RCR to R boundary conditions
    :param show_optimization: True if you want to display a track of the optimization results

    :return preop_config: 0D config with optimized BCs
    :return preop_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values
    q, cardiac_index, mpa_pressures, target_ps = get_clinical_targets(clinical_targets, log_file)

    # load input json as a config dict
    with open(input_file) as ff:
        preop_config = json.load(ff)

    # make inflow steady
    if make_steady:
        make_inflow_steady(preop_config)
        write_to_log(log_file, "inlet BCs converted to steady")

    # change boundary conditions to R
    if change_to_R:
        Pd = convert_RCR_to_R(preop_config)
        write_to_log(log_file, "RCR BCs converted to R, Pd = " + str(Pd))

    # add clinical flow values to the zerod input file
    config_flow(preop_config, q)

    # get resistances from the zerod input file
    resistance = get_resistances(preop_config)
    # get the LPA and RPA branch numbers

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(resistances,
                                     input_config=preop_config,
                                     target_ps=None,
                                     unsteady=unsteady,
                                     lpa_rpa_branch= [1, 2] # in general, this should be [1, 2]
                                     ):
        '''
        objective function for 0D boundary condition optimization

        :param R: list of resistances, given by the optimizer
        :param input_config: config of the simulation to be optimized
        :param target_ps: target pressures to optimize against
        :param unsteady: True if the model to be optimized has an unsteady inflow condition
        :param lpa_rpa_branch: lpa and rpa branch ids (should always be [1, 2]) 

        :return: sum of SSE of pressure targets and flow split targets
        '''

        # write the optimization iteration resistances to the config
        write_resistances(input_config, resistances)
        zerod_result = run_svzerodplus(input_config)

        # get mean, systolic and diastolic pressures
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_pressure(zerod_result, branch=0)

        # get the MPA, RPA, LPA flow rates
        q_MPA = get_branch_result(zerod_result, branch=0, data_name='flow_in')
        q_RPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[0], data_name='flow_in')
        q_LPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[1], data_name='flow_in')

        if unsteady: # if unsteady, take sum of squares of mean, sys, dia pressure
            pred_p = np.array([
                mpa_sys_p,
                mpa_dia_p,
                mpa_mean_p
            ])
            p_diff = np.sum(np.square(np.subtract(pred_p, target_ps)))
        else: # just the mean pressure
            p_diff = abs(target_ps[2] - mpa_mean_p) ** 2

        # add flow split to optimization by checking RPA flow against flow split
        RPA_diff = abs((q_RPA[-1] - (0.52 * q_MPA[-1]))) ** 2
        
        # minimize sum of pressure SSE and flow split SSE
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


def optimize_pa_bcs(input_file,
                    clinical_targets: csv,
                    log_file=None,
                    make_steady=False,
                    unsteady=False,
                    change_to_R=False,
                    show_optimization=True):
    '''
    optimize the outlet boundary conditions of a pulmonary arterial model by splitting the LPA and RPA
    into two RCR blocks. Using Nelder-Mead optimization method.

    :param input_file: 0d solver json input file name string
    :param clinical_targets: clinical targets input csv
    :param log_file: str with path to log_file
    :param make_steady: if the input file has an unsteady inflow, make the inflow steady
    :param unsteady: True if the input file has unsteady inflow
    :param change_to_R: True if you want to change the input config from RCR to R boundary conditions
    :param show_optimization: True if you want to display a track of the optimization results

    :return preop_config: 0D config with optimized BCs
    :return preop_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values
    q, cardiac_index, mpa_pressures, target_ps = get_clinical_targets(clinical_targets)

    # load input json as a config dict
    with open(input_file) as ff:
        preop_config = json.load(ff)

    # make inflow steady
    if make_steady:
        make_inflow_steady(preop_config)
        write_to_log(log_file, "inlet BCs converted to steady")

    # change boundary conditions to R
    if change_to_R:
        Pd = convert_RCR_to_R(preop_config)
        write_to_log(log_file, "RCR BCs converted to R, Pd = " + str(Pd))

    # add clinical flow values to the zerod input file
    config_flow(preop_config, q)

    # get resistances from the zerod input file
    resistance = get_resistances(preop_config)

    # create the PA optimizer config
    pa_optimizer_config = create_pa_optimizer_config(preop_config)

    pass







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
                outlet_tree.optimize_tree_radius(R, log_file)
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

                        write_to_log(log_file, "** building tree for resistance: " + str(R) + " **")

                        outlet_stree.optimize_tree_radius(R, log_file)

                        write_to_log(log_file, "    integrating pries and secomb...")
                        outlet_stree.integrate_pries_secomb()
                        bc_config["bc_values"]["R"] = outlet_stree.root.R_eq
                        write_to_log(log_file, "    pries and secomb integration completed, R_tree = " + str(outlet_stree.root.R_eq))

                        write_to_log(log_file, "     the number of vessels is " + str(outlet_stree.count_vessels()))
                        vessel_config["tree"] = outlet_stree.block_dict
                        trees.append(outlet_stree)
                outlet_idx += 1
                
                # the question is, do we write the adapted resistance to the config and recalculate the flow...


    return trees

    
    def optimize_ps_params():
        '''
        method to optimize the pries and secomb parameters to compare with Ingrid's. To be implemented
        '''
        pass