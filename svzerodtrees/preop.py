import copy
import csv
from pathlib import Path
import numpy as np
import json
from svzerodtrees.utils import *
from svzerodtrees.post_processing.plotting import *
from svzerodtrees.post_processing.stree_visualization import *
from scipy.optimize import minimize, Bounds
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees.adaptation import *


def optimize_outlet_bcs(input_file,
                        clinical_targets: csv,
                        log_file=None,
                        make_steady=False,
                        steady=True,
                        change_to_R=False,
                        show_optimization=True):
    '''
    optimize the outlet boundary conditions of a 0D model by conjugate gradient method

    :param input_file: 0d solver json input file name string
    :param clinical_targets: clinical targets input csv
    :param log_file: str with path to log_file
    :param make_steady: if the input file has an unsteady inflow, make the inflow steady
    :param steady: False if the input file has unsteady inflow
    :param change_to_R: True if you want to change the input config from RCR to R boundary conditions
    :param show_optimization: True if you want to display a track of the optimization results

    :return preop_config: 0D config with optimized BCs
    :return preop_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values, pressures in mmHg
    q, target_ps, rpa_ps, lpa_ps, wedge_p, rpa_split = get_clinical_targets(clinical_targets, log_file) 
    print(target_ps)

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
    if steady:
        config_flow(preop_config, q)

    # get resistances from the zerod input file
    if steady:
        resistance = get_resistances(preop_config)
    else:
        rcr = get_rcrs(preop_config)
        
    # get the LPA and RPA branch numbers

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(resistances,
                                     input_config=preop_config,
                                     target_ps=None,
                                     steady=steady,
                                     lpa_rpa_branch= [1, 2], # in general, this should be [1, 2]
                                     rpa_split=rpa_split
                                     ):
        '''
        objective function for 0D boundary condition optimization

        :param resistances: list of resistances or RCR values, given by the optimizer
        :param input_config: config of the simulation to be optimized
        :param target_ps: target pressures to optimize against
        :param unsteady: True if the model to be optimized has an unsteady inflow condition
        :param lpa_rpa_branch: lpa and rpa branch ids (should always be [1, 2]) 

        :return: sum of SSE of pressure targets and flow split targets
        '''
        # print("resistances: ", resistances)
        # write the optimization iteration resistances to the config
        if steady:
            write_resistances(input_config, resistances)
        else:
            write_rcrs(input_config, resistances)
            
        zerod_result = run_svzerodplus(input_config)

        # get mean, systolic and diastolic pressures
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_pressure(zerod_result, branch=0, convert_to_mmHg=True)

        # get the MPA, RPA, LPA flow rates
        q_MPA = get_branch_result(zerod_result, branch=0, data_name='flow_in', steady=steady)
        q_RPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[0], data_name='flow_in', steady=steady)
        q_LPA = get_branch_result(zerod_result, branch=lpa_rpa_branch[1], data_name='flow_in', steady=steady)

        if steady: # take the mean pressure only
            p_diff = abs(target_ps[2] - mpa_mean_p) ** 2
        else: # if unsteady, take sum of squares of mean, sys, dia pressure
            pred_p = np.array([
                -mpa_sys_p,
                -mpa_dia_p,
                -mpa_mean_p
            ])
            print("pred_p: ", pred_p)
            # p_diff = np.sum(np.square(np.subtract(pred_p, target_ps)))
            p_diff = (pred_p[0] - target_ps[0]) ** 2

        # penalty = 0
        # for i, p in enumerate(pred_p):
        #     penalty += loss_function_bound_penalty(p, target_ps[i])
            

        # add flow split to optimization by checking RPA flow against flow split
        RPA_diff = abs((np.mean(q_RPA) - (rpa_split * np.mean(q_MPA)))) ** 2
        
        # minimize sum of pressure SSE and flow split SSE
        min_obj = p_diff + RPA_diff # + penalty
        if show_optimization:
            obj_fun.append(min_obj)
            plot_optimization_progress(obj_fun)

        return min_obj

    # write to log file for debugging
    write_to_log(log_file, "Optimizing preop outlet resistance...")
    # run the optimization algorithm
    if steady:
        result = minimize(zerod_optimization_objective,
                        resistance,
                        args=(preop_config, target_ps),
                        method="CG",
                        options={"disp": False},
                        )
    else:
        bounds = Bounds(lb=0)
        result = minimize(zerod_optimization_objective,
                          rcr,
                          args=(preop_config, target_ps, steady),
                          method="CG",
                          options={"disp": False},
                          bounds=bounds
                          )
        
    log_optimization_results(log_file, result, '0D optimization')
    # write to log file for debugging
    write_to_log(log_file, "Outlet resistances optimized! " + str(result.x))

    R_final = result.x # get the array of optimized resistances
    write_resistances(preop_config, R_final)

    preop_flow = run_svzerodplus(preop_config)
    print(R_final)
    plot_pressure(preop_flow,branch=0)

    return preop_config, preop_flow


def optimize_pa_bcs(input_file,
                    clinical_targets: csv,
                    log_file=None,
                    make_steady=False,
                    steady=False,
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

    :return optimized_pa_config: 0D pa config with optimized BCs
    :return pa_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values
    q, mpa_ps, rpa_ps, lpa_ps, wedge_p, rpa_split = get_clinical_targets(clinical_targets, log_file)

    # collect the optimization targets [Q_RPA, P_MPA, P_RPA, P_LPA]
    optimization_targets = np.array([q * rpa_split, mpa_ps[2], rpa_ps[2], lpa_ps[2]])

    write_to_log(log_file, "*** clinical targets ****")
    write_to_log(log_file, "Q: " + str(q))
    write_to_log(log_file, "MPA pressure: " + str(mpa_ps[2]))
    write_to_log(log_file, "RPA pressure: " + str(rpa_ps[2]))
    write_to_log(log_file, "LPA pressure: " + str(lpa_ps[2]))
    write_to_log(log_file, "wedge pressure: " + str(wedge_p))
    write_to_log(log_file, "RPA flow split: " + str(rpa_split))

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

    # create the PA optimizer config
    pa_config = create_pa_optimizer_config(preop_config, q, wedge_p)

    # resitance initial guess
    R_0 = get_pa_config_resistances(pa_config)

    # define optimization bounds [0, inf)
    bounds = Bounds(lb=0)

    result = minimize(pa_opt_loss_fcn, R_0, args=(pa_config, optimization_targets), method="Nelder-Mead", bounds=bounds)

    # get optimized resistances [R_RPA_proximal, R_LPA_proximal, R_RPA_distal, R_LPA_distal, R_RPA_BC, R_LPA_BC]
    # attempt 1: [263.27670986 542.12291993  50.79584318  48.28270255 265.38882281 37.34419714]
    # attempt 2: 
    R = result.x
    write_to_log(log_file, "optimized resistances: " + str(R))
    print('RPA, LPA distal resistance: ' + str(R[2:]))

    # write optimized resistances to config
    write_pa_config_resistances(pa_config, R)

    # get outlet areas

    # calculate proportional outlet resistances

    



    return pa_config





def pa_opt_loss_fcn(R, pa_config, targets):
    '''
    loss function for the PA optimization
    '''

    # write the optimization iteration resistances to the config
    write_pa_config_resistances(pa_config, R)

    # run the 0D solver
    pa_result = run_svzerodplus(pa_config)

    # get the result values to compare against targets [Q_rpa, P_mpa, P_rpa, P_lpa]
    pa_eval = get_pa_optimization_values(pa_result)

    loss = np.sum((pa_eval - targets) ** 2)

    # print for debugging
    # print(loss, pa_eval, targets)
    return loss




    







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