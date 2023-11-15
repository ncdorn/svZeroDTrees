import copy
import csv
from pathlib import Path
import numpy as np
import json
import math
from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.post_processing.plotting import *
from svzerodtrees.post_processing.stree_visualization import *
from scipy.optimize import minimize, Bounds
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees.adaptation import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler


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
        
    # initialize the data handlers
    result_handler = ResultHandler.from_config(preop_config)
    config_handler = ConfigHandler(preop_config)

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(resistances,
                                     input_config=config_handler.config,
                                     target_ps=None,
                                     steady=steady,
                                     lpa_branch= result_handler.lpa_branch, # in general, this should be [1, 2]
                                     rpa_branch = result_handler.rpa_branch,
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
        q_RPA = get_branch_result(zerod_result, branch=rpa_branch, data_name='flow_in', steady=steady)
        q_LPA = get_branch_result(zerod_result, branch=lpa_branch, data_name='flow_in', steady=steady)

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
                        args=(config_handler.config, target_ps, steady, result_handler.lpa_branch, result_handler.rpa_branch, rpa_split),
                        method="CG",
                        options={"disp": False},
                        )
    else:
        bounds = Bounds(lb=0, ub=math.inf)
        result = minimize(zerod_optimization_objective,
                          rcr,
                          args=(config_handler.config, target_ps, steady, result_handler.lpa_branch, result_handler.rpa_branch, rpa_split),
                          method="CG",
                          options={"disp": False},
                          bounds=bounds
                          )
        
    log_optimization_results(log_file, result, '0D optimization')
    # write to log file for debugging
    write_to_log(log_file, "Outlet resistances optimized! " + str(result.x))

    R_final = result.x # get the array of optimized resistances
    write_resistances(config_handler.config, R_final)

    

    return config_handler, result_handler


def optimize_pa_bcs(input_file,
                    mesh_surfaces_path,
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

    # initialize the data handlers
    result_handler = ResultHandler.from_config(preop_config)
    config_handler = ConfigHandler(preop_config)

    # create the PA optimizer config
    pa_config = create_pa_optimizer_config(config_handler.config, q, wedge_p)

    # resitance initial guess
    R_0 = get_pa_config_resistances(pa_config)

    # define optimization bounds [0, inf)
    bounds = Bounds(lb=0, ub=math.inf)

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
    rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path)
    
    # calculate proportional outlet resistances
    print('total number of outlets: ' + str(len(rpa_info) + len(lpa_info)))
    print('RPA total area: ' + str(sum(rpa_info.values())) + '\n')
    print('LPA total area: ' + str(sum(lpa_info.values())) + '\n')

    # distribute amongst all resistance conditions in the config
    assign_outlet_pa_bcs(config_handler.config, rpa_info, lpa_info, R[2], R[3], wedge_p)

    return config_handler, result_handler


def pa_opt_loss_fcn(R, pa_config, targets):
    '''
    loss function for the PA optimization

    :param R: list of resistances from optimizer
    :param pa_config: pa config dict
    :param targets: optimization targets [Q_RPA, P_MPA, P_RPA, P_LPA]

    :return loss: sum of squares of differences between targets and model evaluation
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

def assign_outlet_pa_bcs(config, rpa_info, lpa_info, R_rpa, R_lpa, wedge_p=13332.2):
    '''
    assign resistances proportional to outlet area to the RPA and LPA outlet bcs.
    this assumes that the rpa and lpa cap info has not changed info since export from simvascular.
    In the case of AS1, this is LPA outlets first, and then RPA (alphabetical). This will also convert all outlet BCs to resistance BCs.

    :param config: svzerodplus config dict
    :param rpa_info: dict with rpa outlet info from vtk
    :param lpa_info: dict with lpa outlet info from vtk
    :param R_rpa: RPA outlet resistance value
    :param R_lpa: LPA outlet resistance value
    '''

    def Ri(Ai, A, R):
        return R * (A / Ai)
    
    # get RPA and LPA total area
    a_RPA = sum(rpa_info.values())
    a_LPA = sum(lpa_info.values())

    # initialize list of resistances
    all_R = {}

    for name, val in lpa_info.items():
        all_R[name] = Ri(val, a_LPA, R_lpa)
    
    for name, val in rpa_info.items():
        all_R[name] = Ri(val, a_RPA, R_rpa)
    
    # write the resistances to the config
    bc_idx = 0

    # get all resistance values
    R_list = list(all_R.values())

    # loop through boundary conditions to assign resistance values
    for bc_config in config["boundary_conditions"]:

        # add resistance to resistance boundary conditions
        if bc_config["bc_type"] == 'RESISTANCE':
            bc_config["bc_values"] = {
                "R": R_list[bc_idx],
                "Pd": wedge_p * 1333.22 # convert to barye
            } 

            bc_idx += 1

        # change RCR boundary conditions to resistance
        if bc_config["bc_type"] == 'RCR':
            # change type to resistance
            bc_config["bc_type"] = 'RESISTANCE'
            # reset bc_values
            bc_config["bc_values"] = {
                "R": R_list[bc_idx],
                "Pd": wedge_p * 1333.22 # convert to barye
            } 

            bc_idx += 1


def construct_cwss_trees(config_handler, result_handler: ResultHandler, log_file=None, d_min=0.0049, vis_trees=False, fig_dir=None):
    '''
    construct structured trees at every outlet of the 0d model optimized against the outflow BC resistance,
    for the constant wall shear stress assumption.

    :param config: 0D solver config
    :param result: 0D solver result corresponding to config
    :param log_file: optional path to a log file
    :param d_min: minimum vessel diameter for tree optimization
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.

    :return roots: return the root TreeVessel objects of the outlet trees

    '''
    num_outlets = len(config_handler.config["boundary_conditions"]) - 2
    outlet_idx = 0

    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                print("** building tree for outlet " + str(outlet_idx) + " of " + str(num_outlets) + " **, d_min = " + str(d_min) + " **")
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        print(bc_config["bc_name"])
                        outlet_tree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                            config_handler.config["simulation_parameters"],
                                                                            bc_config)

                        R = bc_config["bc_values"]["R"]

                        # write to log file for debugging
                        write_to_log(log_file, "** building tree for resistance: " + str(R) + " **")

                        outlet_tree.optimize_tree_diameter(R, log_file, d_min=d_min)

                        # replace the bc resistance with the optimized value as it may be different than the initial value
                        bc_config["bc_values"]["R"] = outlet_tree.root.R_eq
                        print(' the equivalent resistance being added as the outlet boundary condition is ' + str(outlet_tree.root.R_eq))

                # write to log file for debugging
                write_to_log(log_file, "     the number of vessels is " + str(outlet_tree.count_vessels()))

                config_handler.trees.append(outlet_tree)
                outlet_idx += 1

    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTreeOutlet instances based on the preop result
    # config_handler.update_stree_hemodynamics(preop_result)

    result_handler.add_unformatted_result(preop_result, 'preop')


def construct_pries_trees(config_handler, result_handler, log_file=None, d_min=0.0049, tol=0.01, vis_trees=False, fig_dir=None):
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
    :param d_min: minimum vessel diameter for tree optimization
    :param tol: tolerance for the pries and secomb integration
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.
    '''
    simparams = config_handler.config["simulation_parameters"]
    num_outlets = len(config_handler.config["boundary_conditions"])

    # compute a pretree result to use to optimize the trees
    pretree_result = run_svzerodplus(config_handler.config)

    # get the outlet flowrate
    q_outs = get_outlet_data(config_handler.config, pretree_result, "flow_out", steady=True)
    p_outs = get_outlet_data(config_handler.config, pretree_result, "pressure_out", steady=True)
    outlet_idx = 0 # need this when iterating through outlets 
    # get the outlet vessel
    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                               simparams,
                                                                               bc_config, 
                                                                               Q_outlet=[np.mean(q_outs[outlet_idx])],
                                                                               P_outlet=[np.mean(p_outs[outlet_idx])])
                        R = bc_config["bc_values"]["R"]

                        write_to_log(log_file, "** building tree " + str(outlet_idx) + " for R = " + str(R) + " **")

                        outlet_stree.optimize_tree_diameter(R, log_file, d_min=d_min, pries_secomb=True)

                        bc_config["bc_values"]["R"] = outlet_stree.root.R_eq

                        write_to_log(log_file, "     the number of vessels is " + str(outlet_stree.count_vessels()))

                config_handler.trees.append(outlet_stree)
                outlet_idx += 1

    # compute the preop result
    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTreeOutlet instances based on the preop result
    config_handler.update_stree_hemodynamics(preop_result)

    # add the preop result to the result handler
    result_handler.add_unformatted_result(preop_result, 'preop')

    
def optimize_ps_params():
    '''
    method to optimize the pries and secomb parameters to compare with Ingrid's. To be implemented
    '''
    pass
