import copy
import csv
from pathlib import Path
import numpy as np
import json
import math
from scipy.integrate import trapz
from multiprocess import Pool
from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.post_processing.plotting import *
from svzerodtrees.post_processing.stree_visualization import *
from scipy.optimize import minimize, Bounds
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees.adaptation import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler, Vessel, BoundaryCondition, Junction, SimParams


def optimize_outlet_bcs(input_file,
                        clinical_targets: csv,
                        log_file=None,
                        make_steady=False,
                        steady=True,
                        change_to_R=False,
                        show_optimization=False):
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
    # get the clinical target values
    clinical_targets = ClinicalTargets.from_csv(clinical_targets, steady=steady)

    clinical_targets.log_clinical_targets(log_file)

    # initialize the data handlers
    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config(config_handler.config)

    if steady:
        for bc in config_handler.bcs.values():
            if bc.type == 'RCR':
                bc.change_to_R()

        
    # initialize the data handlers
    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config_handler(config_handler)

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
                                     rpa_split=clinical_targets.rpa_split
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
        print("resistances: ", resistances)
        # write the optimization iteration resistances to the config
        config_handler.update_bcs(resistances, rcr=not steady)
            
        zerod_result = run_svzerodplus(config_handler.config)

        # get mean, systolic and diastolic pressures
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_pressure(zerod_result, branch=0, convert_to_mmHg=True)

        # get the MPA, RPA, LPA flow rates
        q_MPA = get_branch_result(zerod_result, branch=0, data_name='flow_in', steady=steady)
        q_RPA = get_branch_result(zerod_result, branch=rpa_branch, data_name='flow_in', steady=steady)
        q_LPA = get_branch_result(zerod_result, branch=lpa_branch, data_name='flow_in', steady=steady)

        if steady: # take the mean pressure only
            if type(target_ps) == int:
                p_diff = abs(target_ps - mpa_mean_p) ** 2
            else:
                p_diff = abs(target_ps[2] - mpa_mean_p) ** 2
        else: # if unsteady, take sum of squares of mean, sys, dia pressure
            pred_p = np.array([
                -mpa_sys_p,
                -mpa_dia_p
                # -mpa_mean_p
            ])
            print("pred_p: ", pred_p)
            p_diff = np.sum(np.square(np.subtract(pred_p, target_ps[:1]))) + \
            np.sum(np.array([1 / bc.Rp for bc in list(config_handler.bcs.values())[1:]]) ** 2) + \
            np.sum(np.array([1 / bc.Rd for bc in list(config_handler.bcs.values())[1:]]) ** 2)
            # p_diff = (pred_p[0] - target_ps[0]) ** 2

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

    # get the initial resistance values
    initial_r = []
    for bc in config_handler.bcs.values():
        if bc.type == 'RESISTANCE':
            initial_r.append(bc.R)
        if bc.type == 'RCR':
            initial_r.append(bc.Rp)
            initial_r.append(bc.C)
            initial_r.append(bc.Rd)

    # run the optimization algorithm
    if steady:
        result = minimize(zerod_optimization_objective,
                        initial_r,
                        args=(config_handler.config, 
                              clinical_targets.mpa_p, 
                              steady, 
                              result_handler.lpa_branch, 
                              result_handler.rpa_branch, 
                              clinical_targets.rpa_split),
                        method="CG",
                        options={"disp": False},
                        )
    else:
        bounds = Bounds(lb=0, ub=math.inf)
        result = minimize(zerod_optimization_objective,
                          initial_r,
                          args=(config_handler.config, 
                                clinical_targets.mpa_p, 
                                steady, 
                                result_handler.lpa_branch, 
                                result_handler.rpa_branch, 
                                clinical_targets.rpa_split),
                          method="Nelder-Mead",
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
                    steady=True):
    '''
    optimize the outlet boundary conditions of a pulmonary arterial model by splitting the LPA and RPA
    into two Resistance blocks. Using Nelder-Mead optimization method.

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
    clinical_targets = ClinicalTargets.from_csv(clinical_targets, steady=steady)

    clinical_targets.log_clinical_targets(log_file)

    # initialize the data handlers
    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config(config_handler.config)

    # set the inflow in case it is wrong
    if steady:
        if np.mean(config_handler.bcs['INFLOW'].Q) != clinical_targets.q:
            config_handler.set_inflow(clinical_targets.q)

    # if steady, change boundary conditions to R
    if steady:
        for bc in config_handler.bcs.values():
            if bc.type == 'RCR':
                bc.change_to_R()

    if not steady:
        if config_handler.bcs['INFLOW'].t[-1] != clinical_targets.t:
            scale  = clinical_targets.t / config_handler.bcs['INFLOW'].t[-1]
            config_handler.bcs['INFLOW'].t = [t * scale for t in config_handler.bcs['INFLOW'].t]

    pa_config = PAConfig.from_config_handler(config_handler, clinical_targets)


    iterations = 1

    for i in range(iterations):
        print('beginning pa_config optimization iteration ' + str(i) + ' of ' + str(iterations) + '...')

        # distribute amongst all resistance conditions in the config
        pa_config.optimize(steady=steady)

        write_to_log(log_file, "*** optimized values ****")
        write_to_log(log_file, "MPA pressure: " + str(pa_config.P_mpa))
        write_to_log(log_file, "RPA pressure: " + str(pa_config.P_rpa))
        write_to_log(log_file, "LPA pressure: " + str(pa_config.P_lpa))
        write_to_log(log_file, "RPA flow split: " + str(pa_config.Q_rpa / clinical_targets.q))

        # get outlet areas
        rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path)

        assign_pa_bcs(config_handler, pa_config, rpa_info, lpa_info, steady=steady)

        # run the simulation
        result = run_svzerodplus(config_handler.config)

        # get the actual Q_lpa and Q_rpa
        Q_lpa = get_branch_result(result, 'flow_in', config_handler.lpa.branch, steady=steady)
        Q_rpa = get_branch_result(result, 'flow_in', config_handler.rpa.branch, steady=steady)

        flow_split = np.mean(Q_rpa) / (np.mean(Q_lpa) + np.mean(Q_rpa))
        print('\n actual flow split:  ' + str(flow_split))

        if abs(flow_split - clinical_targets.rpa_split) < 0.01:
            print('\n flow split within tolerance')
            break
        else:
            print('\n flow split not within tolerance, adjusting resistance values')
            # get the mean outlet pressure
            p_out_RPA = []
            p_out_LPA = []
            for vessel in config_handler.vessel_map.values():
                if vessel.bc is not None:
                    if "outlet" in vessel.bc:
                        if config_handler.lpa.branch in vessel.path:
                            p_out_LPA.append(np.mean(get_branch_result(result, 'pressure_out', vessel.branch, steady=steady)))
                        elif config_handler.rpa.branch in vessel.path:
                            p_out_RPA.append(np.mean(get_branch_result(result, 'pressure_out', vessel.branch, steady=steady)))
            
            p_mean_out_LPA = np.mean(p_out_LPA)
            p_mean_out_RPA = np.mean(p_out_RPA)
            print(d2m(p_mean_out_LPA), d2m(p_mean_out_RPA))

            R_eq_LPA_dist = (get_branch_result(result, 'pressure_out', config_handler.lpa.branch, steady=steady) - p_mean_out_LPA) / Q_lpa
            R_eq_RPA_dist = (get_branch_result(result, 'pressure_out', config_handler.rpa.branch, steady=steady) - p_mean_out_RPA) / Q_rpa

            print(R_eq_LPA_dist, R_eq_RPA_dist)

            # adjust the resistance values
            pa_config.lpa_dist.R = R_eq_LPA_dist
            pa_config.rpa_dist.R = R_eq_RPA_dist

        
        print('\n LPA Pressure Drop: ' + str(d2m(config_handler.get_branch_resistance(config_handler.lpa.branch) * Q_lpa)))
        print('RPA Pressure Drop: ' + str(d2m(config_handler.get_branch_resistance(config_handler.rpa.branch) * Q_rpa)))


    return config_handler, result_handler, pa_config


def assign_pa_bcs(config_handler, pa_config, rpa_info, lpa_info, steady=True):
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
    
    if not steady:
        def Ci(Ai, A, C):
            return C * (Ai / A)

    # get RPA and LPA total area
    a_RPA = sum(rpa_info.values())
    a_LPA = sum(lpa_info.values())

    # initialize list of resistances
    all_R = {}
    if not steady: 
        all_C = {}

    # TODO: NEED TO ADD CAPACITANCE DISTRIBUTION BELOW

    for name, val in lpa_info.items():
        all_R[name] = Ri(val, a_LPA, pa_config.bcs["LPA_BC"].R)
        if not steady:
            all_C[name] = Ci(val, a_LPA, pa_config.bcs["LPA_BC"].C)

    for name, val in rpa_info.items():
        all_R[name] = Ri(val, a_RPA, pa_config.bcs["RPA_BC"].R)
        if not steady: 
            all_C[name] = Ci(val, a_RPA, pa_config.bcs["RPA_BC"].C)

    # write the resistances to the config
    bc_idx = 0

    # get all resistance values
    R_list = list(all_R.values())
    if not steady: 
        C_list = list(all_C.values())

    # change the proximal LPA and RPA branch resistances
    config_handler.change_branch_resistance(config_handler.lpa.branch, pa_config.lpa_prox.R)
    config_handler.change_branch_resistance(config_handler.rpa.branch, pa_config.rpa_prox.R)


    print('\n LPA RESISTANCE: ' + str(config_handler.get_branch_resistance(config_handler.lpa.branch)))
    print('PREDICTED LPA PRESSURE DROP: ' + str(d2m(config_handler.get_branch_resistance(config_handler.lpa.branch) * pa_config.clinical_targets.q * .61)))
    print('RPA RESISTANCE: ' + str(config_handler.get_branch_resistance(config_handler.rpa.branch)))
    print('PREDICTED RPA PRESSURE DROP: ' + str(d2m(config_handler.get_branch_resistance(config_handler.rpa.branch) * pa_config.clinical_targets.q * .39)))

    # loop through boundary conditions to assign resistance values
    for bc in config_handler.bcs.values():
        if bc.type == 'RESISTANCE':
            bc.R = R_list[bc_idx]
            bc.values['Pd'] = pa_config.clinical_targets.wedge_p * 1333.22 # convert wedge pressure from mmHg to dyn/cm2
            bc_idx += 1
    
        elif bc.type == 'RCR':
            # split the resistance
            bc.Rp = R_list[bc_idx] * 0.1
            bc.C = C_list[bc_idx]
            bc.Rd = R_list[bc_idx] * 0.9
            bc_idx += 1


def construct_cwss_trees(config_handler, result_handler, n_procs=4, log_file=None, d_min=0.0049):
    '''
    construct cwss trees in parallel to increase computational speed
    '''

    for vessel in config_handler.vessel_map.values():
        if vessel.bc is not None:
            if "outlet" in vessel.bc:
                # get the bc object
                bc = config_handler.bcs[vessel.bc["outlet"]]
                # create outlet tree
                outlet_tree = StructuredTree.from_outlet_vessel(vessel, 
                                                                      config_handler.simparams,
                                                                      bc)
                
                config_handler.trees.append(outlet_tree)


    
    # function to run the tree diameter optimization
    def optimize_tree(tree):
        print('building ' + tree.name + ' for resistance ' + str(tree.params["bc_values"]["R"]) + '...')
        tree.optimize_tree_diameter(log_file=log_file, d_min=d_min)
        return tree

    # run the tree 
    with Pool(n_procs) as p: # THIS IS BUGGING
        config_handler.trees = p.map(optimize_tree, config_handler.trees)
    
    # update the resistance in the config according to the optimized tree resistance
    for bc, tree in zip(list(config_handler.bcs.values())[1:], config_handler.trees):
        bc.R = tree.root.R_eq


    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTree instances based on the preop result
    # config_handler.update_stree_hemodynamics(preop_result)

    result_handler.add_unformatted_result(preop_result, 'preop')


def construct_pries_trees(config_handler: ConfigHandler, result_handler,  n_procs=4, log_file=None, d_min=0.0049, tol=0.01, vis_trees=False, fig_dir=None):
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
    for vessel in config_handler.vessel_map.values():
        if vessel.bc is not None:
            if "outlet" in vessel.bc:
                # get the bc object
                bc = config_handler.bcs[vessel.bc["outlet"]]
                # create outlet tree
                outlet_tree = StructuredTree.from_outlet_vessel(vessel, 
                                                                      config_handler.simparams,
                                                                      bc,
                                                                      P_outlet=p_outs[outlet_idx],
                                                                      Q_outlet=q_outs[outlet_idx],)
                
                config_handler.trees.append(outlet_tree)
                # count up the outlets for indexing pressure and flow
                outlet_idx += 1


    if n_procs is None:
        # don't run as parallel processes
        for tree in config_handler.trees:
            print('building ' + tree.name + ' for resistance ' + str(tree.params["bc_values"]["R"]) + '...')
            tree.optimize_tree_diameter(log_file, d_min=d_min, pries_secomb=True)
    else:
        # run as a parallel process
        def build_tree(tree):
            print('building ' + tree.name + ' for resistance ' + str(tree.params["bc_values"]["R"]) + '...')
            tree.optimize_tree_diameter(log_file, d_min=d_min, pries_secomb=True)
            return tree


        with Pool(n_procs) as p:
            config_handler.trees = p.map(build_tree, config_handler.trees)
    
    # update the resistance in the config according to the optimized tree resistance
    for bc, tree in zip(list(config_handler.bcs.values())[1:], config_handler.trees):
        bc.R = tree.root.R_eq


    # compute the preop result
    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTree instances based on the preop result
    config_handler.update_stree_hemodynamics(preop_result)

    if n_procs is None:
        for tree in config_handler.trees:
            tree.pries_n_secomb.optimize_params()
    else:
        # parallel the parameter optimization for Pries and Secomb adaptation
        def optimize_params(tree):
            tree.pries_n_secomb.optimize_params()

            return tree
        
        with Pool(n_procs) as p:
            config_handler.trees = p.map(optimize_params, config_handler.trees)
    


    # add the preop result to the result handler
    result_handler.add_unformatted_result(preop_result, 'preop')


def construct_coupled_cwss_trees(config_handler, simulation_dir, n_procs=4, d_min=.0049):
    '''
    construct cwss trees for a 3d coupled BC'''

    coupled_surfs = get_coupled_surfaces(simulation_dir)

    for coupling_block in config_handler.coupling_blocks.values():
        coupling_block.surface = coupled_surfs[coupling_block.name]

    for bc in config_handler.bcs.values():
        if config_handler.coupling_blocks[bc.name].location == 'inlet':
            diameter = (find_vtp_area(config_handler.coupling_blocks[bc.name].surface) / np.pi)**(1/2)
            config_handler.trees.append(StructuredTree.from_bc_config(bc, config_handler.simparams, diameter))


    # function to run the tree diameter optimization
    def optimize_tree(tree):
        tree.optimize_tree_diameter(d_min=d_min)
        return tree


    # run the tree 
    with Pool(n_procs) as p:
        config_handler.trees = p.map(optimize_tree, config_handler.trees)
    

    # update the resistance in the config according to the optimized tree resistance
    outlet_idx = 0 # linear search, i know. its bad. will fix later
    for bc in config_handler.bcs.values():
        # we assume that an inlet location indicates taht this is an outlet bc and threfore undergoes adaptation
        if config_handler.coupling_blocks[bc.name].location == 'inlet':
            if bc.type == 'RCR':
                bc.Rp = config_handler.trees[outlet_idx].root.R_eq * 0.1
                bc.Rd = config_handler.trees[outlet_idx].root.R_eq * 0.9
            elif bc.type == 'RESISTANCE':
                bc.R = config_handler.trees[outlet_idx].root.R_eq
            outlet_idx += 1


class ClinicalTargets():
    '''
    class to handle clinical target values
    '''

    def __init__(self, mpa_p, lpa_p, rpa_p, q, rpa_split, wedge_p, t, steady):
        '''
        initialize the clinical targets object
        '''
        
        self.t = t
        self.mpa_p = mpa_p
        self.lpa_p = lpa_p
        self.rpa_p = rpa_p
        self.q = q
        self.rpa_split = rpa_split
        self.q_rpa = q * rpa_split
        self.wedge_p = wedge_p
        self.steady = steady


    @classmethod
    def from_csv(cls, clinical_targets: csv, steady=True):
        '''
        initialize from a csv file
        '''
        # get the flowrate
        bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
        cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
        q = bsa * cardiac_index * 16.667 # cardiac output in L/min. convert to cm3/s

        # get the cycle duration
        bpm = float(get_value_from_csv(clinical_targets, 'heart rate'))
        t = 60 / bpm

        # get the mpa pressures
        mpa_pressures = get_value_from_csv(clinical_targets, 'mpa pressures') # mmHg
        mpa_sys_p, mpa_dia_p = mpa_pressures.split("/")
        mpa_sys_p = int(mpa_sys_p)
        mpa_dia_p = int(mpa_dia_p)
        mpa_mean_p = int(get_value_from_csv(clinical_targets, 'mpa mean pressure'))

        # get the lpa pressures
        lpa_pressures = get_value_from_csv(clinical_targets, 'lpa pressures') # mmHg
        lpa_sys_p, lpa_dia_p = lpa_pressures.split("/")
        lpa_sys_p = int(lpa_sys_p)
        lpa_dia_p = int(lpa_dia_p)
        lpa_mean_p = int(get_value_from_csv(clinical_targets, 'lpa mean pressure'))

        # get the rpa pressures
        rpa_pressures = get_value_from_csv(clinical_targets, 'rpa pressures') # mmHg
        rpa_sys_p, rpa_dia_p = rpa_pressures.split("/")
        rpa_sys_p = int(rpa_sys_p)
        rpa_dia_p = int(rpa_dia_p)
        rpa_mean_p = int(get_value_from_csv(clinical_targets, 'rpa mean pressure'))

        # if steady, just take the mean
        if steady:
            mpa_p = mpa_mean_p
            lpa_p = lpa_mean_p
            rpa_p = rpa_mean_p
        else:
            mpa_p = [mpa_sys_p, mpa_dia_p, mpa_mean_p] # just systolic and mean
            lpa_p = [lpa_sys_p, lpa_dia_p, lpa_mean_p]
            rpa_p = [rpa_sys_p, rpa_dia_p, rpa_mean_p]

        # get wedge pressure
        wedge_p = int(get_value_from_csv(clinical_targets, 'wedge pressure'))

        # get RPA flow split
        rpa_split = float(get_value_from_csv(clinical_targets, 'pa flow split')[0:2]) / 100

        return cls(mpa_p, lpa_p, rpa_p, q, rpa_split, wedge_p, t, steady=steady)

        
    def log_clinical_targets(self, log_file):

        write_to_log(log_file, "*** clinical targets ****")
        write_to_log(log_file, "Q: " + str(self.q))
        write_to_log(log_file, "MPA pressures: " + str(self.mpa_p))
        write_to_log(log_file, "RPA pressures: " + str(self.rpa_p))
        write_to_log(log_file, "LPA pressures: " + str(self.lpa_p))
        write_to_log(log_file, "wedge pressure: " + str(self.wedge_p))
        write_to_log(log_file, "RPA flow split: " + str(self.rpa_split))


class PAConfig():
    '''
    a class to handle the reduced pa config for boundary condition optimization
    '''

    def __init__(self, 
                 simparams: SimParams, 
                 mpa: list, 
                 lpa_prox: list, 
                 rpa_prox: list, 
                 lpa_dist: Vessel, 
                 rpa_dist: Vessel, 
                 inflow: BoundaryCondition, 
                 wedge_p: float,
                 clinical_targets: ClinicalTargets,
                 steady: bool):
        '''
        initialize the PAConfig object
        
        :param mpa: dict with MPA config
        :param lpa_prox: list of Vessels with LPA proximal config
        :param rpa_prox: list of Vessels with RPA proximal config
        :param lpa_dist: dict with LPA distal config
        :param rpa_dist: dict with RPA distal config
        :param inflow: dict with inflow config
        :param wedge_p: wedge pressure'''
        self.mpa = mpa
        self.rpa_prox = rpa_prox
        # edit the parameters of the prox rpa, lpa
        self.rpa_prox.length = 10.0
        self.rpa_prox.stenosis_coefficient = 0.0
        self.lpa_prox = lpa_prox
        self.lpa_prox.length = 10.0
        self.lpa_prox.stenosis_coefficient = 0.0
        self.rpa_dist = rpa_dist
        self.lpa_dist = lpa_dist

        self.simparams = simparams

        self.clinical_targets = clinical_targets

        self.steady = steady

        self._config = {}
        self.junctions = {}
        self.vessel_map = {}
        self.bcs = {}
        self.initialize_config_maps()


        self.initialize_bcs(inflow, wedge_p)


    @classmethod
    def from_config_handler(cls, config_handler, clinical_targets: ClinicalTargets, steady: bool=True):
        '''
        initialize from a general config handler
        '''
        mpa = copy.deepcopy(config_handler.mpa)
        rpa_prox = copy.deepcopy(config_handler.rpa)
        lpa_prox = copy.deepcopy(config_handler.lpa)
        rpa_dist = Vessel.from_config({
            "boundary_conditions":{
                "outlet": "RPA_BC"
            },
            "vessel_id": 3, # needs to be changed later
            "vessel_length": 10.0,
            "vessel_name": "branch3_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                # "C": 1 / (config_handler.rpa.C_eq ** -1 - config_handler.rpa.C ** -1), # calculates way too large of a capacitance
                "C": 0.0,
                "L": config_handler.rpa.L_eq - config_handler.rpa.L, # L_RPA_distal
                "R_poiseuille": config_handler.rpa.R_eq - config_handler.rpa.R, # R_RPA_distal
                "stenosis_coefficient": 0.0
            }
        })

        lpa_dist = Vessel.from_config({
            "boundary_conditions":{
                "outlet": "LPA_BC"
            },
            "vessel_id": 4, # needs to be changed later
            "vessel_length": 10.0,
            "vessel_name": "branch4_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                # "C": 1 / (config_handler.lpa.C_eq ** -1 - config_handler.lpa.C ** -1), # calculates way too large of a capacitance
                "C": 0.0,
                "L": config_handler.lpa.L_eq - config_handler.lpa.L, # L_LPA_distal
                "R_poiseuille": config_handler.lpa.R_eq - config_handler.lpa.R, # R_LPA_distal
                "stenosis_coefficient": 0.0
            }
        })

        return cls(config_handler.simparams, 
                   mpa, 
                   lpa_prox, 
                   rpa_prox, 
                   lpa_dist, 
                   rpa_dist, 
                   config_handler.bcs["INFLOW"], 
                   config_handler.bcs[list(config_handler.bcs.keys())[1]].values["Pd"],
                   clinical_targets,
                   steady)


    def to_json(self, output_file):
        '''
        write the config to a json file
        '''

        with open(output_file, 'w') as ff:
            json.dump(self.config, ff)


    def simulate(self):
        '''
        run the simulation with the current config
        '''

        return run_svzerodplus(self.config, dtype='dict')
    

    def initialize_bcs(self, inflow: BoundaryCondition, wedge_p: float):
        '''initialize the boundary conditions for the pa config
        '''

        # initialize the inflow
        if inflow.Q[1] - inflow.Q[0] == 0:
            print('steady inflow, optimizing resistance BCs')
            # assume steady
            self.bcs = {
                "INFLOW": inflow,

                "RPA_BC": BoundaryCondition.from_config({
                    "bc_name": "RPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "R": 1000.0,
                        "Pd": wedge_p
                    }
                }),

                "LPA_BC": BoundaryCondition.from_config({
                    "bc_name": "LPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "R": 1000.0,
                        "Pd": wedge_p
                    }
                })
            }
        else:
            print('unsteady inflow, optimizing RCR BCs')
            # unsteady, need RCR boundary conditions
            self.bcs = {
                "INFLOW": inflow,

                "RPA_BC": BoundaryCondition.from_config({
                    "bc_name": "RPA_BC",
                    "bc_type": "RCR",
                    "bc_values": {
                        "Rp": 100.0,
                        "C": 1e-4,
                        "Rd": 900.0,
                        "Pd": wedge_p
                    }
                }),

                "LPA_BC": BoundaryCondition.from_config({
                    "bc_name": "LPA_BC",
                    "bc_type": "RCR",
                    "bc_values": {
                        "Rp": 100.0,
                        "C": 1e-4,
                        "Rd": 900.0,
                        "Pd": wedge_p
                    }
                })
            }


    def initialize_config_maps(self):
        '''
        initialize the junctions for the pa config
        '''
        
        # change the vessel ids of the proximal vessels

        self.lpa_prox.id = 1
        self.lpa_prox.name = 'branch1_seg0'
        
        self.lpa_dist.id = 2
        self.lpa_dist.name = 'branch2_seg0'
        

        self.rpa_prox.id = 3
        self.rpa_prox.name = 'branch3_seg0'
        
        self.rpa_dist.id = 4
        self.rpa_dist.name = 'branch4' + '_seg0'

        # connect the vessels together
        self.mpa.children = [self.lpa_prox, self.rpa_prox]
        self.lpa_prox.children = [self.lpa_dist]
        self.rpa_prox.children = [self.rpa_dist]

        for vessel in [self.mpa, self.lpa_prox, self.rpa_prox, self.lpa_dist, self.rpa_dist]:
            self.vessel_map[vessel.id] = vessel
        

        for vessel in self.vessel_map.values():
            junction = Junction.from_vessel(vessel)
            if junction is not None:
                self.junctions[junction.name] = junction

        
    def assemble_config(self):
        '''
        assemble the config dict from the config maps
        '''

        # add the boundary conditions
        self._config['boundary_conditions'] = [bc.to_dict() for bc in self.bcs.values()]

        # add the junctions
        self._config['junctions'] = [junction.to_dict() for junction in self.junctions.values()]

        # add the simulation parameters
        self._config['simulation_parameters'] = self.simparams.to_dict()

        # add the vessels
        self._config['vessels'] = [vessel.to_dict() for vessel in self.vessel_map.values()]
        

    def compute_steady_loss(self, R_guess, fun='L2'):
        '''
        compute loss compared to the steady inflow optimization targets
        :param R_f: list of resistances to put into the config
        '''
        blocks_to_optimize = [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]
        for block, R_g in zip(blocks_to_optimize, R_guess):
            block.R = R_g
        # run the simulation
        self.result = self.simulate()

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        # mpa pressure
        self.P_mpa = get_branch_result(self.result, 'pressure_in', 0, steady=True) /  1333.2 

        # rpa pressure
        self.P_rpa = get_branch_result(self.result, 'pressure_out', 1, steady=True) / 1333.2

        # lpa pressure
        self.P_lpa = get_branch_result(self.result, 'pressure_out', 3, steady=True) / 1333.2


        if fun == 'L2':
            loss = np.sum((self.P_mpa - self.clinical_targets.mpa_p) ** 2) + \
                np.sum((self.P_rpa - self.clinical_targets.rpa_p) ** 2) + \
                np.sum((self.P_lpa - self.clinical_targets.lpa_p) ** 2) + \
                np.sum((self.Q_rpa - self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in blocks_to_optimize]) ** 2) # penalize small resistances

        if fun == 'L1':
            loss = np.sum(np.abs(self.P_mpa - self.clinical_targets.mpa_p)) + \
                np.sum(np.abs(self.P_rpa - self.clinical_targets.rpa_p)) + \
                np.sum(np.abs(self.P_lpa - self.clinical_targets.lpa_p)) + \
                np.sum(np.abs(self.Q_rpa - self.clinical_targets.q_rpa))
        
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss
    

    def compute_unsteady_loss(self, R_guess, fun='L2'):
        '''
        compute unsteady loss by adjusting the resistances in the proximal lpa and rpa'''

        blocks_to_optimize = [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]

        self.lpa_prox.R, self.lpa_prox.C, self.rpa_prox.R, self.rpa_prox.C, self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C = R_guess
        
        # run the simulation
        self.result = self.simulate()

        self.result['time'] = np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        self.Q_rpa = trapz(get_branch_result(self.result, 'flow_in', 3, steady=False), self.result['time'])

        # mpa pressure
        P_mpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_in', 0, steady=False)]
        self.P_mpa = [np.max(P_mpa), np.min(P_mpa), np.mean(P_mpa)] # just systolic and mean pressures

        # rpa pressure
        P_rpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 1, steady=False)]
        self.P_rpa = [np.max(P_rpa), np.min(P_rpa), np.mean(P_rpa)]

        # lpa pressure
        P_lpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 3, steady=False)]
        self.P_lpa = [np.max(P_lpa), np.min(P_lpa), np.mean(P_lpa)]

        p_neg_loss = 0

        for p in self.P_mpa + self.P_rpa + self.P_lpa:
            if p < 0:
                p_neg_loss += 10000

        if fun == 'L2':
            loss = np.sum(np.subtract(self.P_mpa, self.clinical_targets.mpa_p) ** 2) + \
                np.sum(np.subtract(self.P_rpa, self.clinical_targets.rpa_p) ** 2) + \
                np.sum(np.subtract(self.P_lpa, self.clinical_targets.lpa_p) ** 2) + \
                100 * np.sum(np.subtract(self.Q_rpa, self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in [self.lpa_prox, self.rpa_prox]]) ** 2) + p_neg_loss
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss


    def compute_unsteady_loss_nonlin(self, R_guess, fun='L2'):
        '''
        compute unsteady loss by adjusting the stenosis coefficient of the proximal lpa and rpa'''

        self.lpa_prox.stenosis_coefficient, self.lpa_prox.C, self.rpa_prox.stenosis_coefficient, self.rpa_prox.C, self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C = R_guess
        
        # run the simulation
        self.result = self.simulate()

        self.result['time'] = np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        self.Q_rpa = trapz(get_branch_result(self.result, 'flow_in', 3, steady=False), self.result['time'])

        # mpa pressure
        P_mpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_in', 0, steady=False)]
        self.P_mpa = [np.max(P_mpa), np.min(P_mpa), np.mean(P_mpa)] # just systolic and mean pressures

        # rpa pressure
        P_rpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 1, steady=False)]
        self.P_rpa = [np.max(P_rpa), np.min(P_rpa), np.mean(P_rpa)]

        # lpa pressure
        P_lpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 3, steady=False)]
        self.P_lpa = [np.max(P_lpa), np.min(P_lpa), np.mean(P_lpa)]

        p_neg_loss = 0

        for p in self.P_mpa + self.P_rpa + self.P_lpa:
            if p < 0:
                p_neg_loss += 10000

        if fun == 'L2':
            loss = np.sum(np.subtract(self.P_mpa, self.clinical_targets.mpa_p) ** 2) + \
                np.sum(np.subtract(self.P_rpa, self.clinical_targets.rpa_p) ** 2) + \
                np.sum(np.subtract(self.P_lpa, self.clinical_targets.lpa_p) ** 2) + \
                100 * np.sum(np.subtract(self.Q_rpa, self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in [self.lpa_prox, self.rpa_prox]]) ** 2) + p_neg_loss
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss


    def optimize(self, steady=True, nonlin=False):
        '''
        optimize the resistances in the pa config
        '''

        # self.to_json('pa_config_pre_opt.json')
        # define optimization bounds [0, inf)
        bounds = Bounds(lb=0, ub=math.inf)

        if steady:
            result = minimize(self.compute_steady_loss, 
                                [obj.R for obj in [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]], 
                                method="Nelder-Mead", bounds=bounds)
        else:
            if nonlin:
                initial_guess = [self.lpa_prox.stenosis_coefficient, self.lpa_prox.C, self.rpa_prox.stenosis_coefficient, self.rpa_prox.C, 
                                 self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, 
                                 self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C]
                result = minimize(self.compute_unsteady_loss_nonlin, 
                                initial_guess, 
                                method="Nelder-Mead", bounds=bounds)
            else:
                initial_guess = [self.lpa_prox.R, self.lpa_prox.C, self.rpa_prox.R, self.rpa_prox.C, 
                                self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, 
                                self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C]
                result = minimize(self.compute_unsteady_loss, 
                                    initial_guess, 
                                    method="Nelder-Mead", bounds=bounds)

        print([self.Q_rpa / self.clinical_targets.q, self.P_mpa, self.P_lpa, self.P_rpa])




    @property
    def config(self):
        self.assemble_config()
        return self._config

        

