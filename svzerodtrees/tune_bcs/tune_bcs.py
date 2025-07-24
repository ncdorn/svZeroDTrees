import csv
import numpy as np
from scipy.optimize import minimize, Bounds
import math
from multiprocessing import Pool

from ..io import *
from ..io.utils import get_branch_result
from ..io.result_handler import ResultHandler # need to deprecate this
from ..utils import *
from ..utils import write_resistances
from ..simulation.threedutils import vtp_info, get_coupled_surfaces, find_vtp_area
from ..microvasculature import StructuredTree

from .clinical_targets import ClinicalTargets
from .utils import *
from . import PAConfig


def construct_impedance_trees(config_handler, mesh_surfaces_path, wedge_pressure, d_min = 0.1, convert_to_cm=False, is_pulmonary=True, tree_params={'lpa': [19992500, -35, 0.0, 50.0], 
                                                                                                                                                    'rpa': [19992500, -35, 0.0, 50.0]},
                                                                                                                                                    n_procs=24,
                                                                                                                                                    use_mean=False,
                                                                                                                                                    specify_diameter=False):
    '''
    construct impedance trees for outlet BCs
    
    :param k2: stiffness parameter 2
    :param k3: stiffness parameter 3'''

    # get outlet areas
    if is_pulmonary:
        rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True)

        cap_info = lpa_info | rpa_info
    else:
        cap_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=False)
    
    # get the mean and standard deviation of the cap areas
    # lpa_areas = np.array(list(lpa_info.values()))
    # rpa_areas = np.array(list(rpa_info.values()))

    outlet_bc_names = [name for name, bc in config_handler.bcs.items() if 'inflow' not in bc.name.lower()]

    # assumed that cap and boundary condition orders match
    if len(outlet_bc_names) != len(cap_info):
        print('number of outlet boundary conditions does not match number of cap surfaces, automatically assigning bc names...')
        for i, name in enumerate(outlet_bc_names):
            # delete the unused bcs
            del config_handler.bcs[name]
        outlet_bc_names = [f'IMPEDANCE_{i}' for i in range(len(cap_info))]
    cap_to_bc = {list(cap_info.keys())[i]: outlet_bc_names[i] for i in range(len(outlet_bc_names))}

    if use_mean:
        '''use the mean diameter of the cap surfaces to construct the lpa and rpa trees and use these trees for all outlets'''
        if specify_diameter:
            k1_l, k2_l, k3_l, lrr_l, lpa_mean_dia = tree_params['lpa']
            k1_r, k2_r, k3_r, lrr_r, rpa_mean_dia = tree_params['rpa']

        else:
            lpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in lpa_info.values()])
            rpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in rpa_info.values()])

            lpa_std_dia = np.std([(area / np.pi)**(1/2) * 2 for area in lpa_info.values()])
            rpa_std_dia = np.std([(area / np.pi)**(1/2) * 2 for area in rpa_info.values()])

            print(f'LPA mean diameter: {lpa_mean_dia}')
            print(f'RPA mean diameter: {rpa_mean_dia}')
            print(f'LPA std diameter: {lpa_std_dia}')
            print(f'RPA std diameter: {rpa_std_dia}')

            k1_l, k2_l, k3_l, lrr_l = tree_params['lpa']

            k1_r, k2_r, k3_r, lrr_r = tree_params['rpa']

        time_array = config_handler.inflows[next(iter(config_handler.inflows))].t

        lpa_tree = StructuredTree(name='LPA', time=time_array, simparams=config_handler.simparams)
        print(f'building LPA tree with lpa parameters: {tree_params["lpa"]}')
        

        lpa_tree.build_tree(initial_d=lpa_mean_dia, d_min=d_min, lrr=lrr_l)
        lpa_tree.compute_olufsen_impedance(k2=k2_l, k3=k3_l, n_procs=n_procs)
        lpa_tree.plot_stiffness(path='lpa_stiffness_plot.png')

        # add tree to config handler
        config_handler.tree_params[lpa_tree.name] = lpa_tree.to_dict()

        rpa_tree = StructuredTree(name='RPA', time=time_array, simparams=config_handler.simparams)
        print(f'building RPA tree with rpa parameters: {tree_params["rpa"]}')

        rpa_tree.build_tree(initial_d=rpa_mean_dia, d_min=d_min, lrr=lrr_r)
        rpa_tree.compute_olufsen_impedance(k2=k2_r, k3=k3_r, n_procs=n_procs)
        rpa_tree.plot_stiffness(path='rpa_stiffness_plot.png')

        # add tree to config handler
        print(rpa_tree.to_dict())
        config_handler.tree_params[rpa_tree.name] = rpa_tree.to_dict()

        # distribute the impedance to lpa and rpa specifically
        for idx, (cap_name, area) in enumerate(cap_info.items()):
            print(f'generating tree {idx + 1} of {len(cap_info)} for cap {cap_name}...')
            if 'lpa' in cap_name.lower():
                config_handler.bcs[cap_to_bc[cap_name]] = lpa_tree.create_impedance_bc(cap_to_bc[cap_name], 0, wedge_pressure * 1333.2)
            elif 'rpa' in cap_name.lower():
                config_handler.bcs[cap_to_bc[cap_name]] = rpa_tree.create_impedance_bc(cap_to_bc[cap_name], 1, wedge_pressure * 1333.2)
            else:
                raise ValueError('cap name not recognized')
            
    else:
        '''build a unique tree for each outlet'''
        for idx, (cap_name, area) in enumerate(cap_info.items()):

            print(f'generating tree {idx} of {len(cap_info)} for cap {cap_name}...')
            cap_d = (area / np.pi)**(1/2) * 2

            tree = StructuredTree(name=cap_name, time=config_handler.bcs['INFLOW'].t, simparams=config_handler.simparams)
            if 'lpa' in cap_name.lower():
                print(f'building tree with lpa parameters: {tree_params["lpa"]}')
                k1, k2, k3, lrr = tree_params['lpa']
            elif 'rpa' in cap_name.lower():
                print(f'building tree with rpa parameters: {tree_params["rpa"]}')
                k1, k2, k3, lrr = tree_params['rpa']
            else:
                raise ValueError('cap name not recognized')
            tree.build_tree(initial_d=cap_d, d_min=d_min, lrr=lrr)

            # compute the impedance in frequency domain
            tree.compute_olufsen_impedance(k2=k2, k3=k3, n_procs=n_procs)

            # add tree to config handler
            config_handler.tree_params[tree.name] = tree.to_dict()

            bc_name = cap_to_bc[cap_name]

            config_handler.bcs[bc_name] = tree.create_impedance_bc(bc_name, idx, wedge_pressure * 1333.2)


def optimize_impedance_bcs(config_handler, mesh_surfaces_path, clinical_targets, opt_config_path='optimized_impedance_config.json', n_procs=24, log_file=None, d_min=0.01, tol=0.01, is_pulmonary=True, convert_to_cm=True):
    '''
    tune the parameters of the impedance trees to match clinical targets
    currently, we tune k2_l, k2_r, lpa_mean_dia, rpa_mean_dia, l_rr'''

    # get mean outlet area
    if is_pulmonary:
        rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True)
    
    # rpa_mean_dia = 0.32
    rpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in rpa_info.values()])
    print(f'RPA mean diameter: {rpa_mean_dia}')
    # lpa_mean_dia = 0.32
    lpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in lpa_info.values()])
    print(f'LPA mean diameter: {lpa_mean_dia}')

    # rpa_total_area = sum(rpa_info.values())
    # lpa_total_area = sum(lpa_info.values())

    # rpa_mean_dia = (rpa_total_area / np.pi)**(1/2) * 2
    # lpa_mean_dia = (lpa_total_area / np.pi)**(1/2) * 2

    # print(f'RPA total diameter: {rpa_mean_dia}')
    # print(f'LPA total diameter: {lpa_mean_dia}')

    # create MPA/LPA/RPA simple config
    if len(config_handler.vessel_map.values()) == 5:
        # we have an already simplified config
        pa_config = PAConfig.from_pa_config(config_handler, clinical_targets)
    else:
        pa_config = PAConfig.from_config_handler(config_handler, clinical_targets)
        if convert_to_cm:
            pa_config.convert_to_cm()
    # rescale inflow by number of outlets ## TODO: figure out scaling for this
    pa_config.bcs['INFLOW'].Q = [q / ((len(lpa_info.values()) + len(rpa_info.values())) // 2) for q in pa_config.bcs['INFLOW'].Q]

    def tree_tuning_objective(params, clinical_targets, lpa_mean_dia, rpa_mean_dia, d_min, n_procs):
        '''
        params: [k1_l, k1_r, k2_l, k2_r, lrr_l, lrr_r, alpha]'''

        # k1_l = params[0]
        # k1_r = params[1]
        # k2_l = params[2]
        # k2_r = params[3]
        # lrr_l = params[4]
        # lrr_r = params[5]

        k1_l = 19992500.0
        k1_r = 19992500.0
        k2_l = params[0]
        k2_r = params[1]
        k3_l = 0.0
        k3_r = 0.0
        d_min = [d_min, d_min] # optimize d_min this time
        lpa_mean_dia = params[2] # this time we vary input diameter
        rpa_mean_dia = params[3]
        lrr_l = params[4]
        lrr_r = params[4]
        # lrr_l = 10.0
        # lrr_r = 10.0
        # xi = params[2]

        # for sheep, try with fixed l_rr (10.0) and optimize other parameters!!

        

        alpha = 0.9 # actual value alpha: 0.9087043307650987, beta: 0.5781530881973255
        beta = 0.6

        tree_params = {
            'lpa': [k1_l, k2_l, k3_l, lrr_l, alpha, beta],
            'rpa': [k1_r, k2_r, k3_r, lrr_r, alpha, beta]
        }

        ### WITH xi
        # tree_params = {
        #     'lpa': [k1_l, k2_l, k3_l, lrr_l, xi],
        #     'rpa': [k1_r, k2_r, k3_r, lrr_r, xi]
        # }

        pa_config.create_impedance_trees(lpa_mean_dia, rpa_mean_dia, d_min, tree_params, n_procs)

        pa_config.to_json(f'pa_config_test_tuning.json')

        if pa_config.bcs['LPA_BC'].Z[0] != pa_config.bcs['LPA_BC'].Z[0]:
            print('NaN in LPA impedance')
            pressure_loss = 5e5
            flowsplit_loss = 5e5
        elif pa_config.bcs['RPA_BC'].Z[0] != pa_config.bcs['RPA_BC'].Z[0]:
            print('\n\nNaN in RPA impedance\n\n')
            pressure_loss = 5e5
            flowsplit_loss = 5e5
        else:
            try:
                pa_config.simulate()

                print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n params: {params}')

                pa_config.plot_mpa(path='figures/pa_config_plot.png')

                if clinical_targets.mpa_p[1] < clinical_targets.wedge_p:
                    # diastolic pressure < wedge pressure so we neglect it in the optimization
                    weights = np.array([1, 0, 1])
                else:
                    weights = np.array([1.5, 1, 1.2])
                
                pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(clinical_targets.mpa_p)) / clinical_targets.mpa_p, weights)) ** 2 * 100

                flowsplit_loss = ((pa_config.rpa_split - clinical_targets.rpa_split) / clinical_targets.rpa_split) ** 2 * 100

                # write parameters to csv file
                # write the optimized results + params to file
                    
            except:
                pressure_loss = 5e5
                flowsplit_loss = 5e5

        loss = pressure_loss + flowsplit_loss

        with open('optimized_params.csv', 'w') as f:
            f.write("pa,k1,k2,k3,lrr,diameter,loss,flow_split,p_mpa\n")
            # do unique for lpa, rpa
            f.write(f'lpa,{k1_l},{k2_l},{k3_l},{lrr_l},{lpa_mean_dia},{loss},{1 - pa_config.rpa_split},[{pa_config.P_mpa[0]} {pa_config.P_mpa[1]} {pa_config.P_mpa[2]}]\n')
            f.write(f'rpa,{k1_r},{k2_r},{k3_r},{lrr_r},{rpa_mean_dia},{loss},{pa_config.rpa_split},[{pa_config.P_mpa[0]} {pa_config.P_mpa[1]} {pa_config.P_mpa[2]}]\n')
        
        print(f'\n***PRESSURE LOSS: {pressure_loss}, FS LOSS: {flowsplit_loss}, TOTAL LOSS: {loss} ***\n')

        return loss


    # bounds = Bounds(lb=[0.0, 0.0, -np.inf,-np.inf, 10.0, 10.0], ub= [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    ### WITH ALPHA
    # bounds = Bounds(lb=[-np.inf,-np.inf, 10.0, 10.0, 0.0], ub= [np.inf, np.inf, np.inf, np.inf, np.inf])
    ### WITHOUT ALPHA

    # need to implement grid search for stiffnesses...
    l_rr_guess = 10.0
    print("performing search for best k2 stiffness...")
    # k2_search = [-50]
    k2_search = [-10, -25, -50, -75]
    min_loss = 1e5
    k2_opt = 0
    for k2 in k2_search:
        loss = tree_tuning_objective([k2, k2, lpa_mean_dia, rpa_mean_dia, l_rr_guess], clinical_targets, lpa_mean_dia, rpa_mean_dia, d_min, n_procs)
        print(f'k2: {k2}, loss: {loss}')
        if loss < min_loss:
            min_loss = loss
            k2_opt = k2

    # print(f'optimal k2: {k2_opt} with loss {min_loss}')

    # bounds for optimization
    bounds = Bounds(lb=[-np.inf, -np.inf, 0.01, 0.01, 1.0])

    # result = minimize(tree_tuning_objective, [2e7, 2e7, -30, -30, 50.0, 50.0], args=(clinical_targets, lpa_mean_dia, rpa_mean_dia, d_min, n_procs), method='Nelder-Mead', bounds=bounds, tol=1.0)
    ### WITH ALPHA
    # result = minimize(tree_tuning_objective, [-30, -30, 66.0, 66.0, 2.7], args=(clinical_targets, lpa_mean_dia, rpa_mean_dia, d_min, n_procs), method='Nelder-Mead', bounds=bounds, tol=1.0)
    ### WITHOUT ALPHA
    result = minimize(tree_tuning_objective, [k2_opt, k2_opt, lpa_mean_dia, rpa_mean_dia, l_rr_guess], args=(clinical_targets, lpa_mean_dia, rpa_mean_dia, d_min, n_procs), method='Nelder-Mead', bounds=bounds, options={'maxiter': 100})

    # format of result.x: [k2_l, k2_r, lrr_l, lrr_r]
    print(f'Optimized parameters: {result.x}')

    # simulate the final pa config
    pa_config.simulate()

    pa_config.plot_mpa()


### NOT INCLUDED IN INIT ###
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
                
                config_handler.trees[bc.name] = outlet_tree


    
    # function to run the tree diameter optimization
    def optimize_tree(tree):
        print('building ' + tree.name + ' for resistance ' + str(tree.params["bc_values"]["R"]) + '...')
        tree.optimize_tree_diameter(log_file=log_file, d_min=d_min)
        return tree

    # run the tree 
    with Pool(n_procs) as p:
        # TODO: fix this
        config_handler.trees = p.map(optimize_tree, list(config_handler.trees.values()))
    
    # update the resistance in the config according to the optimized tree resistance
    for bc, tree in zip(list(config_handler.bcs.values())[1:], list(config_handler.trees.values())):
        bc.R = tree.root.R_eq


    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTree instances based on the preop result
    # config_handler.update_stree_hemodynamics(preop_result)

    result_handler.add_unformatted_result(preop_result, 'preop')


def construct_coupled_cwss_trees(config_handler, simulation_dir, n_procs=4, d_min=.0049):
    '''
    construct cwss trees for a 3d coupled BC'''

    coupled_surfs = get_coupled_surfaces(simulation_dir)

    for coupling_block in config_handler.coupling_blocks.values():
        coupling_block.surface = coupled_surfs[coupling_block.name]

    for bc in config_handler.bcs.values():
        if config_handler.coupling_blocks[bc.name].location == 'inlet':
            diameter = (find_vtp_area(config_handler.coupling_blocks[bc.name].surface) / np.pi)**(1/2) * 2
            config_handler.trees[bc.name] = StructuredTree.from_bc_config(bc, config_handler.simparams, diameter)


    # function to run the tree diameter optimization
    def optimize_tree(tree):
        tree.optimize_tree_diameter(d_min=d_min)
        return tree


    # run the tree 
    with Pool(n_procs) as p:
        config_handler.trees = p.map(optimize_tree, list(config_handler.trees.values()))
    

    # update the resistance in the config according to the optimized tree resistance
    outlet_idx = 0 # linear search, i know. its bad. will fix later
    for bc in config_handler.bcs.values():
        # we assume that an inlet location indicates taht this is an outlet bc and threfore undergoes adaptation
        if config_handler.coupling_blocks[bc.name].location == 'inlet':
            if bc.type == 'RCR':
                bc.Rp = config_handler.trees[bc.name].root.R_eq * 0.1
                bc.Rd = config_handler.trees[bc.name].root.R_eq * 0.9
            elif bc.type == 'RESISTANCE':
                bc.R = config_handler.trees[bc.name].root.R_eq
            outlet_idx += 1

class BCTuner:
    '''
    Class to handle the boundary condition tuning for the SVZeroDtrees model.
    '''

    #TODO: implement later after we have a robust test setup.

    pass


