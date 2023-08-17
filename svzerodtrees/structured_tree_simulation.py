import copy

from svzerodsolver import runner
import csv
from pathlib import Path
import numpy as np
import json
from struct_tree_utils import *
from post_processing.stree_data_processing import *
from post_processing.stree_visualization import *
from scipy.optimize import minimize, Bounds
from structuredtreebc import StructuredTreeOutlet
import os
import matplotlib.pyplot as plt



def optimize_preop_0d(clinical_targets: csv,
                      input_file,
                      log_file,
                      working_dir,
                      make_steady=False,
                      unsteady=False,
                      change_to_R=False):
    '''

    :param clinical_targets: clinical targets input csv
    :param input_file: 0d solver json input file name string
    :param output_file: 0d solver json output file name string
    :return: preop simulation with optimized BCs
    '''
    # get the clinical target values
    with open(log_file, "a") as log:
        log.write("Getting clinical target values... \n")
    bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
    cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
    q = bsa * cardiac_index # cardiac output in L/min
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
        zeroD_config = json.load(ff)

    if make_steady:
        make_inflow_steady(zeroD_config)
        with open(log_file, "a") as log:
            log.write("inlet BCs converted to steady \n")

    if change_to_R:
        Pd = convert_RCR_to_R(zeroD_config)
        with open(log_file, "a") as log:
            log.write("RCR BCs converted to R, Pd = " + str(Pd) + "\n")

    # get resistances from the zerod input file
    resistance = get_resistances(zeroD_config)
    # get the LPA and RPA branch numbers
    lpa_rpa_branch = ["V" + str(idx) for idx in zeroD_config["junctions"][0]["outlet_vessels"]]

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(r,
                                     input_config=zeroD_config,
                                     target_ps=None,
                                     unsteady=unsteady,
                                     lpa_rpa_branch=lpa_rpa_branch
                                     ):
        # r = abs(r)
        # r = [r, r]
        write_resistances(input_config, r)
        zerod_result = runner.run_from_config(input_config)
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_mpa_pressure(zerod_result, branch_name='V0') # get mpa pressure

        # lpa_rpa_branch = ["V" + str(idx) for idx in input_config["junctions"][0]["outlet_vessels"]]

        q_MPA = get_df_data(zerod_result, branch_name='V0', data_name='flow_in')
        q_RPA = get_df_data(zerod_result, branch_name=lpa_rpa_branch[0], data_name='flow_in')
        q_LPA = get_df_data(zerod_result, branch_name=lpa_rpa_branch[1], data_name='flow_in')
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
        obj_fun.append(min_obj)
        plot_optimization_progress(obj_fun)
        return min_obj

    # write to log file for debugging
    with open(log_file, "a") as log:
        log.write("Optimizing preop outlet resistance... \n")
    # run the optimization algorithm
    result = minimize(zerod_optimization_objective,
                      resistance,
                      args=(zeroD_config, target_ps),
                      method="CG",
                      options={"disp": False},
                      )
    log_optimization_results(log_file, result, '0D optimization')
    # write to log file for debugging
    with open(log_file, "a") as log:
        log.write("Outlet resistances optimized! " + str(result.x) +  "\n")

    R_final = result.x # get the array of optimized resistances
    write_resistances(zeroD_config, R_final)

    with open(str(working_dir) + '/preop_config.in', "w") as ff:
        json.dump(zeroD_config, ff)

    return zeroD_config, R_final


def construct_trees(config: dict, log_file=None, vis_trees=False, fig_dir=None):
    roots = []
    zerod_result = runner.run_from_config(config)
    q_out = get_outlet_data(config, zerod_result, 'flow_out', steady=True)
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                outlet_tree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, config["simulation_parameters"])
                # R = resistances[get_resistance_idx(vessel_config)]
                for bc in config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc["bc_name"]:
                        R = bc["bc_values"]["R"]
                # write to log file for debugging
                if log_file is not None:
                    with open(log_file, "a") as log:
                        log.write("** building tree for resistance: " + str(R) + " ** \n")
                # outlet_tree.optimize_tree_radius(R)
                outlet_tree.optimize_tree_radius(R, log_file)
                # write to log file for debugging
                if log_file is not None:
                    with open(log_file, "a") as log:
                        log.write("     the number of vessels is " + str(len(outlet_tree.block_dict["vessels"])) + "\n")
                vessel_config["tree"] = outlet_tree.block_dict
                roots.append(outlet_tree.root)

    # if vis_trees:
    #     visualize_trees(config, roots, fig_dir=fig_dir, fig_name='_preop')

    return roots


def calculate_flow(preop_config: dict, repair=None, repair_degree=1, log_file=None):
    preop_result = runner.run_from_config(preop_config)
    postop_config = copy.deepcopy(preop_config)  # make sure not to edit the preop_config
    lpa_rpa_branch = [idx for idx in preop_config["junctions"][0]["outlet_vessels"]]
    with open(log_file, "a") as log:
        log.write("     LPA and RPA branches identified: " + str(lpa_rpa_branch))
    repair_vessels=None
    # repair_vessels based on desired strategy: extensive, proximal or custom
    if repair == 'proximal':
        repair_vessels = lpa_rpa_branch
    elif repair == 'extensive':
        pass
    else:
        repair_vessels=repair
    repair_stenosis(postop_config, repair_vessels, repair_degree, log_file=log_file)
    postop_result = runner.run_from_config(postop_config)

    return preop_result, postop_result, postop_config


def adapt_trees(config, preop_roots, preop_result, postop_result, pries_secomb = False):
    preop_q = get_outlet_data(config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(config, postop_result, 'flow_out', steady=True)
    # q_diff = [postop_q[i] - q_old for i, q_old in enumerate(preop_q)]
    adapted_config = config
    outlet_idx = 0 # index through outlets
    R_new = []
    roots = copy.deepcopy(preop_roots)
    adapted_roots = []
    for vessel_config in adapted_config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for root in roots:
                    if root.name in vessel_config["tree"]["name"]:
                        outlet_tree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, config["simulation_parameters"], tree_exists=True, root=root)
                        if pries_secomb:
                            # ps_params in the form
                            pass
                        else: # constant wss adaptation
                            R_new.append(outlet_tree.adapt_constant_wss(preop_q[outlet_idx], postop_q[outlet_idx], disp=False))
                        

                        adapted_roots.append(outlet_tree.root)
                        vessel_config["tree"] = outlet_tree.block_dict
                        outlet_idx +=1
                        # adapted_roots.append(outlet_tree.root)

    write_resistances(adapted_config, R_new)
    return adapted_config, adapted_roots



def run_final_flow(config, preop_result, postop_result, output_file, summary_result_file, log_file, condition: str=None):

    final_result = runner.run_from_config(config)
    with open(log_file, "a") as log:
        log.write("Writing result to file... \n")
    result = {'name': 'results ' + condition, 'data': final_result.to_dict('index')}
    with open(output_file, "w") as ff:
        json.dump(result, ff)

    summ_results = {condition: {}}
    # get the flowrates preop, postop, and post adaptation
    preop_q = get_outlet_data(config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(config, postop_result, 'flow_out', steady=True)
    final_q = get_outlet_data(config, final_result, 'flow_out', steady=True)
    summ_results[condition]['q'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}
    # get the pressures preop, postop and post adaptation
    preop_p = get_outlet_data(config, preop_result, 'pressure_out', steady=True)
    postop_p = get_outlet_data(config, postop_result, 'pressure_out', steady=True)
    final_p = get_outlet_data(config, final_result, 'pressure_out', steady=True)
    summ_results[condition]['p'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}
    # get the wall shear stress at the outlet
    preop_wss = get_outlet_data(config, preop_result, 'wss', steady=True)
    postop_wss = get_outlet_data(config, postop_result, 'wss', steady=True)
    final_wss = get_outlet_data(config, final_result, 'wss', steady=True)
    summ_results[condition]['wss'] = {'preop': preop_wss, 'postop': postop_wss, 'final': final_wss}

    with open(summary_result_file, "a") as sr:
        json.dump(summ_results, sr)



    with open(log_file, "a") as log:
        log.write("** RESULT COMPARISON ** \n")
        log.write("     preop outlet flowrates: " + str(preop_q) + "\n")
        log.write("     pre-adaptation outlet flowrates: " + str(postop_q) + "\n")
        log.write("     post-adaptation outlet flowrates: " + str(final_q) + "\n \n")
        log.write("Simulation completed!")

    return summ_results

def generate_results(result_df):
    '''
    generate a reduced results dict
    Args:
        result_df:

    Returns:

    '''
    pass


# Press the green button in the gutter to run the script.


def run_simulation(model_dir: str, expname: str, optimized: bool=False, vis_trees: bool=False):
    '''
    run a structured tree simulation with experimental conditions
    Args:
        test_dir_str: script directory
        model_dir: model directory
        exp_file: name of experiment txt file

    Returns:

    '''
    # make path variable, starting from script dir
    os.chdir('structured_trees/models') # cd into models dir
    modeldir=Path(model_dir) # make pathlib variable
    # experiment name
    exp_filename = expname + '.txt'
    # load experiment file
    if optimized:
        os.system(
            'mv ' + model_dir + '/' + expname + '/' + exp_filename + ' ' + model_dir)  # move the experiment file into the experiment directory
    with open(modeldir / exp_filename) as ff:
        expfile = json.load(ff) # load experimental condition file

    # make experiment directory
    expfile['name'] = expname # get experiment name
    repair = expfile['repair type'] # get repair type
    repair_degrees = expfile['repair degrees'] # get repair degrees

    os.system('pwd') # check that we are in the correct directory to start the experiment
    os.system('mkdir ' + model_dir + '/' + expname) # make the experiment directory
    os.system('mkdir ' + model_dir + '/' + expname + '/figures') # make the figures directory

    exp_dir = model_dir + '/'+ expname # string
    expdir = modeldir / expname # path variable
    input_file = modeldir / '{}.in'.format(model_dir)
    log_file = expdir / '{}.log'.format(model_dir + '-' + expname)
    summary_result_file = expdir / '{}_summary_results.txt'.format(model_dir + '-' + expname)
    fig_dir = expdir / 'figures'
    with open(log_file, "w") as log:
        log.write("Experiment directory and log file created. Beginning experiment " + expname +"...  \n")

    if not optimized:
        preop_config, R_final = optimize_preop_0d('clinical_targets.csv',
                                            input_file,
                                            log_file,
                                            exp_dir,
                                            make_steady=False,
                                            unsteady=False,
                                            change_to_R=True)
    else:
        with open(log_file, "a") as log:
            log.write("Outlets have already been optimized. Constructing trees...  \n")
        with open(exp_dir + '/preop_config.in') as ff:
            preop_config = json.load(ff)

    preop_roots = construct_trees(preop_config, log_file, vis_trees=vis_trees, fig_dir=str(fig_dir))
    # calculate and visualize repair results
    for i, degree in enumerate(repair_degrees):
        condition = 'repair_' + str(degree)
        output_file = expdir / '{}.out'.format(model_dir + '-' + expname)
        preop_result, postop_result, postop_config = calculate_flow(preop_config, repair=repair, repair_degree=degree, log_file=log_file)
        adapted_config, postop_roots = adapt_trees(postop_config, preop_roots, preop_result, postop_result)
        # write the config to a file for observation
        with open(str(expdir / "adapted_config") + "_" + str(degree) + ".txt", "w") as ff:
            json.dump(adapted_config, ff)
        condensed_results = run_final_flow(adapted_config,
                       preop_result,
                       postop_result,
                       output_file,
                       summary_result_file,
                       log_file,
                       condition)
        plot_LPA_RPA_changes_subfigs(fig_dir, condensed_results, 'flow changes in LPA vs RPA', condition=condition)

        if not optimized:
            plot_optimization_progress(obj_fun, save=True, path=str(fig_dir))  # save the plot of the objective function

        if vis_trees:
            visualize_trees(preop_config, adapted_config, preop_roots, postop_roots, fig_dir=str(fig_dir), fig_name=condition)

    print('Experiment complete!')
    os.system('mv ' + model_dir + '/' + exp_filename + ' ' + model_dir + '/' + expname)  # move the experiment file into the experiment directory
