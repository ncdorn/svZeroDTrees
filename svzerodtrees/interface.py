import os
import sys
import json
import pickle
from svzerodtrees import preop, operation, adaptation, postop
from svzerodtrees.post_processing import plotting
from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees.post_processing.pa_plotter import PAanalyzer
from scipy.optimize import minimize

def run_from_file(exp_config_file: str, optimized: bool=False, vis_trees: bool=True):
    '''
    run the structured tree optimization pipeline from an experiment config file

    :param exp_config_file: path to the experiment config file
    :param optimized: True if the outlet bcs have previously been optimized. Use the optimized config (preop_config.in)
    :param vis_trees: if true, make tree visualization figures
    '''
    
    # start off somewhere in the models directory, same level as the experiment config file
    with open(exp_config_file) as ff:
        exp_config = json.load(ff)

    # unpack the experiment config parameters
    expname = exp_config["name"]
    modelname = exp_config["model"]
    adapt = exp_config["adapt"] # either ps (pries and secomb) or cwss (constant wall shear stress)
    optimized = exp_config["optimized"] # true if the experiment has been optimized before, to skip the preop optimization
    is_full_pa = exp_config["is_full_pa_tree"]
    trees_exist = exp_config["trees_exist"]
    mesh_surfaces_path = exp_config["mesh_surfaces_path"]
    repair_config = exp_config["repair"]

    # check if we are in an experiments directory, if not assume we are in it

    if os.path.exists('../experiments'): # we are in the experiments directory already

        if os.path.exists(expname):
            ovwrite_dir = input('experiment ' + expname + ' already exists, do you want to overwrite? (y/n) ')

            if ovwrite_dir == 'y':
                pass

            else:
                sys.exit()
        
        else:
            os.system('mkdir ' + expname)

    elif os.path.exists('experiments'): # we are in the directory above the experiments directory
        
        if os.path.exists(expname):
            ovwrite_dir = input('experiment ' + expname + ' already exists, do you want to overwrite? (y/n) ')
            
            if ovwrite_dir == 'y':
                pass

            else:
                sys.exit()
        
        else:
            os.chdir('experiments')
            os.system('mkdir ' + expname)

    else: # we are not in the experiments directory and need to create one

        print('no experiments directory for this model, creating one...')

        os.system('mkdir experiments')
        os.system('mv ' + exp_config_file + ' experiments')
        os.chdir('experiments')
        # make the experiments directory
        os.system('mkdir ' + expname)
    
    # move the experiment config file into the experiment directory
    # os.system('mv ' + exp_config_file + ' ' + expname)
    
    # cd into the models directory and delineate exp directory path
    os.chdir('../')
    expdir_path = 'experiments/' + expname + '/'

    # if visualizing trees, make a figures directory
    if vis_trees:

        # define figure directory path
        fig_dir = expdir_path + '/figures'

        # make figure direcotry if it does not exist
        if not os.path.exists(fig_dir):
            os.system('mkdir ' + fig_dir)
    else:
        fig_dir=None

    # delineate important file names
    input_file = modelname + '.json'
    clinical_targets = 'clinical_targets.csv'
    log_file = expdir_path + expname + '.log'

    # initialize log file
    write_to_log(log_file, 'beginning experiment ' + expname + '!',  write=True)
    write_to_log(log_file, 'with the following configuration: ')
    write_to_log(log_file, str(exp_config))

    # optimize preoperative outlet boundary conditions
    if not optimized:
        if is_full_pa:
            config_handler, result_handler, pa_config = preop.optimize_pa_bcs(
                input_file,
                mesh_surfaces_path,
                clinical_targets,
                log_file
            )
        else:
            config_handler, result_handler = preop.optimize_outlet_bcs(
                input_file,
                clinical_targets,
                log_file,
                show_optimization=False
            )

        # save optimized config and result
        config_handler.to_json('preop_config.json')
        

    else: # use previous optimization results
        if trees_exist:
            # load the pickled config handler
            if adapt == 'cwss':
                with open('config_w_cwss_trees.in', 'rb') as ff:
                    config_handler = pickle.load(ff)
            elif adapt == 'ps':
                with open('config_w_ps_trees.in', 'rb') as ff:
                    config_handler = pickle.load(ff)
            
        else:
            config_handler = ConfigHandler.from_json('preop_config.json')
        
        # get preop result
        preop_flow = run_svzerodplus(config_handler.config)

        # load a new result handler
        result_handler = ResultHandler.from_config(config_handler.config)

        # add result to the handler
        result_handler.add_unformatted_result(preop_flow, 'preop')


    if repair_config[0]['type'] == 'optimize stent':
        print("optimizing stent diameter...")
        optimize_stent_diameter(config_handler,
                                result_handler,
                                repair_config[0],
                                adapt,
                                n_procs=12,
                                trees_exist=trees_exist)
    else:
        if adapt == 'ps': # use pries and secomb adaptation scheme
            
            run_pries_secomb_adaptation(config_handler, 
                                        result_handler, 
                                        repair_config, 
                                        log_file,
                                        n_procs=12,
                                        trees_exist=trees_exist)

        elif adapt == 'cwss': # use constant wall shear stress adaptation scheme
            
            run_cwss_adaptation(config_handler, 
                                result_handler, 
                                repair_config, 
                                log_file,
                                n_procs=12,
                                trees_exist=trees_exist)

        else:
            raise Exception('invalid adaptation scheme chosen')
    
    # format the results
    result_handler.format_results()

    # save the adapted config
    config_handler.to_json_w_trees(expdir_path + 'adapted_config.json')
    
    # save the result
    result_handler.to_json(expdir_path + 'full_results.json')

    # save the result handler to use later
    result_handler.to_file(expdir_path + 'result_handler.out')
    
    if vis_trees:
        # initialize the data plotter
        plotter = PAanalyzer.from_files('preop_config.json', expdir_path + 'full_results.json', expdir_path + 'figures/')

        # scatter outflow vs. distance
        plotter.scatter_qoi_adaptation_distance('all', 'q_out')
        plotter.scatter_qoi_adaptation_distance('outlets', 'q_out', filename='adaptation_scatter_outlets.png')

        # scatter pressure vs. distance
        plotter.scatter_qoi_adaptation_distance('all', 'p_out')
        plotter.scatter_qoi_adaptation_distance('outlets', 'p_out', filename='adaptation_scatter_outlets.png')

        # scatter wss vs. distance
        plotter.scatter_qoi_adaptation_distance('all', 'wss')
        plotter.scatter_qoi_adaptation_distance('outlets', 'wss', filename='adaptation_scatter_outlets.png')

        # plot lpa and rpa flow adaptation
        plotter.plot_lpa_rpa_adaptation()

        # plot lpa and rpa changes
        plotter.plot_lpa_rpa_diff()

        # plot the mpa pressure changes
        plotter.plot_mpa_pressure()


def run_pries_secomb_adaptation(config_handler: ConfigHandler, result_handler, repair_config, log_file, n_procs=12, trees_exist=False):
    '''
    run the pries and secomb adaptation scheme from preop config to result

    :param preop_config: preop config dict
    :param preop_result: preop result array
    :param repair_config: config specifying repair (usually contained in the experiment config file)
    :param log_file: path to log file
    :param vis_trees: True if trees are to be visualized
    :param fig_dir: path to directory to save figures if vis_trees is true

    :return result: summarized results
    '''

    if trees_exist:

        preop_result = run_svzerodplus(config_handler.config)

        result_handler.add_unformatted_result(preop_result, 'preop')

    else:
        preop.construct_pries_trees(config_handler, 
                                    result_handler, 
                                    n_procs=n_procs,
                                    log_file=log_file,
                                    d_min=.01)
        
        # save preop config to json
        config_handler.to_file_w_trees('config_w_ps_trees.in')
    

    # compute statistics on the optimized pries and secomb parameters
    ps_param_set = np.empty((len(config_handler.trees), 8))

    for i, tree in enumerate(config_handler.trees):
        ps_param_set[i, :] = tree.pries_n_secomb.ps_params

    # get the mean and standard deviation of the optimized parameters
    ps_param_mean = np.mean(ps_param_set, axis=0)
    ps_param_std = np.std(ps_param_set, axis=0)
    # output this to the log file
    write_to_log(log_file, "Pries and Secomb parameter statistics: ")
    write_to_log(log_file, "of the form [k_p, k_m, k_c, k_s, L (cm), S_0, tau_ref, Q_ref]")
    write_to_log(log_file, "    mean: " + str(ps_param_mean))
    write_to_log(log_file, "    std: " + str(ps_param_std))

    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    operation.repair_stenosis(config_handler,
                              result_handler, 
                              repair_config[0], 
                              log_file)

    # adapt trees
    adaptation.adapt_pries_secomb(config_handler,
                                  result_handler,
                                  log_file)
    

def run_cwss_adaptation(config_handler: ConfigHandler, result_handler: ResultHandler, repair_config, log_file, n_procs=12, trees_exist=False):
    '''
    run the constant wall shear stress adaptation scheme from preop config to result

    :param preop_config: preop config dict
    :param preop_result: preop result array
    :param repair_config: config specifying repair (usually contained in the experiment config file)
    :param log_file: path to log file
    :param vis_trees: True if trees are to be visualized
    :param fig_dir: path to directory to save figures if vis_trees is true

    :return result: summarized results
    '''

    if trees_exist:

        preop_result = run_svzerodplus(config_handler.config)

        result_handler.add_unformatted_result(preop_result, 'preop')

    else:
        # construct trees
        preop.construct_cwss_trees(config_handler,
                                           result_handler,
                                           n_procs=n_procs,
                                           log_file=log_file,
                                           d_min=.0049) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

        # save preop config to as pickle, with StructuredTreeOutlet objects
        write_to_log(log_file, 'saving preop config with cwss trees...')

        config_handler.to_file_w_trees('config_w_cwss_trees.in')

    
    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    operation.repair_stenosis(config_handler, 
                                          result_handler,
                                          repair_config[0], 
                                          log_file)

    # adapt trees
    adaptation.adapt_constant_wss(config_handler,
                                  result_handler,
                                  log_file)
    

def optimize_stent_diameter(config_handler, result_handler, repair_config: dict, adapt='ps' or 'cwss', log_file=None, n_procs=12, trees_exist=True):
    '''
    optimize stent diameter based on some objective function containing flow splits or pressures
    
    :param config_handler: ConfigHandler instance
    :param result_handler: ResultHandler instance
    :param repair_config: repair config dict containing 
                        "type"="optimize stent", 
                        "location"="proximal" or some list of locations, 
                        "objective"="flow split" or "mpa pressure" or "all"
    :param adaptation: adaptation scheme to use, either 'ps' or 'cwss'
    :param n_procs: number of processors to use for tree construction
    '''

    if trees_exist:
        # compute preop result
        preop_result = run_svzerodplus(config_handler.config)
        # add the preop result to the result handler
        result_handler.add_unformatted_result(preop_result, 'preop')

    else:
        # construct trees
        if adapt == 'ps':
            # pries and secomb adaptationq
            preop.construct_pries_trees(config_handler,
                                        result_handler,
                                        n_procs=n_procs,
                                        log_file=log_file,
                                        d_min=.01)
            
        elif adapt == 'cwss':
            # constant
            preop.construct_cwss_trees(config_handler,
                                            result_handler,
                                            n_procs=n_procs,
                                            log_file=log_file,
                                            d_min=.0049) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

        # save preop config to as pickle, with StructuredTreeOutlet objects
        write_to_log(log_file, 'saving preop config with' + adapt + 'trees...')

        # pickle the config handler with the trees
        config_handler.to_file_w_trees('config_w_' + adapt + '_trees.in')

    

    def objective_function(diameters, result_handler, repair_config, adapt):
        '''
        objective function to minimize based on input stent diameters
        '''

        repair_config['type'] = 'stent'
        repair_config['value'] = diameters

        with open('config_w_' + adapt + '_trees.in', 'rb') as ff:
            config_handler = pickle.load(ff)

        # perform repair. this needs to be updated to accomodate a list of repairs > length 1
        operation.repair_stenosis(config_handler, 
                                            result_handler,
                                            repair_config, 
                                            log_file)

        # adapt trees
        adaptation.adapt_constant_wss(config_handler,
                                    result_handler,
                                    log_file)

        rpa_split = get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.rpa.branch, steady=True) / get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.mpa.branch, steady=True)

        mpa_pressure = get_branch_result(result_handler.results['adapted'], 'pressure_in', config_handler.mpa.branch, steady=True)

        if repair_config['objective'] == 'flow split':
            return (rpa_split - 0.5) ** 2
        elif repair_config['objective'] == 'mpa pressure':
            return (mpa_pressure - 20) ** 2
        elif repair_config['objective'] == 'all':
            return ((rpa_split - 0.5) * 100) ** 2 + mpa_pressure
        else:
            raise Exception('invalid objective function specified')
    
    result = minimize(objective_function, repair_config["value"], args=(result_handler, repair_config, adapt), method='Nelder-Mead')

    print('optimized stent diameters: ' + str(result.x))
    # write this to the log file as well
    write_to_log(log_file, 'optimized stent diameters: ' + str(result.x))


 