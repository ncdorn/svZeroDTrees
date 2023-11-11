import os
import json
import pickle
from svzerodtrees import preop, operation, adaptation, postop
from svzerodtrees.post_processing import plotting
from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees.post_processing.pa_plotter import PAanalyzer

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

        os.system('mkdir ' + expname)

    elif os.path.exists('experiments'): # we are ond directory above the experiments directory

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
            config_handler, result_handler = preop.optimize_pa_bcs(
                input_file,
                mesh_surfaces_path,
                clinical_targets,
                log_file,
                show_optimization=False
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
        
        # json won't work for results dump
        with open('preop_result_handler.out', 'wb') as ff:
            pickle.dump(result_handler, ff)

    else: # use previous optimization results

        config_handler = ConfigHandler.from_json('preop_config.json')

        # json won't work for results load
        with open('preop_result_handler.out', 'rb') as ff:
            result_handler = pickle.load(ff)

        # get preop result
        preop_flow = run_svzerodplus(config_handler.config)

        # load a new result handler
        result_handler = ResultHandler.from_config(config_handler.config)

        # add result to the handler
        result_handler.add_unformatted_result(preop_flow, 'preop')

    if adapt == 'ps': # use pries and secomb adaptation scheme
        
        run_pries_secomb_adaptation(config_handler, 
                                    result_handler, 
                                    repair_config, 
                                    log_file, vis_trees, 
                                    fig_dir,
                                    trees_exist)

    elif adapt == 'cwss': # use constant wall shear stress adaptation scheme
        
        run_cwss_adaptation(config_handler, 
                            result_handler, 
                            repair_config, 
                            log_file, 
                            vis_trees, 
                            fig_dir,
                            trees_exist)

    else:
        raise Exception('invalid adaptation scheme chosen')
    
    # format the results
    result_handler.format_results()

    # save the adapted config
    config_handler.to_json_w_trees(expdir_path + 'adapted_config.json')
    
    # save the result
    result_handler.to_json(expdir_path + 'full_results.json')
    
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


def run_pries_secomb_adaptation(config_handler: ConfigHandler, result_handler, repair_config, log_file, vis_trees, fig_dir, trees_exist=False):
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
        config_handler.from_file_w_trees('config_w_pries_trees.in')
    else:
        preop.construct_pries_trees(config_handler, 
                                    result_handler, 
                                    log_file,
                                    fig_dir=fig_dir, 
                                    d_min=.0049)
        
        # save preop config to json
        config_handler.to_file_w_trees('config_w_pries_trees.in')

    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    operation.repair_stenosis_coefficient(config_handler,
                                          result_handler, 
                                          repair_config[0], 
                                          log_file)

    # adapt trees
    adaptation.adapt_pries_secomb(config_handler,
                                  result_handler,
                                  log_file)

def run_cwss_adaptation(config_handler: ConfigHandler, result_handler: ResultHandler, repair_config, log_file, vis_trees, fig_dir, trees_exist=False):
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
        config_handler.from_file_w_trees('config_w_cwss_trees.in')

        preop_result = run_svzerodplus(config_handler.config)

        result_handler.add_unformatted_result(preop_result, 'preop')

    else:
        # construct trees
        preop.construct_cwss_trees(config_handler,
                                           result_handler,
                                           log_file,
                                           fig_dir=fig_dir,
                                           d_min=.01) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

        # save preop config to as pickle, with StructuredTreeOutlet objects
        write_to_log(log_file, 'saving preop config with cwss trees...')

        config_handler.to_file_w_trees('config_w_cwss_trees.in')
    
    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    operation.repair_stenosis_coefficient(config_handler, 
                                          result_handler,
                                          repair_config[0], 
                                          log_file)

    # adapt trees
    adaptation.adapt_constant_wss(config_handler,
                                  result_handler,
                                  log_file)



