import os
import json
import pickle
from svzerodtrees import preop, operation, adaptation, postop
from svzerodtrees.utils import *

def run_from_file(exp_config_file: str, optimized: bool=False, vis_trees: bool=False):
    '''
    run the structured tree optimization pipeline from an experiment config file

    :param exp_config_file: path to the experiment config file
    :param optimized: True if the outlet bcs have previously been optimized. Use the optimized config (preop_config.in)
    :param vis_trees: if true, make tree visualization figures
    '''
    
    with open(exp_config_file) as ff:
        exp_config = json.load(ff)

    # unpack the experiment config parameters
    expname = exp_config["name"]
    modelname = exp_config["model"]
    adapt = exp_config["adapt"] # either ps (pries and secomb) or cwss (constant wall shear stress)
    optimized = exp_config["optimized"] # true if the experiment has been optimized before, to skip the preop optimization
    repair_config = exp_config["repair"]

    # check if we are in an experiments directory, if not assume we are in it
    if os.path.exists('../experiments'): # we are in the experiments directory already

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
    input_file = modelname + '.in'
    clinical_targets = 'clinical_targets.csv'
    log_file = expdir_path + expname + '.log'

    # initialize log file
    write_to_log(log_file, 'beginning experiment ' + expname,  write=True)

    # optimize preoperative outlet boundary conditions
    if not optimized:

        preop_config, preop_result = preop.optimize_outlet_bcs(
            input_file,
            clinical_targets,
            log_file,
            show_optimization=False
        )

        # save optimized config and result
        with open('preop_config.in', 'w') as ff:
            json.dump(preop_config, ff)
        
        # json won't work for results dump
        with open('preop_result.out', 'wb') as ff:
            pickle.dump(preop_result, ff)

    else: # use previous optimization results

        with open('preop_config.in') as ff:
            preop_config = json.load(ff)

        # json won't work for results load
        with open('preop_result.out', 'rb') as ff:
            preop_result = pickle.load(ff)

    if adapt == 'ps': # use pries and secomb adaptation scheme
        
        result = run_pries_secomb_adaptation(preop_config, preop_result, repair_config, log_file, vis_trees, fig_dir)

    elif adapt == 'cwss': # use constant wall shear stress adaptation scheme
        
        result = run_cwss_adaptation(preop_config, preop_result, repair_config, log_file, vis_trees, fig_dir)

    else:
        raise Exception('invalid adaptation scheme chosen')
    
    with open(expdir_path + 'summary_results.out', 'w') as ff:
        json.dump(result, ff)
        
    
def run_pries_secomb_adaptation(preop_config, preop_result, repair_config, log_file, vis_trees, fig_dir):
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

    # construct trees
    trees = preop.construct_pries_trees(preop_config, 
                                        preop_result, 
                                        log_file, 
                                        vis_trees, 
                                        fig_dir)

    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    postop_config, postop_result = operation.repair_stenosis_coefficient(preop_config, 
                                                                            repair_config[0], 
                                                                            log_file)

    # adapt trees
    adapted_config, adapted_result, trees = adaptation.adapt_pries_secomb(postop_config, 
                                                                            trees, 
                                                                            preop_result, 
                                                                            postop_result, 
                                                                            log_file)
    
    # summarize results
    results = postop.summarize_results(adapted_config, preop_result, postop_result, adapted_result)

    return results


def run_cwss_adaptation(preop_config, preop_result, repair_config, log_file, vis_trees, fig_dir):
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

    # construct trees
    trees = preop.construct_cwss_trees(preop_config,
                                       preop_result,
                                       log_file,
                                       vis_trees,
                                       fig_dir)
    
    # perform repair. this needs to be updated to accomodate a list of repairs > length 1
    postop_config, postop_result = operation.repair_stenosis_coefficient(preop_config, 
                                                                            repair_config[0], 
                                                                            log_file)

    # adapt trees
    adapted_config, adapted_result, trees = adaptation.adapt_constant_wss(postop_config, 
                                                                            trees, 
                                                                            preop_result, 
                                                                            postop_result, 
                                                                            log_file)
    
    # summarize results
    results = postop.summarize_results(adapted_config, preop_result, postop_result, adapted_result)

    return results



