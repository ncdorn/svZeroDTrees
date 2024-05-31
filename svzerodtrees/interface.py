import os
import sys
import json
import time
import pickle
from svzerodtrees import preop, operation, adaptation
from svzerodtrees.optimization import StentOptimization
from svzerodtrees.post_processing import plotting, project_to_centerline
from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees.post_processing.pa_plotter import PAanalyzer
from scipy.optimize import minimize

def run_from_file(exp_config_file: str, vis_trees=True):
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
    task = exp_config["task"]
    task_params = exp_config[task]

    if task == 'repair':
        # compute 0D repair
        repair_config = exp_config['repair']
    elif task == 'threed_adaptation':
        # compute 3D adaptation
        # run_threed_adaptation(task_params['preop_dir'], task_params['postop_dir'], task_params['adapted_dir'])
        if optimized:
            run_threed_from_msh(
                task_params['preop_dir'],
                task_params['postop_dir'],
                task_params['adapted_dir'],
                task_params['zerod_config'],
                task_params['svpre_path'],
                task_params['svsolver_path'],
                task_params['svpost_path']
            )
        else:
            # optimize the zerod model and then run the 3d simulations
            print('optimizing zerod model to find BCs')
            if is_full_pa:
                config_handler, result_handler, pa_config = preop.optimize_pa_bcs(
                    task_params['zerod_config'],
                    os.path.join(task_params['preop_dir'], 'mesh-complete', 'mesh-surfaces'),
                    os.path.join(os.path.dirname(task_params['zerod_config']), 'clinical_targets.csv')
                )

            else:
                config_handler, result_handler = preop.optimize_outlet_bcs(
                    task_params['zerod_config'],
                    os.path.join(os.path.dirname(task_params['zerod_config']), 'clinical_targets.csv')
                )

            # save optimized config and result
            preop_config_path = os.path.join(os.path.dirname(task_params['zerod_config']), 'preop_config.json')
            config_handler.to_json(preop_config_path)

            run_threed_from_msh(
                task_params['preop_dir'],
                task_params['postop_dir'],
                task_params['adapted_dir'],
                preop_config_path,
                task_params['svpre_path'],
                task_params['svsolver_path'],
                task_params['svpost_path']
            )

        sys.exit() # exit the program


    elif task == 'construct_trees':
        log_file = expname + '.log'
        config_handler = ConfigHandler.from_json(modelname + '.json', is_pulmonary=is_full_pa)
        result_handler = ResultHandler.from_config_handler(config_handler)
        if task_params['tree_type'] == 'cwss':
            # construct trees
            preop.construct_cwss_trees(config_handler,
                                        result_handler,
                                        n_procs=24,
                                        log_file=log_file,
                                        d_min=.0049) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

            # save preop config to as pickle, with StructuredTreeOutlet objects
            write_to_log(log_file, 'saving preop config with cwss trees...')

            config_handler.to_file_w_trees('config_w_cwss_trees.in')
        elif task_params['tree_type'] == 'ps':
            # construct trees
            preop.construct_pries_trees(config_handler,
                                        result_handler,
                                        n_procs=24,
                                        log_file=log_file,
                                        d_min=.01)

            write_to_log(log_file, 'saving preop config with ps trees...')

            config_handler.to_file_w_trees('config_w_ps_trees.in')
        else:
            raise Exception('invalid tree type specified')
        
        sys.exit() # exit the program
    else:
        raise Exception('invalid task specified')


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
                log_file
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
        # optimize_stent_diameter(config_handler,
        #                         result_handler,
        #                         repair_config[0],
        #                         adapt,
        #                         log_file,
        #                         n_procs=12,
        #                         trees_exist=trees_exist)

        stent_optimization = StentOptimization(config_handler,
                                               result_handler,
                                               repair_config[0],
                                               adapt,
                                               log_file,
                                               n_procs=12,
                                               trees_exist=trees_exist)
        
        stent_optimization.minimize_nm()


    elif repair_config[0]['type'] == 'estimate_bcs':
        print("estimating bcs...")
        cwd = os.getcwd()
        config_handler.generate_threed_coupler(cwd)
        print('estimated bcs written to ' + cwd + '/svzerod_3Dcoupling.json')
        
    else:
        if adapt == 'ps': # use pries and secomb adaptation scheme
            
            run_pries_secomb_adaptation(config_handler, 
                                        result_handler, 
                                        repair_config[0], 
                                        log_file,
                                        n_procs=12,
                                        trees_exist=trees_exist)

        elif adapt == 'cwss': # use constant wall shear stress adaptation scheme
            
            run_cwss_adaptation(config_handler, 
                                result_handler, 
                                repair_config[0], 
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

        if not os.path.exists(expdir_path + 'cl_projection'):
            os.system('mkdir ' + expdir_path + 'cl_projection')
        
        for period in ['preop', 'postop', 'adapted', 'adaptation']:
            if repair_config[0]["location"] == 'proximal':
                repair_location = [result_handler.lpa_branch, result_handler.rpa_branch]
            else:
                repair_location = repair_config[0]["location"]
            project_to_centerline.map_0d_on_centerline('centerlines.vtp',
                                                        config_handler,
                                                        result_handler,
                                                        period,
                                                        expdir_path + 'cl_projection',
                                                        repair_location)
        

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
                                repair_config, 
                                log_file)

    # adapt trees
    adaptation.adapt_constant_wss(config_handler,
                                  result_handler,
                                  log_file)
    

def run_threed_adaptation(preop_simulation_dir, postop_simulation_dir, adapted_simulation_dir):
    '''
    compute the microvasular adaptation for a 3d coupled soluiton and output an adapted config handler

    required in each simulation directory:
    - solver.inp
    - .svpre file
    - svzerod_3Dcoupling.json
    - svZeroD_interface.dat
    '''

    # check that the directories are unique so we dont end up screwing stuff up by accident
    if preop_simulation_dir == postop_simulation_dir:
        raise Exception('preop and postop simulation directories are the same')
    if preop_simulation_dir == adapted_simulation_dir:
        raise Exception('preop and adapted simulation directories are the same')
    if postop_simulation_dir == adapted_simulation_dir:
        raise Exception('postop and adapted simulation directories are the same')

    preop_simname = os.path.basename(preop_simulation_dir)
    # load the preop config handler
    preop_config_handler = ConfigHandler.from_json(preop_simulation_dir + '/svzerod_3Dcoupling.json', is_pulmonary=False, is_threed_interface=True)

    preop.construct_coupled_cwss_trees(preop_config_handler, preop_simulation_dir, n_procs=12)

    # need to get the period and timestep size of the simulation to accurately compute the mean flow
    n_steps = get_nsteps(preop_simulation_dir + '/solver.inp', preop_simulation_dir + '/' + preop_simname + '.svpre')

    # load in the preop and postop outlet flowrates from the 3d simulation.
    # the Q_svZeroD file needs to be in the top level of the simulation directory
    preop_q = pd.read_csv(preop_simulation_dir + '/Q_svZeroD', sep='\s+')
    preop_mean_q = preop_q.iloc[-n_steps:].mean(axis=0).values

    postop_q = pd.read_csv(postop_simulation_dir + '/Q_svZeroD', sep='\s+')
    postop_mean_q = postop_q.iloc[-n_steps:].mean(axis=0).values

    # adapt the bcs
    adaptation.adapt_constant_wss_threed(preop_config_handler, preop_mean_q, postop_mean_q)

    # save the adapted config

    print('adapted config being saved to ' + adapted_simulation_dir + '/svzerod_3Dcoupling.json')

    preop_config_handler.to_json(adapted_simulation_dir + '/svzerod_3Dcoupling.json')

    prepare_adapted_simdir(postop_simulation_dir, adapted_simulation_dir)


def run_threed_from_msh(preop_simulation_dir, 
                        postop_simulation_dir, 
                        adapted_simulation_dir, 
                        zerod_config,
                        svpre_path=None,
                        svsolver_path=None,
                        svpost_path=None):
    '''
    run threed adaptation from preop and postop mesh files only
    '''

    # check that the directories are unique so we dont end up screwing stuff up by accident
    if preop_simulation_dir == postop_simulation_dir:
        raise Exception('preop and postop simulation directories are the same')
    if preop_simulation_dir == adapted_simulation_dir:
        raise Exception('preop and adapted simulation directories are the same')
    if postop_simulation_dir == adapted_simulation_dir:
        raise Exception('postop and adapted simulation directories are the same')

    # save which directory we are in
    wd = os.getcwd()
    # setup preop dir
    num_timesteps = setup_simdir_from_mesh(preop_simulation_dir, zerod_config)
    # run preop simulation
    os.chdir(preop_simulation_dir)
    print('submitting preop simulation job...')
    os.system('sbatch run_solver.sh')
    os.chdir(wd)

    # setup postop dir, num timesteps assumed to be same
    setup_simdir_from_mesh(postop_simulation_dir, zerod_config)
    # run postop simulation
    os.chdir(postop_simulation_dir)
    print('submitting postop simulation job...')
    os.system('sbatch run_solver.sh')
    os.chdir(wd)

    # check if the simulations have run
    preop_complete = False
    postop_complete = False

    time.sleep(300)
    while not preop_complete and not postop_complete:
        time.sleep(150)
        timestep = 0
        with open(os.path.join(preop_simulation_dir, '*-procs_case/histor.dat'), 'r') as histor_dat:
            lines = histor_dat.readlines()
            if num_timesteps == int(lines[-1].split()[0]):
                # wait to finish up last timestep
                time.sleep(120)
                preop_complete = True
                print('preop simulation complete!')
        
        with open(os.path.join(postop_simulation_dir, '*-procs_case/histor.dat'), 'r') as histor_dat:
            lines = histor_dat.readlines()
            if num_timesteps == int(lines[-1].split()[0]):
                # wait to finish up last timestep
                time.sleep(120)
                postop_complete = True
                print('postop simulation complete!')

    # load in the preop and postop outlet flowrates from the 3d simulation.
    # the Q_svZeroD file needs to be in the top level of the simulation directory
    preop_q = pd.read_csv(preop_simulation_dir + '/Q_svZeroD', sep='\s+')
    preop_mean_q = preop_q.iloc[-num_timesteps / 2:].mean(axis=0).values

    postop_q = pd.read_csv(postop_simulation_dir + '/Q_svZeroD', sep='\s+')
    postop_mean_q = postop_q.iloc[-num_timesteps / 2:].mean(axis=0).values

    # initialize preop config handler
    preop_config_handler = ConfigHandler.from_json(zerod_config)

    print('computing boundary condition adaptation...')

    # adapt the bcs
    adaptation.adapt_constant_wss_threed(preop_config_handler, preop_mean_q, postop_mean_q)

    # save the adapted config

    print('adapted config being saved to ' + adapted_simulation_dir + '/svzerod_3Dcoupling.json')

    preop_config_handler.to_json(adapted_simulation_dir + '/svzerod_3Dcoupling.json')

    prepare_adapted_simdir(postop_simulation_dir, adapted_simulation_dir)

    # run simulation
    os.chdir(adapted_simulation_dir)
    print('submitting adapted simulation job...')
    os.system('sbatch run_solver.sh')
    os.chdir(wd)

    print('all simulations complete!')
    

