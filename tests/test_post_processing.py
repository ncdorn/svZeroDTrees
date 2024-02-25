from svzerodtrees.post_processing import plotting
from svzerodtrees.post_processing.pa_plotter import PAanalyzer
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees.post_processing.project_to_centerline import map_0d_on_centerline
from svzerodtrees.utils import *
import os
import json
import pickle

def test_plot_LPA_RPA():
    '''
    test the LPA/RPA plotting method
    '''

    # define experiment directory path
    expdir_path = 'tests/cases/LPA_RPA_0d_steady/experiments/exp_config_test_9.5.23/'

    # define figure directory path
    fig_dir = expdir_path + 'figures/'

    # get result array
    with open(expdir_path + 'summary_results.out') as ff:
        results = json.load(ff)
    
    plotting.plot_LPA_RPA_changes(fig_dir, results, 'LPA_RPA_results', 'repair')

def test_plot_MPA():
    '''
    test the MPA plotting method
    '''

    # define experiment directory path
    expdir_path = 'tests/cases/LPA_RPA_0d_steady/experiments/exp_config_test_9.5.23/'

    # define figure directory path
    fig_dir = expdir_path + 'figures/'

    # get result array
    with open(expdir_path + 'summary_results.out') as ff:
        results = json.load(ff)
    
    plotting.plot_MPA_changes(fig_dir, results, 'MPA_results', 'repair')


def test_distal_wss_plot():
    '''
    test the distal wss plotting method
    '''

    config_path = 'tests/cases/full_pa_test/optimized_pa_config.json'
    result_path = 'tests/cases/full_pa_test/preop_result.out'

    with open(config_path) as ff:
        config = json.load(ff)

    with open(result_path, 'rb') as ff:
        result = pickle.load(ff)

    pa.plot_distal_wss(config, result)


def test_cl_projection():
    '''
    test the centerline projection plotting method
    '''

    config_handler = ConfigHandler.from_file('tests/cases/LPA_RPA_0d_steady/preop_config.in')

    with open('tests/cases/LPA_RPA_0d_steady/experiments/exp_config_test_9.5.23/result_handler.out', 'rb') as ff:
        result_handler = pickle.load(ff)

    # print(result_handler.format_result_for_cl_projection('preop'))
    
    map_0d_on_centerline('tests/cases/LPA_RPA_0d_steady/centerlines.vtp', 
                        config_handler, result_handler, 
                        'adaptation', 'tests/cases/LPA_RPA_0d_steady/cl_projection/')


def test_pa_plotter_functions():
    '''
    test various functions of the pa plotter'''

    os.chdir('tests/cases/LPA_RPA_0d_steady/')
    experiment = 'exp_config_test_9.5.23'
    plotter = PAanalyzer.from_files('preop_config.json', 'experiments/' + experiment + '/full_results.json', 'experiments/' + experiment + '/figures/')

    # test R_eq calculation
    plotter.get_R_eq()

    print('R_poiseuille: ' + str(plotter.root.zero_d_element_values["R_poiseuille"]) + '  R_eq: ' + str(plotter.root.R_eq))



if __name__ =='__main__':

    test_pa_plotter_functions()
