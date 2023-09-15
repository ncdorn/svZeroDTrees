from svzerodtrees.post_processing import plotting
from svzerodtrees.post_processing import pa_tree_analysis as pa
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

if __name__ =='__main__':

    test_distal_wss_plot()
    
