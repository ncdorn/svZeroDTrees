from svzerodtrees.post_processing import plotting
import json

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


if __name__ =='__main__':

    test_plot_MPA()
    
