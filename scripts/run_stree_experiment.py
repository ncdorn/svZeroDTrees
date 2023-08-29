from structured_tree_simulation import *
from struct_tree_utils import *


if __name__ == '__main__':
    # name of model to analyze
    model_name = 'LPA_RPA_0d_steady'
    # experiment name, referring to config file or folder (json to be loaded as dict)
    expname = 'recursive_tree_test_7.26.23'
    '''
        Requirements of experiment params json file:
        "name": name of experiment
        "repair type": extensive, proximal or a list of vessels to repair
        "repair degrees": list of ints with degrees of stenosis repair
        '''
    # starting from svzerodplus directory
    run_simulation(model_name, expname, optimized=True, vis_trees=True)
