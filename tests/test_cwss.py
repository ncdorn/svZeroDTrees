import json
import sys
import os
import numpy as np
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
from svzerodtrees import operation, preop, interface, adaptation
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler

def test_cwss_adaptation():
    '''
    test the cwss adaptation algorithm against an analytical solution
    '''
    
    config_handler = ConfigHandler.from_json('tests/cases/simple_config/simple_config_2out.json')

    result_handler = ResultHandler.from_config_handler(config_handler)

    with open('tests/cases/simple_config/repair.json') as ff:
        repair_dict = json.load(ff)
    
    repair_config = repair_dict['adjust_resistance']

    preop.construct_cwss_trees(config_handler, result_handler, n_procs=12, d_min=0.5)

    operation.repair_stenosis(config_handler, result_handler, repair_config)

    adaptation.adapt_constant_wss(config_handler, result_handler)

    result_handler.format_results()

    print(result_handler.clean_results)

def verify_single_tree():
    '''
    compare a single tree to an analytical solution
    '''

    config_handler = ConfigHandler.from_json('tests/cases/simple_config/simple_config_1out.json', is_pulmonary=False)

    result_handler = ResultHandler.from_config_handler(config_handler)

    preop.construct_cwss_trees(config_handler, result_handler, n_procs=12, d_min=0.8)

    print([tree.root.d / 2 for tree in config_handler.trees])
    print('tree resistance: ', config_handler.trees[0].root.R_eq)

    config_handler.bcs["INFLOW"].values["Q"] = [200.0, 200.0]

    config_handler.simulate(result_handler, 'postop')

    adaptation.adapt_constant_wss(config_handler, result_handler)

    result_handler.format_results(is_pulmonary=False)

    print('R_adapted: ', config_handler.trees[0].root.R_eq)
    print('adapted radius: ', config_handler.trees[0].root.d / 2)

    # print(result_handler.clean_results)



if __name__ == '__main__':

    verify_single_tree()