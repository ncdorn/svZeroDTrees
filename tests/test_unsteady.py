import json
import sys
import os
import numpy as np
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
# print(sys.path)
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees import operation, preop, interface
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
import pickle


def test_unsteady_optimization():
    '''
    test the preop optimization algorithm with unsteady flow
    '''

    input_file = 'tests/cases/simple_config/simple_config_rcr.json'
    log_file = 'tests/cases/simple_config/simple_config_rcr.log'
    # input_file = 'tests/cases/LPA_RPA_0d/LPA_RPA_0d.json'
    # log_file = 'tests/cases/LPA_RPA_0d/LPA_RPA_0d.log'
    clinical_targets = 'tests/cases/LPA_RPA_0d/clinical_targets.csv'
    working_dir = 'tests/cases/simple_config'

    write_to_log(log_file, 'unsteady test started', write=True)

    config_handler, result_handler = preop.optimize_outlet_bcs(
        input_file,
        clinical_targets,
        log_file,
        steady=False,
        show_optimization=False
    )

    config_handler.plot_inflow()

    print('unsteady test completed')


def test_rh_chamber():
    '''
    test the rh_chamber model'''

    input_file = 'tests/cases/rh_chamber/rh_chamber.json'

    with open(input_file, 'r') as f:
        config = json.load(f)


    result = pysvzerod.simulate(config)

    plot_result(result, 'pressure_in', 'tests/cases/rh_chamber/pressure_in.png')
    plot_result(result, 'flow_in', 'tests/cases/rh_chamber/flow_in.png')


def rh_chamber_param_sweep():
    pass


if __name__ == "__main__":

    test_rh_chamber()