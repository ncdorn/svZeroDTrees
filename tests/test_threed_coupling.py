import json
import sys
import os
import numpy as np
import time
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
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
from svzerodtrees.threedutils import *
import pickle


def test_config_handler():
    '''
    test the config handler with a 3d-0d coupling file
    '''
    # load the config file
    threed_coupling_config = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/svzerod_3Dcoupling.json'

    config_handler = ConfigHandler.from_json(threed_coupling_config, is_pulmonary=False, is_threed_interface=True)

    print(config_handler.config)

def test_coupled_tree_construction():
    '''
    test the construction of a coupled tree
    '''
    # load the config file
    threed_coupling_config = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/svzerod_3Dcoupling.json'
    simulation_dir = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/'

    config_handler = ConfigHandler.from_json(threed_coupling_config, is_pulmonary=False, is_threed_interface=True)

    preop.construct_coupled_cwss_trees(config_handler, simulation_dir)


def test_data_handling():
    '''
    test random data handlers'''

    Q_svZeroD = 'tests/cases/test_cylinder/Simulations/steady/Q_svZeroD'

    df = get_outlet_flow(Q_svZeroD)

    print(df)


def test_interface():
    '''
    test the interface
    '''
    preop_dir = 'tests/cases/test_cylinder/Simulations/steady'
    interface.run_threed_adaptation(preop_dir, ' ')


if __name__ == '__main__':
    test_interface()
