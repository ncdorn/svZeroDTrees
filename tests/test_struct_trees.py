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
from svzerodtrees import operation, preop, interface, postop
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
import pickle


def build_simple_tree():
    '''
    build a simple tree for testing
    '''
    
    os.chdir('tests/cases/simple_config')
    input_file = 'simple_config_2out.json'
    
    config_handler = ConfigHandler.from_json(input_file)

    result_handler = ResultHandler.from_config_handler(config_handler)

    config_handler.simulate(result_handler, 'preop')
    config_handler = ConfigHandler.from_config_file