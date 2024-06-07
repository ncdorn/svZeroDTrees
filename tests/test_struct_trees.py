import json
import sys
import os
import numpy as np
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
# print(sys.path)
from svzerodtrees.structuredtree import StructuredTree
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
import pickle


def build_simple_tree():
    '''
    build a simple tree from a config for testing
    '''
    
    os.chdir('tests/cases/simple_config')
    input_file = 'simple_config_1out.json'
    
    config_handler = ConfigHandler.from_json(input_file)

    result_handler = ResultHandler.from_config_handler(config_handler)

    

def build_tree_R_optimized():
    '''
    build a tree from the class method
    '''

    tree = StructuredTree(name='test_tree')

    tree.optimize_tree_diameter(resistance=100.0)

    # example: compute pressure and flow in the tree with inlet flow 10.0 cm3/s and distal pressure 100.0 dyn/cm2
    tree_result = tree.simulate(Q_in = [10.0, 10.0], Pd=100.0)

    # example: adapt the tree
    R_old, R_new = tree.adapt_constant_wss(10.0, 5.0)


    print(f'R_old = {R_old}, R_new = {R_new}')



if __name__ == '__main__':


    build_tree_R_optimized()



    