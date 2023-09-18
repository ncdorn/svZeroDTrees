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
import pickle


def test_unsteady_optimization():
    '''
    test the preop optimization algorithm with unsteady flow
    '''

    input_file = 'tests/cases/LPA_RPA_0d/LPA_RPA_0d.json'
    log_file = 'tests/cases/LPA_RPA_0d/LPA_RPA_0d.log'
    clinical_targets = 'tests/cases/LPA_RPA_0d/clinical_targets.csv'
    working_dir = 'tests/cases/LPA_RPA_0d'

    preop_config, preop_result = preop.optimize_outlet_bcs(
        input_file,
        clinical_targets,
        log_file,
        steady=False,
        show_optimization=True,
    )

    with open('tests/cases/LPA_RPA_0d/preop_result.out', 'wb') as ff:
        pickle.dump(preop_result, ff)


if __name__ == "__main__":

    test_unsteady_optimization()