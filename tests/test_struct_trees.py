import json
import sys
import os
import numpy as np
import pandas as pd
import scipy.signal
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
from svzerodtrees._config_handler import ConfigHandler, SimParams
from svzerodtrees._result_handler import ResultHandler
import pickle
import scipy
import pysvzerod


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


def test_fft():
    '''
    test the olufsen imedance calculation
    '''
    # test fft
    with open('tests/cases/pa_unsteady/inflow.flow') as ff:
        inflow = pd.read_csv(ff, delimiter=' ', header=None, names=['t', 'q'])
    
    inflow['q'] = inflow['q'] * -1
    
    Y = np.fft.fft(inflow['q'])

    Y_half = copy.deepcopy(Y)

    np.put(Y_half, range(101, 201), 0.0)

    print(Y_half, Y)

    y_half = np.fft.ifft(Y_half)
    y = np.fft.ifft(Y)


    plt.plot(inflow.t, inflow.q, label='original signal')
    plt.plot(inflow.t, y_half, label='first n/2 fft components')
    plt.plot(inflow.t, y, '--', label='full fft components')
    plt.legend()
    plt.show()


def test_impedance():
    '''
    test the impedance calculations in the frequence domain
    
    it is interesting that the flow and pressure does not actually depend on the outlet flow or pressure.
    we just sample some frequencies in the time period of the inflow and calculate the impedance at each frequency'''

    # take the inflow and get the period
    with open('tests/cases/pa_unsteady/inflow.flow') as ff:
        inflow_raw = pd.read_csv(ff, delimiter=' ', header=None, names=['t', 'q'])
    
    inflow_raw['q'] = inflow_raw['q'] * -1

    # Q = 1000.0

    # inflow_raw = pd.DataFrame({'q': [Q, Q], 't': [0.0, 1.0]})

    # make some simulation parameters
    simparams = SimParams({
        'number_of_time_pts_per_cardiac_cycle': 100
    })

    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', Q_in=inflow_raw['q'], time=inflow_raw['t'], simparams=simparams)

    # build the tree
    tree.build_tree(initial_d=0.5, d_min=.01)

    # compute the impedance in frequency domain
    Z_om = tree.compute_olufsen_impedance()

    # convert flow to frequency domain
    Q_t = scipy.signal.resample(inflow_raw['q'], len(Z_om))
    t = np.linspace(0, inflow_raw['t'].iloc[-1], len(Q_t))
    Q_om = np.fft.fft(Q_t)

    # Z_om[0] = Z_om[0] * 50000
    # convert pressure to frequency domain
    P_om = Z_om * Q_om

    # P_om[0] = 5000

    P_t = np.fft.ifft(P_om) / 1333.2 # convert to dyn/cm2

    # Z_t = np.fft.ifft(Z_om)

    fig, axs = plt.subplots(2)
    axs[0].plot(t, P_t)
    axs[0].set_ylabel('Pressure (mmHg)')
    # axs[0].set_ylim([P_t[0] - 5, P_t[0] + 5])
    fig.suptitle(f'structured tree with {tree.count_vessels()} vessels, d_min = {tree.d_min} cm')

    axs[1].plot(t, Q_t)
    axs[1].set_ylabel('Flow (cm3/s)')
    axs[1].set_title('Inflow')

    plt.savefig('tests/cases/olufsen_impedance/test_tree_qp.png')


    # print(P_t)



