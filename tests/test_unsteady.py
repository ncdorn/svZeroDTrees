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
from svzerodtrees.preop import ClinicalTargets, PAConfig
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

    input_file = 'tests/cases/rh_chamber/rh_chamber_mmhg.json'

    with open(input_file, 'r') as f:
        config = json.load(f)


    result = pysvzerod.simulate(config)

    plot_result(result, 'pressure_in', 'vessel', 'tests/cases/rh_chamber/pressure_in_mmhg.png')
    plot_result(result, 'flow_in', 'vessel', 'tests/cases/rh_chamber/flow_in_mmhg.png')


def test_unsteady_pa():
    '''
    test the simulation with unsteady pa flow
    '''

    input_file = 'tests/cases/pa_unsteady/AS2_unsteady.json'

    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config_handler(config_handler)


    # config_handler.to_json('tests/cases/pa_unsteady/AS2_rh_chamber_rcr_new.json')

    config_handler.simulate(result_handler, 'preop')

    config_handler.to_json('tests/cases/pa_unsteady/AS2_unsteady.json')

    print('simulation complete!')

    print(f"cardiac output: {result_handler.get_cardiac_output(0)}")
    print(f"mpa min pressure: {result_handler.results['preop']['pressure_in'][0].min()}")

    # result_handler.plot('preop', 'pressure_in', 0, 'tests/cases/pa_unsteady/0.2scaled_pressure_in.png', show_mean=True)
    result_handler.plot('preop', 'flow_in', 0, 'tests/cases/pa_unsteady/scaled_flow_in.png', show_mean=True)


def test_unsteady_pa_optimization():
    '''
    test the optimization algorithm with unsteady pa flow
    '''

    input_file = 'tests/cases/pa_unsteady/AS2_unsteady.json'
    clinical_targets = 'tests/cases/pa_unsteady/clinical_targets.csv'
    log_file = 'tests/cases/pa_unsteady/AS2_unsteady.log'
    msh_surfaces = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/Simvascular/threed_models/AS2_prestent/Meshes/1.6M_elements/mesh-surfaces'
    working_dir = 'tests/cases/pa_unsteady'

    config_handler, result_handler, pa_config = preop.optimize_pa_bcs(
        input_file,
        msh_surfaces,
        clinical_targets,
        log_file,
        steady=False
    )

    config_handler.simulate(result_handler, 'preop')

    config_handler.to_json('tests/cases/pa_unsteady/AS2_unsteady_preop.json')

    result_handler.plot('preop', 'pressure_in', 0, 'tests/cases/pa_unsteady/dia+C_opt_pressure_in.png', show_mean=True)
    result_handler.plot('preop', 'flow_in', 0, 'tests/cases/pa_unsteady/dia+C_opt_flow_in.png', show_mean=True)

    print('unsteady test completed')


def test_unsteady_pa_config():
    input_file = 'tests/cases/pa_unsteady/AS2_unsteady.json'
    clinical_targets = 'tests/cases/pa_unsteady/clinical_targets.csv'

     # get the clinical target values
    clinical_targets = ClinicalTargets.from_csv(clinical_targets, steady=False)


    # initialize the data handlers
    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config(config_handler.config)


    pa_config = PAConfig.from_config_handler(config_handler, clinical_targets)

    pa_config.optimize(steady=False, nonlin=False)
    result = pa_config.simulate()
    result["time"] = config_handler.get_time_series()

    pa_config.to_json('tests/cases/pa_unsteady/AS2_pa_config_opt.json')

    result_handler.results['pa_config'] = result

    result_handler.plot('pa_config', 'pressure_in', [0, 2, 4], filepath='tests/cases/pa_unsteady/AS2_pa_config_opt_pressure')
    result_handler.plot('pa_config', 'flow_in', [0, 2, 4], filepath='tests/cases/pa_unsteady/AS2_pa_config_opt_flow')


def test_unsteady_simple():

    input_file = 'tests/cases/simple_config/simple_config_1rcr.json'
    
    config_handler = ConfigHandler.from_json(input_file, is_pulmonary=False)

    result_handler = ResultHandler.from_config_handler(config_handler)

    config_handler.simulate(result_handler, 'preop')

    # do a simple optimization with systolic, diastolic and mean pressure targets
    def objective(RC_guess, config_handler, result_handler):
        '''
        simple loss function for optimizing the simple config
        '''
        config_handler.bcs['RCR_0'].R = RC_guess[0]
        config_handler.bcs['RCR_0'].C = RC_guess[1]

        config_handler.simulate(result_handler, 'preop')

        # get the inlet pressures
        pressure_in = result_handler.results['preop']['pressure_in'][0]

        P_sys = pressure_in.max() / 1333.2
        P_dia = pressure_in.min() / 1333.2
        P_mean = np.mean(pressure_in) / 1333.2

        targets = np.array([120, 80, 100])

        loss = np.sum(np.subtract(targets, np.array([P_sys, P_dia, P_mean])) ** 2)

        print(f"loss: {loss} P_sys: {P_sys} P_dia: {P_dia} P_mean: {P_mean}, R: {RC_guess[0]}, C: {RC_guess[1]}")

        return loss
    
    # do the optimization
    # res = minimize(objective, [1000, 0.0001], args=(config_handler, result_handler))

    # config_handler.to_json('tests/cases/simple_config/simple_config_1rcr_opt.json')
        

    result_handler.plot('preop', 'pressure_in', [0], 'tests/cases/simple_config/pressure_in.png', show_mean=True)
    result_handler.plot('preop', 'flow_in', [0], 'tests/cases/simple_config/flow_in.png', show_mean=True)

    



def rh_chamber_param_sweep():
    pass


if __name__ == "__main__":

    test_unsteady_simple()