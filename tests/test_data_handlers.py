import json
import sys
import os
import numpy as np
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees.operation import Stenosis, repair_stenosis
from svzerodtrees.utils import run_svzerodplus
from svzerodtrees.preop import PAConfig
import pysvzerod
import pandas as pd
from io import StringIO
from deepdiff import DeepDiff


def test_pa_handling():
    '''
    test the routines to handle a pulmonary artery 0D config file
    '''

    # load the config file
    config_handler = ConfigHandler.from_json('tests/cases/full_pa_test/preop_config.json')

    config_handler.assemble_config()

    output = pysvzerod.simulate(config_handler.assembled_config)
    result = pd.read_csv(StringIO(output))



    with open('tests/cases/full_pa_test/assembled_result.json', 'w') as ff:
        json.dump(result, ff)


def test_config_handler():
    '''
    test config handler on a small model
    '''
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d/LPA_RPA_0d.json')

    result = run_svzerodplus(config_handler.config)

    config_handler.assemble_config()

    assembled_result = run_svzerodplus(config_handler.config)

    result_comparison = DeepDiff(assembled_result, result)

    print('ran simulations')

    print(config_handler.config)
    
    assert result_comparison == {}


def test_coupling_block_generation():
    '''
    test the generation of coupling blocks from a config handler
    '''
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d/LPA_RPA_0d.json')

    ConfigHandler.generate_threed_coupler(config_handler, 'tests/cases/LPA_RPA_0d')


def test_config_handler_methods():
    '''
    test various methods within the config handler
    '''
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    old_R = []
    for vessel in config_handler.get_segments(1):
        old_R.append(vessel.R)
    
    config_handler.change_branch_resistance(1, 100)

    new_R = []
    for vessel in config_handler.get_segments(1):
        new_R.append(vessel.R)
    
    print(old_R, new_R)
    



def test_stenosis_ops():
    '''
    test stenosis operations
    '''
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    result_handler = ResultHandler.from_config(config_handler.config)

    config_handler.simulate(result_handler, 'preop')

    repair_config = {'location': 'proximal', 'type': 'stent', 'value': [0.9, 0.9]}

    repair_stenosis(config_handler, result_handler, repair_config)

    print(result_handler.results['preop'], result_handler.results['postop'])


def test_pa_config():
    '''
    test the creation of a pulmonary artery config file
    '''

    config_handler = ConfigHandler.from_json('tests/cases/full_pa_test/preop_config.json')

    pa_config = PAConfig.from_config_handler(config_handler)

    pa_config.assemble_config()

    with open('tests/cases/full_pa_test/pa_config.json', 'w') as ff:
        json.dump(pa_config.config, ff)

    simulation_run = False
    
    result = pa_config.simulate()

    simulation_run = True

    assert simulation_run == True




if __name__ == '__main__':

    test_coupling_block_generation()



