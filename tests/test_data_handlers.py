import json
import sys
import os
import numpy as np
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees.utils import run_svzerodplus
import svzerodplus
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

    output = svzerodplus.simulate(config_handler.assembled_config)
    result = pd.read_csv(StringIO(output))



    with open('tests/cases/full_pa_test/assembled_result.json', 'w') as ff:
        json.dump(result, ff)


def test_config_handler():
    '''
    test config handler on a small model
    '''
    config_handler = ConfigHandler.from_json('tests/cases/LPA_RPA_0d_steady/preop_config.json')

    config_handler.assemble_config()

    with open('tests/cases/LPA_RPA_0d_steady/assembled_config.json', 'w') as ff:
        json.dump(config_handler.assembled_config, ff)

    assembled_result = run_svzerodplus(config_handler.assembled_config)

    result = run_svzerodplus(config_handler.config)

    result_comparison = DeepDiff(assembled_result, result)

    assert result_comparison == {}


if __name__ == '__main__':

    test_config_handler()



