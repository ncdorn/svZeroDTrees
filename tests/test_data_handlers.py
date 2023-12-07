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



if __name__ == '__main__':

    test_pa_handling()



