import json
import sys
import os
import numpy as np
from svzerodtrees._config_handler import ConfigHandler
from svzerodtrees._result_handler import ResultHandler

def test_pa_handling():
    '''
    test the routines to handle a pulmonary artery 0D config file
    '''

    # load the config file
    config_handler = ConfigHandler.from_json('tests/cases/full_pa_test/preop_config.in')

    config_handler.load_pa_model()

    print(config_handler.lpa.R_eq, config_handler.rpa.R_eq)


if __name__ == '__main__':
    test_pa_handling()



