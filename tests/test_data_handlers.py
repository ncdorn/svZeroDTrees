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
    config_handler = ConfigHandler.from_file('tests/cases/full_pa_test/preop_config.in')

