import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pysvzerod

def run_cl():
    '''
    load and run the closed loop simulation
    '''

    with open('pa_closed_loop.json') as f:
        config = json.load(f)
    
    result = pysvzerod.simulate(config)

    print(result)


if __name__ == '__main__':
    os.chdir('tests/cases/closed_loop/')

    run_cl()

