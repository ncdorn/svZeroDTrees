from svzerodtrees.interface import *
import os

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
    os.chdir('tests/cases/full_pa_test/experiments/')
    run_from_file('pa_outlet_bc_test.json', vis_trees=True)