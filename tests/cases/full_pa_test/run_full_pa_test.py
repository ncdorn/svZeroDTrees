from svzerodtrees.interface import *
import os

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
	
    if os.path.isdir('tests'):
        # assume we are running from the vs code 
        # debug profile and not in the model directory and need to cd into it
        os.chdir('tests/cases/full_pa_test/experiments/')
    else: # assume we are running directly from the model dir    
        os.chdir('experiments')

    run_from_file('pa_outlet_bc_test.json', vis_trees=True)
