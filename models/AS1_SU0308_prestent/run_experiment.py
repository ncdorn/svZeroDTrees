from svzerodtrees.interface import *
import os

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
    os.chdir('experiments')
    run_from_file('AS1_pries_adaptation.json', vis_trees=True)
