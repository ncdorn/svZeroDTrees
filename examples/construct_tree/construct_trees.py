import os
from svzerodtrees.interface import run_from_file

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
    os.chdir('examples/construct_tree/')
    run_from_file('construct_trees_example.json')