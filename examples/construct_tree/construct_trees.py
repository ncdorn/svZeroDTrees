import os
from svzerodtrees import load_config, ConstructTreesWorkflow

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
    os.chdir('examples/construct_tree/')
    cfg = load_config('construct_trees_example.yml')
    ConstructTreesWorkflow.from_config(cfg).run()
