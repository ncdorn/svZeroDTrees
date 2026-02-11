import os
from svzerodtrees import load_config, PipelineWorkflow

'''
example script for running an svZeroDTrees pipeline from a yaml config file
'''
if __name__ == '__main__':
    os.chdir('examples')
    cfg = load_config('pipeline_example.yml')
    PipelineWorkflow.from_config(cfg).run()
