from svzerodtrees.interface import *

if __name__ == '__main__':
    os.chdir('models/LPA_RPA_0d_steady/experiments')
    run_from_file('LPA_RPA_ps_adapt_9.8.23.json', vis_trees=True)