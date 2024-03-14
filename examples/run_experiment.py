from svzerodtrees.interface import *
import os

'''
example script for running an svZeroDTrees experiment from a json config file
'''
if __name__ == '__main__':
    os.chdir('/Users/ndorn/ndorn@stanford.edu - Google Drive/My Drive/Stanford/PhD/Simvascular/zerod_models/AS2_prestent/experiments')
    run_from_file('AS2_stent.json')