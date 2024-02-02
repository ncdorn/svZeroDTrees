#!/bin/bash

# these are the modules required to run svzerodtrees on sherlock.
# everything must be run with python 3.9

ml python/3.9.0
ml viz
ml py-numpy/1.24.2_py39
ml py-scipy/1.10.1_py39
ml py-pandas/2.0.1_py39
ml py-matplotlib/3.7.1_py39
ml cmake
ml gcc/12.1.0
ml binutils/2.38

# installing svzerodtrees package
pip install -e .
# install svzerodplus (must be build on sherlock)
pip install git+https://github.com/StanfordCBCL/svZeroDPlus.git
# install vtk
pip install vtk

# uncomment the line below to run an experiment from a run_experiment.py file
# python3 run_experiment.py
