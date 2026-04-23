#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solver_root="$(cd "${repo_root}/../svZeroDSolver" && pwd)"

# Validated Sherlock module stack for svZeroDTrees and pysvzerod on Python 3.12.
module --force purge
ml devel
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-scipy/1.12.0_py312
ml py-pandas/2.2.1_py312
ml py-matplotlib/3.8.3_py312
ml py-numba/0.60.0_py312
ml viz
ml py-vtk/9.4.1_py312
ml cmake
ml binutils/2.38

# Install the sibling solver first, then svZeroDTrees itself.
python -m pip install -e "${solver_root}"
python -m pip install -e "${repo_root}"

# Uncomment the line below to run an experiment from a run_experiment.py file.
# python run_experiment.py
