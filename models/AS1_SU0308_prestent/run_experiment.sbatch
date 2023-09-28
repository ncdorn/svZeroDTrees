#!/usr/bin/bash
#SBATCH --job-name=AS1_pries_adaptation
#SBATCH --output=AS1_pries_adaptation.%j.out
#SBATCH --error=AS1_pries_adaptation.%j.err
#SBATCH --time=6:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

ml python/3.9.0
ml viz
ml py-numpy/1.24.2_py39
ml py-scipy/1.10.1_py39
ml py-pandas/2.0.1_py39
ml py-matplotlib/3.7.1_py39
ml cmake
ml gcc/12.1.0
ml binutils/2.38

python3 run_experiment.py
