#!/bin/bash

#name of your job
#SBATCH --job-name=svFlowsolver
#SBATCH --partition=amarsden

# Specify the name of the output file. The %j specifies the job ID
#SBATCH --output=svFlowsolver.o%j

# Specify the name of the error file. The %j specifies the job ID
#SBATCH --error=svFlowsolver.e%j

# The walltime you require for your job
#SBATCH --time=2:00:00

# Job priority. Leave as normal for now
#SBATCH --qos=normal

# Number of nodes are you requesting for your job. You can have 24 processors per node
#SBATCH --nodes=1

# Amount of memory you require per node. The default is 4000 MB per node
#SBATCH --mem=16000

# Number of processors per node
#SBATCH --ntasks-per-node=24

# Send an email to this address when your job starts and finishes
#SBATCH --mail-user=ndorn@stanford.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module --force purge
ml devel
ml math
ml openmpi/4.1.2
ml openblas/0.3.4
ml boost/1.79.0
ml system
ml x11
ml mesa
ml qt/5.9.1
ml gcc/12.1.0
ml cmake

/home/users/ndorn/svSolver/svSolver-build/svSolver-build/mypre steady_svZeroD.svpre
srun /home/users/ndorn/svSolver/svSolver-build/svSolver-build/mysolver
