import os

from .simulation_file import SimulationFile


class SolverRunscript(SimulationFile):
    '''
    class to handle the solver runscript file in the simulation directory (run_solver.sh)'''

    def __init__(self, path):
        '''
        initialize the solver runscript object'''
        super().__init__(path)

    def initialize(self):
        '''
        initialize the solver runscript object'''

        with open(self.path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        pass

    def write(self, 
              nodes=3, 
              procs_per_node=24, 
              hours=6, 
              memory=16,
              svfsiplus_path='/home/users/ndorn/svMP-build/svMultiPhysics-build/bin/svmultiphysics',
              working_dir=None,
              partition='amarsden',
              qos='normal',
              modules=None,
              mail_user='ndorn@stanford.edu',
              mail_types=None):
        '''
        write the solver runscript file'''

        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.hours = hours
        self.memory = memory
        self.svfsiplus_path = svfsiplus_path
        self.working_dir = os.path.abspath(working_dir or os.path.dirname(self.path))
        self.partition = partition
        self.qos = qos
        output_path = os.path.join(self.working_dir, "svFlowSolver.o%j")
        error_path = os.path.join(self.working_dir, "svFlowSolver.e%j")
        if modules is None:
            modules = [
                "devel",
                "math",
                "openmpi",
                "openblas",
                "boost",
                "system",
                "x11",
                "mesa",
                "qt",
                "gcc/14.2.0",
                "cmake",
            ]
        if mail_types is None:
            mail_types = ["begin", "end"]

        print('writing solver runscript file...')

        with open(self.path, 'w') as ff:
            ff.write("#!/bin/bash\n\n")
            ff.write("#name of your job \n")
            ff.write("#SBATCH --job-name=svFlowSolver\n")
            ff.write(f"#SBATCH --partition={partition}\n\n")
            ff.write(f"#SBATCH --chdir={self.working_dir}\n\n")
            ff.write("# Specify the name of the output file. The %j specifies the job ID\n")
            ff.write(f"#SBATCH --output={output_path}\n\n")
            ff.write("# Specify the name of the error file. The %j specifies the job ID \n")
            ff.write(f"#SBATCH --error={error_path}\n\n")
            ff.write("# The walltime you require for your job \n")
            ff.write(f"#SBATCH --time={hours}:00:00\n\n")
            ff.write("# Job priority. Leave as normal for now \n")
            ff.write(f"#SBATCH --qos={qos}\n\n")
            ff.write("# Number of nodes are you requesting for your job. You can have 24 processors per node \n")
            ff.write(f"#SBATCH --nodes={nodes} \n\n")
            ff.write("# Amount of memory you require per node. The default is 4000 MB per node \n")
            ff.write(f"#SBATCH --mem={memory}G\n\n")
            ff.write("# Number of processors per node \n")
            ff.write(f"#SBATCH --ntasks-per-node={procs_per_node} \n\n")
            if mail_user:
                ff.write("# Send an email to this address when your job starts and finishes \n")
                ff.write(f"#SBATCH --mail-user={mail_user} \n")
                for mail_type in mail_types:
                    ff.write(f"#SBATCH --mail-type={mail_type} \n")
            ff.write("module --force purge\n\n")
            for module in modules:
                ff.write(f"ml {module}\n")
            ff.write("\n")
            ff.write(f"cd {self.working_dir}\n")
            ff.write('if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ -n "${SLURM_TRES_PER_TASK:-}" ]; then\n')
            ff.write('  case "${SLURM_TRES_PER_TASK}" in\n')
            ff.write('    cpu=*)\n')
            ff.write('      _svzt_tres_cpus="${SLURM_TRES_PER_TASK#cpu=}"\n')
            ff.write('      _svzt_tres_cpus="${_svzt_tres_cpus%%,*}"\n')
            ff.write('      if [ "${SLURM_CPUS_PER_TASK}" != "${_svzt_tres_cpus}" ]; then\n')
            ff.write('        unset SLURM_TRES_PER_TASK\n')
            ff.write('      fi\n')
            ff.write('      unset _svzt_tres_cpus\n')
            ff.write('      ;;\n')
            ff.write('  esac\n')
            ff.write('fi\n')
            ff.write(f"srun {svfsiplus_path} svFSIplus.xml\n")
        
        self.is_written = True
