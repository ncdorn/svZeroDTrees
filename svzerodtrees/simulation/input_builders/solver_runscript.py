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
              svfsiplus_path='/home/users/ndorn/svMP-procfix/svMP-build/svMultiPhysics-build/bin/svmultiphysics'):
        '''
        write the solver runscript file'''

        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.hours = hours
        self.memory = memory
        self.svfsiplus_path = svfsiplus_path

        print('writing solver runscript file...')

        with open(self.path, 'w') as ff:
            ff.write("#!/bin/bash\n\n")
            ff.write("#name of your job \n")
            ff.write("#SBATCH --job-name=svFlowSolver\n")
            ff.write("#SBATCH --partition=amarsden\n\n")
            ff.write("# Specify the name of the output file. The %j specifies the job ID\n")
            ff.write("#SBATCH --output=svFlowSolver.o%j\n\n")
            ff.write("# Specify the name of the error file. The %j specifies the job ID \n")
            ff.write("#SBATCH --error=svFlowSolver.e%j\n\n")
            ff.write("# The walltime you require for your job \n")
            ff.write(f"#SBATCH --time={hours}:00:00\n\n")
            ff.write("# Job priority. Leave as normal for now \n")
            ff.write("#SBATCH --qos=normal\n\n")
            ff.write("# Number of nodes are you requesting for your job. You can have 24 processors per node \n")
            ff.write(f"#SBATCH --nodes={nodes} \n\n")
            ff.write("# Amount of memory you require per node. The default is 4000 MB per node \n")
            ff.write(f"#SBATCH --mem={memory}G\n\n")
            ff.write("# Number of processors per node \n")
            ff.write(f"#SBATCH --ntasks-per-node={procs_per_node} \n\n")
            ff.write("# Send an email to this address when your job starts and finishes \n")
            ff.write("#SBATCH --mail-user=ndorn@stanford.edu \n")
            ff.write("#SBATCH --mail-type=begin \n")
            ff.write("#SBATCH --mail-type=end \n")
            ff.write("module --force purge\n\n")
            ff.write("ml devel\n")
            ff.write("ml math\n")
            ff.write("ml openmpi\n")
            ff.write("ml openblas\n")
            ff.write("ml boost\n")
            ff.write("ml system\n")
            ff.write("ml x11\n")
            ff.write("ml mesa\n")
            ff.write("ml qt\n")
            ff.write("ml gcc/14.2.0\n")
            ff.write("ml cmake\n\n")
            ff.write(f"srun {svfsiplus_path} svFSIplus.xml\n")
        
        self.is_written = True
