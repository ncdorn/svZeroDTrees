import vtk
import glob
import pandas as pd
import os

def find_vtp_area(infile):
    # with open(infile):
        # print('file able to be opened!')
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(infile)
    reader.Update()
    poly = reader.GetOutputPort()
    masser = vtk.vtkMassProperties()
    masser.SetInputConnection(poly)
    masser.Update()
    return masser.GetSurfaceArea()

# Sort cap VTP files into inflow / RPA branches / LPA branches. Obtain their names & cap areas.
def vtp_info(mesh_surfaces_path, inflow_tag='inflow', rpa_branch_tag='RPA', lpa_branch_tag='LPA'):
    
    if (mesh_surfaces_path[-1] != '/'):
        mesh_surfaces_path += '/'
    if (mesh_surfaces_path[-1] != '*'):
        mesh_surfaces_path += '*'

    # Isolate just the .vtp's from all files in the mesh-surfaces directory
    filelist_raw = glob.glob(mesh_surfaces_path)
    filelist_raw.sort()               # sort alphabetically, LPA's first, then RPA's
    filelist = []
    for trial in filelist_raw:
        if (trial[-4 : ] == ".vtp"):
          filelist.append(trial)

    # Sort caps into inflow / RPA branches / LPA branches. Store their names & cap areas.
    rpa_info = {}
    lpa_info = {}
    inflow_info = {}

    for vtp_file in filelist:
        tail_name = vtp_file[len(mesh_surfaces_path) - 1 : ]
        if (tail_name[ : len(rpa_branch_tag)] == rpa_branch_tag):
            rpa_info[vtp_file] = find_vtp_area(vtp_file)

        elif (tail_name[ : len(lpa_branch_tag)] == lpa_branch_tag):
            lpa_info[vtp_file] = find_vtp_area(vtp_file)

        elif (tail_name[ : len(inflow_tag)] == inflow_tag):
            inflow_info[vtp_file] = find_vtp_area(vtp_file)
    

    return rpa_info, lpa_info, inflow_info


def compute_flow():
    '''
    compute the flow at the outlet surface of a mesh
    
    awaiting advice from Martin on how to do this'''
    
    pass

def get_coupled_surfaces(simulation_dir):
    '''
    get a map of coupled surfaces to vtp file to find areas and diameters for tree initialization and coupling
    
    assume we aer already in the simulation directory
    '''

    # get a map of surface id's to vtp files
    simulation_name = os.path.basename(simulation_dir)
    surface_id_map = {}
    with open(simulation_dir + '/' + simulation_name + '.svpre', 'r') as ff:
        for line in ff:
            line = line.strip()
            if line.startswith('set_surface_id'):
                line_objs = line.split(' ')
                vtp_file = simulation_dir + '/' + line_objs[1]
                surface_id = line_objs[2]
                surface_id_map[surface_id] = vtp_file
    
    # get a map of sruface id's to coupling blocks
    coupling_map = {}
    reading_coupling_blocks=False
    with open(simulation_dir + '/svZeroD_interface.dat', 'r') as ff:
        for line in ff:
            line = line.strip()
            if not reading_coupling_blocks:
                if line.startswith('svZeroD external coupling block names'):
                    reading_coupling_blocks=True
                    pass
                else:
                    continue
            else:
                if line == '':
                    break
                else:
                    line_objs = line.split(' ')
                    coupling_block = line_objs[0]
                    surface_id = line_objs[1]
                    coupling_map[surface_id] = coupling_block
    
    block_surface_map = {coupling_map[id]: surface_id_map[id] for id in coupling_map.keys()}

    return block_surface_map


def get_outlet_flow(Q_svZeroD):
    '''
    get the outlet flow from a 3D-0D coupled simulation
    '''
    df = pd.read_csv(Q_svZeroD)

    return df


def get_nsteps(solver_input_file, svpre_file):
    '''
    get the timesteps from the solver input file
    '''

    with open(solver_input_file, 'r') as ff:
        for line in ff:
            line = line.strip()
            if line.startswith('Time Step Size:'):
                line_objs = line.split(' ')
                dt = float(line_objs[-1])

    with open(svpre_file, 'r') as ff:
        for line in ff:
            line = line.strip()
            if line.startswith('bct_period'):
                line_objs = line.split(' ')
                period = float(line_objs[-1])

    n_timesteps = int(period / dt)

    return n_timesteps

def prepare_simulation_dir(postop_dir, adapted_dir):
    '''
    prepare the adapted simulation directory with the properly edited simulation files
    '''
    # copy the simulation files to the adapted simulation directory
    os.system('cp -rp ' + postop_dir + '/{*.svpre,*.flow,mesh-complete,solver.inp,svZeroD_interface.dat,*.sh,numstart.dat} ' + adapted_dir)
    # cd into the adapted simulation directory
    os.chdir(adapted_dir)
    # clean uselesss files from the directory
    os.system('rm -r svZeroD_data histor.dat bct.* svFlowsolver.* *-procs_case/ *_svZeroD echo.dat error.dat restart.* geombc.* rcrt.dat')

    # change the name of the svpre file
    os.system('mv ' + os.path.basename(postop_dir) + '.svpre ' + os.path.basename(adapted_dir) + '.svpre')

    # edit the svZeroD_interface.dat file to point to the adapted svzerod_3Dcoupling.json file
    with open('svZeroD_interface.dat', 'r') as ff:
        lines = ff.readlines()

    # change the svZeroD interface path
    lines[4] = os.path.abspath('svzerod_3Dcoupling.json') + '\n'

    with open('svZeroD_interface.dat', 'w') as ff:
        ff.writelines(lines)

    write_svsolver_runscript(os.path.basename(adapted_dir))

def write_svsolver_runscript(model_name, job_name='svFlowSolver', hours=6, nodes=2, procs_per_node=24):
    '''
    write a bash script to submit a job on sherlock'''

    with open('run_solver.sh', 'w') as ff:
        ff.write('#!/bin/bash \n\n')
        ff.write('#name of your job \n')
        ff.write('#SBATCH --job-name=' + job_name + '\n')
        ff.write('#SBATCH --partition=amarsden \n\n')
        ff.write('# Specify the name of the output file. The %j specifies the job ID \n')
        ff.write('#SBATCH --output=' + job_name + '.o%j \n\n')
        ff.write('# Specify the name of the error file. The %j specifies the job ID \n')
        ff.write('#SBATCH --error=' + job_name + '.e%j \n\n')
        ff.write('# The walltime you require for your job \n')
        ff.write('#SBATCH --time=' + str(hours) + ':00:00 \n\n')
        ff.write('# Job priority. Leave as normal for now \n')
        ff.write('#SBATCH --qos=normal \n\n')
        ff.write('# Number of nodes are you requesting for your job. You can have 24 processors per node \n')
        ff.write('#SBATCH --nodes=' + str(nodes) + ' \n\n')
        ff.write('# Amount of memory you require per node. The default is 4000 MB per node \n')
        ff.write('#SBATCH --mem=8000 \n\n')
        ff.write('# Number of processors per node \n')
        ff.write('#SBATCH --ntasks-per-node=' + str(procs_per_node) + ' \n\n')
        ff.write('# Send an email to this address when your job starts and finishes \n')
        ff.write('#SBATCH --mail-user=ndorn@stanford.edu \n')
        ff.write('#SBATCH --mail-type=begin \n')
        ff.write('#SBATCH --mail-type=end \n\n')
        ff.write('module --force purge \n')
        ff.write('ml devel \n')
        ff.write('ml math \n')
        ff.write('ml openmpi/4.1.2 \n')
        ff.write('ml openblas/0.3.4 \n')
        ff.write('ml boost/1.79.0 \n')
        ff.write('ml system \n')
        ff.write('ml x11 \n')
        ff.write('ml mesa \n')
        ff.write('ml qt/5.9.1 \n')
        ff.write('ml gcc/12.1.0 \n')
        ff.write('ml cmake \n\n')
        ff.write('/home/users/ndorn/svSolver/svSolver-build/svSolver-build/mypre ' + model_name + '.svpre \n')
        ff.write('srun /home/users/ndorn/svSolver/svSolver-build/svSolver-build/mysolver \n')