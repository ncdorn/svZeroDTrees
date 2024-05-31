import vtk
import glob
import json
import math
import pandas as pd
import os
import svzerodtrees
from svzerodtrees._config_handler import ConfigHandler

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
    '''
    Sort cap VTP files into inflow / RPA branches / LPA branches. Obtain their names & cap areas.
    
    :param mesh_surfaces_path: path to the mesh surfaces directory
    :param inflow_tag: tag for the inflow surface
    :param rpa_branch_tag: tag for the RPA branch surface
    :param lpa_branch_tag: tag for the LPA branch surface
    
    :return: rpa_info, lpa_info, inflow_info'''
    
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


def prepare_adapted_simdir(postop_dir, adapted_dir):
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


def setup_simdir_from_mesh(sim_dir, zerod_config, write_shell_script=False):
    '''
    setup a simulation directory solely from a mesh-complete.
    :param sim_dir: path to the simulation directory where the mesh complete is located
    '''
    # get the period of the inflow file
    # period = get_inflow_period(inflow_file)


    mesh_complete = os.path.join(sim_dir, 'mesh-complete')
    
    # write svzerod_3dcoupling file
    zerod_config_handler = ConfigHandler.from_json(zerod_config)
    period = zerod_config_handler.generate_inflow_file(sim_dir)
    zerod_config_handler.generate_threed_coupler(sim_dir)
    

    # write svpre file
    inlet_idx, outlet_idxs = write_svpre_file(sim_dir, mesh_complete, period)

    # write svzerod interface file
    write_svzerod_interface(sim_dir, outlet_idxs) # PATH TO ZEROD COUPLER NEEDS TO BE CHANGED IF ON SHERLOCK

    # write solver input file
    dt, num_timesteps, timesteps_btwn_restart = write_solver_inp(sim_dir, outlet_idxs, period, 2)

    # write numstart file
    write_numstart(sim_dir)

    # write run script
    write_svsolver_runscript(sim_dir)

    # move inflow file to simulation directory
    # os.system('cp ' + inflow_file + ' ' + sim_dir)

    return num_timesteps


def get_inflow_period(inflow_file):
    '''
    get the period of the inflow waveform
    '''
    df = pd.read_csv(inflow_file, sep='\s+', header=None, names=['time', 'flow'])
    period = df['time'].max()

    return period


def write_svsolver_runscript(sim_dir, job_name='svFlowSolver', hours=6, nodes=4, procs_per_node=24):
    '''
    write a bash script to submit a job on sherlock'''

    print('writing svsolver runscript...')

    with open(os.path.join(sim_dir, 'run_solver.sh'), 'w') as ff:
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
        ff.write('/home/users/ndorn/svSolver/svSolver-build/svSolver-build/mypre ' + os.path.basename(sim_dir) + '.svpre \n')
        ff.write('srun /home/users/ndorn/svSolver/svSolver-build/svSolver-build/mysolver \n')
        ff.write(f'cd {nodes * procs_per_node}-procs_case \n')
        ff.write(f'postsolver -start 1500 -stop 2000 -incr 50 -sol -vtkcombo -vtu post.vtu \n')
        ff.write('mv post.vtu .. \n')


def write_svpre_file(sim_dir, mesh_complete, period=1.0):
    '''
    write the svpre file for the simulation
    
    :param sim_dir: path to the simulation directory
    :param mesh_complete: path to the mesh-complete directory
    :param period: period of the flow waveform in seconds'''

    print(f'writing {os.path.basename(os.path.abspath(sim_dir))}.svpre...')

    mesh_vtu = glob.glob(os.path.join(mesh_complete, '*.mesh.vtu'))[0]
    walls_combined = os.path.join(mesh_complete, 'walls_combined.vtp')
    mesh_exterior = glob.glob(os.path.join(mesh_complete, '*.exterior.vtp'))[0]
    inflow_vtp = glob.glob(os.path.join(mesh_complete, 'mesh-surfaces/inflow.vtp'))[0]

    filelist_raw = glob.glob(os.path.join(mesh_complete, 'mesh-surfaces/*.vtp'))

    filelist = [file for file in filelist_raw if 'wall' not in file]
    filelist.sort()

    common_path = os.path.commonpath(filelist + [mesh_vtu] + [walls_combined] + [mesh_exterior] + [inflow_vtp])
    print(common_path)

    filelist = [path.replace(common_path, 'mesh-complete') for path in filelist]
    mesh_vtu = mesh_vtu.replace(common_path, 'mesh-complete')
    walls_combined = walls_combined.replace(common_path, 'mesh-complete')
    mesh_exterior = mesh_exterior.replace(common_path, 'mesh-complete')
    inflow_vtp = inflow_vtp.replace(common_path, 'mesh-complete')

    with open(os.path.join(sim_dir, os.path.basename(os.path.abspath(sim_dir)) + '.svpre'), 'w') as svpre:
        svpre.write('mesh_and_adjncy_vtu ' + mesh_vtu + '\n')
        # assign the surface ids
        outlet_idxs = []
        for i, file in enumerate([mesh_exterior] + filelist):
            svpre.write('set_surface_id_vtp ' + file + ' ' + str(i + 1) + '\n')
            if file != inflow_vtp and file != mesh_exterior:
                outlet_idxs.append(str(i + 1))
            elif file == inflow_vtp:
                inlet_idx = str(i + 1)
        
        svpre.write('fluid_density 1.06\n')
        svpre.write('fluid_viscosity 0.04\n')
        svpre.write('initial_pressure 0\n')
        svpre.write('initial_velocity 0.0001 0.0001 0.0001\n')
        svpre.write('prescribed_velocities_vtp ' + inflow_vtp + '\n')
        svpre.write('bct_analytical_shape parabolic\n')
        svpre.write('bct_period ' + str(period) + '\n')
        svpre.write('bct_point_number 201\n')
        svpre.write('bct_fourier_mode_number 10\n')
        svpre.write('bct_create ' + inflow_vtp + ' inflow.flow\n')
        svpre.write('bct_write_dat bct.dat\n')
        svpre.write('bct_write_vtp bct.vtp\n')
        # remove inflow and list the pressure vtps for the outlet caps
        filelist.remove(inflow_vtp)
        for cap in filelist:
            svpre.write('pressure_vtp ' + cap + ' 0\n')
        svpre.write('noslip_vtp ' + walls_combined + '\n')
        svpre.write('write_geombc geombc.dat.1\n')
        svpre.write('write_restart restart.0.1\n')

    return inlet_idx, outlet_idxs


def write_svzerod_interface(sim_dir, outlet_idxs, interface_path='/home/users/ndorn/svZeroDSolver/Release/src/interface/libsvzero_interface.so'):
    '''
    write the svZeroD_interface.dat file for the simulation
    
    :param sim_dir: path to the simulation directory
    :param zerod_coupler: path to the 3D-0D coupling file'''

    print('writing svZeroD interface file...')

    zerod_coupler = os.path.join(sim_dir, 'svzerod_3Dcoupling.json')

    threed_coupler = ConfigHandler.from_json(zerod_coupler, is_pulmonary=False, is_threed_interface=True)

    # get a map of bc names to outlet idxs
    outlet_blocks = [block.name for block in list(threed_coupler.coupling_blocks.values())]

    bc_to_outlet = zip(outlet_blocks, outlet_idxs)

    with open(os.path.join(sim_dir, 'svZeroD_interface.dat'), 'w') as ff:
        ff.write('interface library path: \n')
        ff.write(interface_path + '\n\n')

        ff.write('svZeroD input file: \n')
        ff.write(os.path.abspath(zerod_coupler + '\n\n'))
        
        ff.write('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file): \n')
        for bc, idx in bc_to_outlet:
            ff.write(bc + ' ' + idx + '\n')
        ff.write('\n')
        ff.write('Initialize external coupling block flows: \n')
        ff.write('0\n\n')

        ff.write('External coupling block initial flows (one number is provided, it is applied to all coupling blocks: \n')
        ff.write('0.0\n\n')

        ff.write('Initialize external coupling block pressures: \n')
        ff.write('1\n\n')

        ff.write('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks): \n')
        ff.write('60.0\n\n')


def write_solver_inp(sim_dir, outlet_idxs, period, n_cycles, dt=.001):
    '''
    write the solver.inp file for the simulation'''

    print('writing solver.inp...')

    # Determine the number of time steps
    num_timesteps = int(math.ceil(period * n_cycles / dt))
    steps_btw_restarts = int(round(period / dt / 50))

    with open(os.path.join(sim_dir, 'solver.inp'),'w') as solver_inp:
        solver_inp.write('Density: 1.06\n')
        solver_inp.write('Viscosity: 0.04\n\n')

        solver_inp.write('Number of Timesteps: ' + str(num_timesteps) + '\n')
        solver_inp.write('Time Step Size: ' + str(dt) + '\n\n')

        solver_inp.write('Number of Timesteps between Restarts: ' + str(steps_btw_restarts) + '\n')
        solver_inp.write('Number of Force Surfaces: 1\n')
        solver_inp.write('Surface ID\'s for Force Calculation: 1\n')
        solver_inp.write('Force Calculation Method: Velocity Based\n')
        solver_inp.write('Print Average Solution: True\n')
        solver_inp.write('Print Error Indicators: False\n\n')

        solver_inp.write('Time Varying Boundary Conditions From File: True\n\n')

        solver_inp.write('Step Construction: 0 1 0 1 0 1 0 1 0 1\n\n')

        solver_inp.write('Number of Neumann Surfaces: ' + str(len(outlet_idxs)) + '\n')
        # List of RCR surfaces starts at ID no. 3 (bc 1 = mesh exterior, 2 = inflow) [Ingrid] but we generalize this just in case [Nick]
        rcr_list = 'List of Neumann Surfaces:\t'
        for idx in outlet_idxs:
            rcr_list += str(idx) + '\t'
        solver_inp.write(rcr_list + '\n')
        solver_inp.write('Use svZeroD for Boundary Conditions: True\n')
        solver_inp.write('Number of Timesteps for svZeroD Initialization: 0\n\n')

        solver_inp.write('Pressure Coupling: Implicit\n')
        solver_inp.write('Number of Coupled Surfaces: ' + str(len(outlet_idxs)) + '\n\n')

        solver_inp.write('Backflow Stabilization Coefficient: 0.2\n')
        solver_inp.write('Residual Control: True\n')
        solver_inp.write('Residual Criteria: 0.001\n')
        solver_inp.write('Minimum Required Iterations: 3\n')
        solver_inp.write('svLS Type: NS\n')
        solver_inp.write('Number of Krylov Vectors per GMRES Sweep: 100\n')
        solver_inp.write('Number of Solves per Left-hand-side Formation: 1\n')
        solver_inp.write('Tolerance on Momentum Equations: 0.05\n')
        solver_inp.write('Tolerance on Continuity Equations: 0.05\n')
        solver_inp.write('Tolerance on svLS NS Solver: 0.05\n')
        solver_inp.write('Maximum Number of Iterations for svLS NS Solver: 10\n')
        solver_inp.write('Maximum Number of Iterations for svLS Momentum Loop: 15\n')
        solver_inp.write('Maximum Number of Iterations for svLS Continuity Loop: 400\n')
        solver_inp.write('Time Integration Rule: Second Order\n')
        solver_inp.write('Time Integration Rho Infinity: 0.5\n')
        solver_inp.write('Flow Advection Form: Convective\n')
        solver_inp.write('Quadrature Rule on Interior: 2\n')
        solver_inp.write('Quadrature Rule on Boundary: 3\n')
    return dt, num_timesteps, steps_btw_restarts


def write_numstart(sim_dir):

    print('writing numstart.dat...')
    with open(os.path.join(sim_dir, 'numstart.dat'), 'w') as numstart:
        numstart.write('0')


def get_lpa_rpa_idxs(svpre_file):
    '''
    get the lpa and rpa indexes from the svpre file'''
    lpa_idxs = []
    rpa_idxs = []
    with open(svpre_file, 'r') as ff:
        for line in ff:
            line.strip()
            if line.startswith('set_surface_id_vtp'):
                line_objs = line.split(' ')
                if 'lpa' in line_objs[1].lower():
                    lpa_idxs.append(line_objs[2].strip('\n'))
                elif 'rpa' in line_objs[1].lower():
                    rpa_idxs.append(line_objs[2].strip('\n'))

    return lpa_idxs, rpa_idxs


def compute_flow_split(Q_svZeroD, svpre_file, n_steps=1000):
    '''
    compute the flow at the outlet surface of a mesh
    
    :param Q_svZeroD: path to the svZeroD flow file
    :param svpre_file: path to the svpre file'''

    # get the indices of the LPA and RPA outlets
    lpa_idxs, rpa_idxs = get_lpa_rpa_idxs(svpre_file)

    q = pd.read_csv(Q_svZeroD, sep='\s+')

    q_lpa = 0
    q_rpa = 0

    for idx, q in q.items():
        if idx in lpa_idxs:
            q_lpa += q[-n_steps:].mean()
        elif idx in rpa_idxs:
            q_rpa += q[-n_steps:].mean()

    lpa_split = round(q_lpa / (q_lpa + q_rpa), 3)
    rpa_split = round(q_rpa / (q_lpa + q_rpa), 3)

    print('flow split LPA/RPA: ' + str(round(q_lpa / (q_lpa + q_rpa), 3) * 100) + '% /' + str(round(q_rpa / (q_lpa + q_rpa), 3) * 100) + '%')

    return lpa_split, rpa_split


def generate_flowsplit_results():

    preop_split = compute_flow_split('preop/Q_svZeroD', 'preop/preop.svpre')
    postop_split = compute_flow_split('postop/Q_svZeroD', 'postop/postop.svpre')
    adapted_split = compute_flow_split('adapted/Q_svZeroD', 'adapted/adapted.svpre')

    with open('flow_split_results.txt', 'w') as f:
        f.write('flow splits as LPA/RPA\n\n')
        f.write('preop flow split: ' + str(preop_split[0]) + '/' + str(preop_split[1]) + '\n')
        f.write('postop flow split: ' + str(postop_split[0]) + '/' + str(postop_split[1]) + '\n')
        f.write('adapted flow split: ' + str(adapted_split[0]) + '/' + str(adapted_split[1]) + '\n')


if __name__ == '__main__':
    # setup a simulation dir from mesh
    os.chdir('../threed_models/AS2_opt_fs')

    setup_simdir_from_mesh('preop', 'zerod/AS2_prestent.json')


    
