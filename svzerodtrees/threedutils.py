import vtk
import glob
import json
import math
import pandas as pd
import os
import svzerodtrees
from svzerodtrees._config_handler import ConfigHandler
import xml.etree.ElementTree as ET
import xml.dom.minidom

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
def vtp_info(mesh_surfaces_path, inflow_tag='inflow', rpa_branch_tag='RPA', lpa_branch_tag='LPA', pulmonary=True):
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

    # store inflow area
    inflow_info = {}


    # if pulmonary, sort caps into inflow / RPA branches / LPA branches. Store their names & cap areas.
    if pulmonary:
        rpa_info = {}
        lpa_info = {}

        for vtp_file in filelist:
            tail_name = vtp_file[len(mesh_surfaces_path) - 1 : ]
            if (tail_name[ : len(rpa_branch_tag)] == rpa_branch_tag):
                rpa_info[vtp_file] = find_vtp_area(vtp_file)

            elif (tail_name[ : len(lpa_branch_tag)] == lpa_branch_tag):
                lpa_info[vtp_file] = find_vtp_area(vtp_file)

            elif (tail_name[ : len(inflow_tag)] == inflow_tag):
                inflow_info[vtp_file] = find_vtp_area(vtp_file)
        

        return rpa_info, lpa_info, inflow_info
    
    else: # return cap info
        cap_info = {}

        for vtp_file in filelist:
            basename = os.path.basename(vtp_file)
            if inflow_tag in basename:
                continue
            elif 'wall' not in basename:
                cap_info[basename] = find_vtp_area(vtp_file)
        
        return cap_info


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

    write_svsolver_runscript(os.getcwd(), 20)


def setup_simdir_from_mesh(sim_dir, zerod_config, 
                           svfsiplus_path='/home/users/ndorn/svfsiplus-build/svFSI-build/mysvfsi'):
    '''
    setup a simulation directory solely from a mesh-complete.
    :param sim_dir: path to the simulation directory where the mesh complete is located
    '''
    # get the period of the inflow file
    # period = get_inflow_period(inflow_file)

    mesh_complete = os.path.join(sim_dir, 'mesh-complete')
    
    # check that the mesh surfaces are alright
        # check mesh surface names and amke sure they are good to go
    rename_msh_surfs(os.path.join(mesh_complete, 'mesh-surfaces'))

    # write svzerod_3dcoupling file
    zerod_config_handler = ConfigHandler.from_json(zerod_config)
    # period = zerod_config_handler.generate_inflow_file(sim_dir)
    os.system('pwd')
    zerod_config_handler.generate_threed_coupler(sim_dir, inflow_from_0d=True)

    # write svpre file
    # inlet_idx, outlet_idxs = write_svpre_file(sim_dir, mesh_complete)

    # write svzerod interface file
    write_svzerod_interface(sim_dir) # PATH TO ZEROD COUPLER NEEDS TO BE CHANGED IF ON SHERLOCK

    # write solver input file
    # dt, num_timesteps, steps_btwn_restart = write_solver_inp(sim_dir, outlet_idxs, n_cycles=2)

    write_svfsiplus_xml(sim_dir)

    # write numstart file
    # write_numstart(sim_dir)

    # write run script
    write_svfsi_runscript(sim_dir, svfsiplus_path)

    # move inflow file to simulation directory
    # os.system('cp ' + inflow_file + ' ' + sim_dir)


def setup_svfsi_simdir(sim_dir, zerod_config, svfsi_path='/home/users/ndorn/svfsiplus-build/svFSI-build/mysvfsi'):

    # need to write a method to create a svfsiplus.xml file
    pass


def write_svfsiplus_xml(sim_dir, n_tsteps=5000, dt=0.001, mesh_complete='mesh-complete'):
    '''
    write an svFSIplus.xml file from a simulation directory which contains a mesh surfaces directory
    '''

    # set mesh complete diretory
    mesh_complete = os.path.join(sim_dir, mesh_complete)

    print('writing svFSIplus.xml...')

    # generate XML tree
    svfsifile = ET.Element("svFSIFile")
    svfsifile.set("version", "0.1")

    # General Simulation Parameters
    gensimparams = ET.SubElement(svfsifile, "GeneralSimulationParameters")

    cont_prev_sim = ET.SubElement(gensimparams, "Continue_previous_simulation")
    cont_prev_sim.text = "false"

    num_spatial_dims = ET.SubElement(gensimparams, "Number_of_spatial_dimensions")
    num_spatial_dims.text = "3"

    num_time_steps = ET.SubElement(gensimparams, "Number_of_time_steps")
    num_time_steps.text = str(n_tsteps)

    time_step_size = ET.SubElement(gensimparams, "Time_step_size")
    time_step_size.text = str(dt)

    spec_radius = ET.SubElement(gensimparams, "Spectral_radius_of_infinite_time_step")
    spec_radius.text = "0.5"

    stop_trigger = ET.SubElement(gensimparams, "Searched_file_name_to_trigger_stop")
    stop_trigger.text = "STOP_SIM"

    save_results_to_vtk = ET.SubElement(gensimparams, "Save_results_to_VTK_format")
    save_results_to_vtk.text = "1"

    name_prefix = ET.SubElement(gensimparams, "Name_prefix_of_saved_VTK_files")
    name_prefix.text = "result"

    increment_vtk = ET.SubElement(gensimparams, "Increment_in_saving_VTK_files")
    increment_vtk.text = "20"

    start_saving_tstep = ET.SubElement(gensimparams, "Start_saving_after_time_step")
    start_saving_tstep.text = "1"

    incrememnt_restart = ET.SubElement(gensimparams, "Increment_in_saving_restart_files")
    incrememnt_restart.text = "10"

    convert_bin_vtk = ET.SubElement(gensimparams, "Convert_BIN_to_VTK_format")
    convert_bin_vtk.text = "0"

    verbose = ET.SubElement(gensimparams, "Verbose")
    verbose.text = "1"

    warning = ET.SubElement(gensimparams, "Warning")
    warning.text = "0"

    debug = ET.SubElement(gensimparams, "Debug")
    debug.text = "0"

    # add mesh
    add_mesh = ET.SubElement(svfsifile, "Add_mesh")
    add_mesh.set("name", "msh")

    msh_file_path = ET.SubElement(add_mesh, "Mesh_file_path")
    msh_file_path.text = os.path.join(mesh_complete, 'mesh-complete.mesh.vtu')

    # add faces to mesh
    filelist_raw = glob.glob(os.path.join(mesh_complete, 'mesh-surfaces/*.vtp'))

    filelist = [file for file in filelist_raw if 'wall' not in file]
    filelist.sort()


    for file in filelist:
        add_face = ET.SubElement(add_mesh, "Add_face")
        add_face.set("name", os.path.basename(file).split('.')[0])

        face_file_path = ET.SubElement(add_face, "Face_file_path")
        face_file_path.text = file
    
    add_wall = ET.SubElement(add_mesh, "Add_face")
    add_wall.set("name", "wall")

    wall_file_path = ET.SubElement(add_wall, "Face_file_path")
    wall_file_path.text = os.path.join(mesh_complete, 'walls_combined.vtp')

    # add equation
    add_eqn = ET.SubElement(svfsifile, "Add_equation")
    add_eqn.set("type", "fluid")

    coupled = ET.SubElement(add_eqn, "Coupled")
    coupled.text = "1"

    min_iterations = ET.SubElement(add_eqn, "Min_iterations")
    min_iterations.text = "3"

    max_iterations = ET.SubElement(add_eqn, "Max_iterations")
    max_iterations.text = "10"

    tolerance = ET.SubElement(add_eqn, "Tolerance")
    tolerance.text = "1e-3"

    backflow_stab = ET.SubElement(add_eqn, "Backflow_stabilization_coefficient")
    backflow_stab.text = "0.2"

    density = ET.SubElement(add_eqn, "Density")
    density.text = "1.06"

    viscosity = ET.SubElement(add_eqn, "Viscosity", {"model": "Constant"})
    value = ET.SubElement(viscosity, "Value")
    value.text = "0.04"

    output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
    velocity = ET.SubElement(output, "Velocity")
    velocity.text = "true"

    pressure = ET.SubElement(output, "Pressure")
    pressure.text = "true"

    traction = ET.SubElement(output, "Traction")
    traction.text = "true"

    wss = ET.SubElement(output, "WSS")
    wss.text = "true"

    vorticity = ET.SubElement(output, "Vorticity")
    vorticity.text = "true"

    divergence = ET.SubElement(output, "Divergence")
    divergence.text = "true"

    ls = ET.SubElement(add_eqn, "LS", {"type": "NS"})

    linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
    preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
    preconditioner.text = "fsils"

    ls_max_iterations = ET.SubElement(ls, "Max_iterations")
    ls_max_iterations.text = "10"

    ns_gm_max_iterations = ET.SubElement(ls, "NS_GM_max_iterations")
    ns_gm_max_iterations.text = "3"

    ns_cg_max_iterations = ET.SubElement(ls, "NS_CG_max_iterations")
    ns_cg_max_iterations.text = "500"

    ls_tolerance = ET.SubElement(ls, "Tolerance")
    ls_tolerance.text = "1e-3"

    ns_gm_tolerance = ET.SubElement(ls, "NS_GM_tolerance")
    ns_gm_tolerance.text = "1e-3"

    ns_cg_tolerance = ET.SubElement(ls, "NS_CG_tolerance")
    ns_cg_tolerance.text = "1e-3"

    krylov_space_dim = ET.SubElement(ls, "Krylov_space_dimension")
    krylov_space_dim.text = "50"



    couple_to_svzerod = ET.SubElement(add_eqn, "Couple_to_svZeroD")
    couple_to_svzerod.set("type", "SI")

    # add boundary conditions
    for file in filelist:
        add_bc = ET.SubElement(add_eqn, "Add_BC")
        add_bc.set("name", os.path.basename(file).split('.')[0])
        
        typ = ET.SubElement(add_bc, "Type")
        typ.text = "Neu"
        time_dep = ET.SubElement(add_bc, "Time_dependence")
        time_dep.text = "Coupled"
    
    # add wall bc
    add_wall_bc = ET.SubElement(add_eqn, "Add_BC")
    add_wall_bc.set("name", "wall")
    typ = ET.SubElement(add_wall_bc, "Type")
    typ.text = "Dir"
    time_dep = ET.SubElement(add_wall_bc, "Time_dependence")
    time_dep.text = "Steady"
    value = ET.SubElement(add_wall_bc, "Value")
    value.text = "0.0"

    # Create the XML tree
    tree = ET.ElementTree(svfsifile)

    ET.indent(tree.getroot())

    # def prettify(elem):
    #     """Return a pretty-printed XML string for the Element."""
    #     rough_string = ET.tostring(elem, 'utf-8')
    #     reparsed = xml.dom.minidom.parseString(rough_string)
    #     return reparsed.toprettyxml(indent="  ")
    

    # pretty_xml_str = prettify(svfsifile)

    # print(pretty_xml_str)

    # Write the XML to a file
    with open(os.path.join(sim_dir, "svFSIplus.xml"), "wb") as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)


def get_inflow_period(inflow_file):
    '''
    get the period of the inflow waveform
    '''
    df = pd.read_csv(inflow_file, sep='\s+', header=None, names=['time', 'flow'])
    period = df['time'].max()

    return period


def write_svfsi_runscript(sim_dir,
                             svfsiplus_path='/home/users/ndorn/svfsiplus-build/svFSI-build/mysvfsi',
                             hours=6, nodes=2, procs_per_node=24):
    '''
    write a bash script to submit a job on sherlock'''

    print('writing svFSIplus runscript...')

    with open(os.path.join(sim_dir, 'run_solver.sh'), 'w') as ff:
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
        ff.write("#SBATCH --mem=8G\n\n")
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


def write_svpre_file(sim_dir, mesh_complete):
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
        # svpre.write('prescribed_velocities_vtp ' + inflow_vtp + '\n')
        # svpre.write('bct_analytical_shape parabolic\n')
        # svpre.write('bct_period ' + str(period) + '\n')
        # svpre.write('bct_point_number 201\n')
        # svpre.write('bct_fourier_mode_number 10\n')
        # svpre.write('bct_create ' + inflow_vtp + ' inflow.flow\n')
        # svpre.write('bct_write_dat bct.dat\n')
        # svpre.write('bct_write_vtp bct.vtp\n')
        # # remove inflow and list the pressure vtps for the outlet caps
        # filelist.remove(inflow_vtp)
        for cap in filelist:
            svpre.write('pressure_vtp ' + cap + ' 0\n')
        svpre.write('noslip_vtp ' + walls_combined + '\n')
        svpre.write('write_geombc geombc.dat.1\n')
        svpre.write('write_restart restart.0.1\n')

    return inlet_idx, outlet_idxs


def write_svzerod_interface(sim_dir, interface_path='/home/users/ndorn/svZeroDSolver/Release/src/interface/libsvzero_interface.so'):
    '''
    write the svZeroD_interface.dat file for the simulation
    
    :param sim_dir: path to the simulation directory
    :param zerod_coupler: path to the 3D-0D coupling file'''

    print('writing svZeroD interface file...')

    zerod_coupler = os.path.join(sim_dir, 'svzerod_3Dcoupling.json')

    threed_coupler = ConfigHandler.from_json(zerod_coupler, is_pulmonary=False, is_threed_interface=True)

    # get a map of bc names to outlet idxs
    outlet_blocks = [block.name for block in list(threed_coupler.coupling_blocks.values())]

    # bc_to_outlet = zip(outlet_blocks, outlet_idxs)

    with open(os.path.join(sim_dir, 'svZeroD_interface.dat'), 'w') as ff:
        ff.write('interface library path: \n')
        ff.write(interface_path + '\n\n')

        ff.write('svZeroD input file: \n')
        ff.write(os.path.abspath(zerod_coupler + '\n\n'))
        
        ff.write('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file): \n')
        for idx, bc in enumerate(outlet_blocks):
            ff.write(f'{bc} {idx}\n')

        ff.write('\n')
        ff.write('Initialize external coupling block flows: \n')
        ff.write('0\n\n')

        ff.write('External coupling block initial flows (one number is provided, it is applied to all coupling blocks): \n')
        ff.write('0.0\n\n')

        ff.write('Initialize external coupling block pressures: \n')
        ff.write('1\n\n')

        ff.write('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks): \n')
        ff.write('60.0\n\n')


def write_solver_inp(sim_dir, outlet_idxs, n_cycles, dt=.001):
    '''
    write the solver.inp file for the simulation'''

    print('writing solver.inp...')

    # Determine the number of time steps
    num_timesteps = int(math.ceil(n_cycles / dt))
    steps_btw_restarts = int(round(1.0 / dt / 50.0))

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

        solver_inp.write('Time Varying Boundary Conditions From File: False\n\n')

        solver_inp.write('Step Construction: 0 1 0 1 0 1 0 1 0 1\n\n')

        solver_inp.write('Number of Neumann Surfaces: ' + str(len(outlet_idxs)) + '\n')
        # List of RCR surfaces starts at ID no. 3 (bc 1 = mesh exterior, 2 = inflow) [Ingrid] but we generalize this just in case [Nick]
        rcr_list = 'List of Neumann Surfaces:\t' # TODO: need to update this to accept the inflow and adjust this file to not take BCT
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


def generate_flowsplit_results(preop_simdir, postop_simdir, adapted_simdir):

    preop_split = compute_flow_split(os.path.join(preop_simdir, 'Q_svZeroD'), os.path.join(preop_simdir, 'preop.svpre'))
    postop_split = compute_flow_split(os.path.join(postop_simdir, 'Q_svZeroD'), os.path.join(postop_simdir, 'postop.svpre'))
    adapted_split = compute_flow_split(os.path.join(adapted_simdir, 'Q_svZeroD'), os.path.join(adapted_simdir, 'adapted.svpre'))

    with open(os.path.join(os.path.dirname(preop_simdir), 'flow_split_results.txt'), 'w') as f:
        f.write('flow splits as LPA/RPA\n\n')
        f.write('preop flow split: ' + str(preop_split[0]) + '/' + str(preop_split[1]) + '\n')
        f.write('postop flow split: ' + str(postop_split[0]) + '/' + str(postop_split[1]) + '\n')
        f.write('adapted flow split: ' + str(adapted_split[0]) + '/' + str(adapted_split[1]) + '\n')


def rename_msh_surfs(msh_surf_dir):
    '''
    rename the mesh surfaces to have a consistent naming convention
    '''
    filelist_raw = glob.glob(msh_surf_dir + '/*')
    # remove potential duplicate and triplicate faces

    # make the mpa inlet the inflow.vtp if that does not exist
    filelist = [file for file in filelist_raw if 'wall' not in file.lower()]
 
    if 'inflow.vtp' not in filelist:
        for file in filelist:
            if 'mpa' in file.lower() and 'wall' not in file.lower():
                user = input(f'inflow vtp found! would you like to replace {file} with inflow.vtp? (y/n)')
                if user == 'y':
                    os.system('mv ' + file + ' ' + msh_surf_dir + '/inflow.vtp')
        
    
    # # make sure LPA and RPA are named correctly, and not named for the stent
    # if 'RPA.vtp' not in filelist:
    #     for file in filelist:
    #         if 'RPA' in file and os.path.basename(file)[4] != '0':
    #             user = input(f'RPA vtp found! would you like to replace {file} with RPA.vtp? (y/n)')
    #             if user == 'y':
    #                 os.system('mv ' + file + ' ' + msh_surf_dir + '/RPA.vtp')

    dup_files = []
    for file in filelist:
        if os.path.basename(file)[3] == '_' and os.path.basename(file)[-6] != '0':
            # this is a file which may have a duplicate
            dup_files.append(file)
        
    if len(dup_files) > 0:
        raise Exception(f'duplicate mesh surfaces detected in this directory! these will need to be cleaned up. \n List of potential duplicate surfaces: {dup_files}')


def scale_vtp_to_cm(vtp_file, scale_factor=0.1):
    '''
    scale a vtp file from mm to cm (multiply by 0.1) using vtkTransform
    '''

    print(f'scaling {vtp_file} by factor {scale_factor}...')
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()

    # get the area before scaling
    area = find_vtp_area(vtp_file)
    print(f'area before scaling: {area}')

    transform = vtk.vtkTransform()
    transform.Scale(scale_factor, scale_factor, scale_factor)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(reader.GetOutput())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(transform_filter.GetOutput())   
    writer.SetFileName(vtp_file)
    writer.Write()

    # get the area after scaling
    area = find_vtp_area(vtp_file)
    print(f'area after scaling: {area}')


def scale_msh_complete(msh_complete_dir, scale_factor=0.1):
    '''
    scale all vtp files in a mesh-complete directory to cm
    '''
    filelist_mshcomp = glob.glob(msh_complete_dir + '/*')
    filelist_mshcomp = [file for file in filelist_mshcomp if 'mesh-surfaces' not in file]
    
    filelist_mshsurf = glob.glob(os.path.join(msh_complete_dir, 'mesh-surfaces/*'))

    filelist = filelist_mshcomp + filelist_mshsurf

    for file in filelist:
        scale_vtp_to_cm(file, scale_factor=scale_factor)

if __name__ == '__main__':
    # setup a simulation dir from mesh
    
    msh_dir = '../svZeroDTrees-tests/cases/threed/LPA_RPA/mesh-complete-scaled'

    scale_msh_complete(msh_dir, scale_factor=10.0)




    
