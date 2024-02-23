import vtk
import glob
import pandas as pd

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
    simulation_name = simulation_dir.split('/')[-1]
    surface_id_map = {}
    with open(simulation_name + '.svpre', 'r') as ff:
        for line in ff:
            line = line.strip()
            if line.startswith('set_surface_id'):
                line_objs = line.split(' ')
                vtp_file = line_objs[1]
                surface_id = line_objs[2]
                surface_id_map[surface_id] = vtp_file
    
    # get a map of sruface id's to coupling blocks
    coupling_map = {}
    reading_coupling_blocks=False
    with open('svZeroD_interface.dat', 'r') as ff:
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