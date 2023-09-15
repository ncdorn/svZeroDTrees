import vtk
import glob

def find_vtp_area(infile):
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