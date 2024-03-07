import csv
import glob
from svzerodtrees.threedutils import *
from svzerodtrees.preop import optimize_pa_bcs, optimize_outlet_bcs
from svzerodtrees._config_handler import ConfigHandler



def simdir_from_mesh(mesh_complete, inflow_file, threed_dir, zerod_dir):
    '''
    create a simulation directory from a mesh-complete directory
    
    :param mesh_complete: path to the mesh-complete directory
    :param simname: path to the simulation directory to create'''

    # create the simulation directory
    if not os.path.exists(threed_dir):
        os.makedirs(threed_dir)

    if os.path.exists(threed_dir + '/mesh-complete'):
        print('mesh-complete directory already exists in the simulation directory')
    else:
        os.system('cp -rp ' + mesh_complete + ' ' + threed_dir + '/mesh-complete')
    os.system('cp ' + inflow_file + ' ' + threed_dir)

    inflow_df = pd.read_csv(inflow_file, names=['time', 'flow'], sep='\s+')
    period = inflow_df.time.max()

    os.chdir(threed_dir)
    # write the svpre file
    inlet_idx, outlet_idxs = write_svpre_file(mesh_complete, os.path.basename(threed_dir), period)

    print('simulation directory created at ' + threed_dir + '\n\n')

    mesh_surfaces = os.path.join('mesh-complete', 'mesh-surfaces')
    # get the path to clinical targets and zerod config within the zerod directory
    clinical_targets = os.path.join('../', zerod_dir, 'clinical_targets.csv')
    zerod_config = glob.glob(os.path.join('../', zerod_dir, '*.json'))[0]

    # create the optimized config handler to get bcs from 
    print('optimizing boudnary conditions with 0d model... \n\n')
    # config_handler, result_handler = optimize_outlet_bcs(zerod_config, clinical_targets)
    config_handler, result_handler, pa_config = optimize_pa_bcs(zerod_config, mesh_surfaces, clinical_targets)

    print('writing simulation files... \n\n')
    # generate the 3d coupling config from optimized bcs
    config_handler.generate_threed_coupler('.')
    # write the svzerod interface file
    write_svzerod_interface('svzerod_3Dcoupling.json', outlet_idxs)
    # write the solver.inp
    write_solver_inp(outlet_idxs, period, 2)
    # write the numstart file
    write_numstart()
    # write svsolver runscript
    write_svsolver_runscript(os.path.basename(threed_dir), hours=6, nodes=4, procs_per_node=24)

    # leave this directory
    os.chdir('../')
    print('setup simulation directory ' + threed_dir)





if __name__ == '__main__':
    preop_mesh = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/AS2_prestent/Meshes/1.6M_elements'
    postop_mesh = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/AS2_stent/Meshes/5mm-stent_prox_1.7M_elements'

    os.chdir('/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/AS2')

    simdir_from_mesh(preop_mesh, 'inflow.flow', 'preop', 'zerod')
    simdir_from_mesh(postop_mesh, 'inflow.flow', 'postop', 'zerod')



