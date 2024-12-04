import csv
import glob
from svzerodtrees.threedutils import *
from svzerodtrees.preop import optimize_pa_bcs, optimize_outlet_bcs, construct_impedance_trees
from svzerodtrees.config_handler import ConfigHandler


def simdir_from_mesh(mesh_complete, threed_dir, zerod_dir, trees='resistance'):
    '''
    create a simulation directory from a mesh-complete directory
    
    :param mesh_complete: path to the mesh-complete directory
    :param inflow_file: path to the simulation directory to create
    :param threed_dir: path to threed simulation directory
    :param zerod_dir: path to zerod simulation directory with zerod model form simvascular and clinical targets
    :param trees: type of trees build (resistance or impedance)'''


    mesh_surfaces = os.path.join(mesh_complete, 'mesh-surfaces')
    # get the path to clinical targets and zerod config within the zerod directory
    clinical_targets = os.path.join(zerod_dir, 'clinical_targets.csv')
    zerod_config = glob.glob(os.path.join(zerod_dir, '*.json'))[0]

    # create the optimized config handler to get bcs from 
    if trees == 'resistance':
        print('optimizing boundary conditions with 0d model... \n\n')
        # config_handler, result_handler = optimize_outlet_bcs(zerod_config, clinical_targets)
        config_handler, result_handler, pa_config = optimize_pa_bcs(zerod_config, mesh_surfaces, clinical_targets)

        config_handler.to_json(os.path.join(zerod_dir, f'optimized_{os.path.basename(zerod_config)}'))

        setup_simdir_from_mesh(threed_dir, f'optimized_{os.path.basename(zerod_config)}')

    elif trees == 'impedance':
        config_handler = ConfigHandler.from_json(zerod_config)

        construct_impedance_trees(config_handler, mesh_surfaces, clinical_targets, d_min=0.05)

        config_handler.to_json(os.path.join(zerod_dir, f'impedance_{os.path.basename(zerod_config)}'))

        setup_simdir_from_mesh(threed_dir, f'impedance_{os.path.basename(zerod_config)}')



if __name__ == '__main__':
    preop_mesh = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/AS2_prestent/Meshes/1.6M_elements'
    # postop_mesh = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/AS2_stent/Meshes/5mm-stent_extv_1.7M_elements'

    os.chdir('/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/threed_automate_test')

    simdir_from_mesh(preop_mesh, 'preop', 'zerod')
    # simdir_from_mesh(postop_mesh, 'inflow.flow', 'postop', 'zerod')



