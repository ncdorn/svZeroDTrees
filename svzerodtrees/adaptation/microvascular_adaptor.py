import json
import csv
from ..simulation.simulation_directory import SimulationDirectory
from ..io import ConfigHandler
import matplotlib.pyplot as plt
import os
from ..microvasculature import StructuredTree
from ..utils import *
from ..simulation.threedutils import vtp_info
from ..tune_bcs import ClinicalTargets
from .setup import *
from .integrator import run_adaptation
from .models import CWSSIMSAdaptation
class MicrovascularAdaptor: 
    '''
    class for computing microvascular adaptation from a preop and postop result
    '''

    def __init__(self, 
                 preop_simdir: SimulationDirectory, 
                 postop_simdir: SimulationDirectory, 
                 adapted_simdir: SimulationDirectory,
                 reduced_order_pa: json,
                 tree_params: csv, 
                 clinical_targets: ClinicalTargets,
                 method: str = 'cwss', 
                 location: str = 'uniform',
                 n_iter: int = 100,
                 convert_to_cm: bool = False):
        '''
        initialize the MicrovascularAdaptation class
        
        :param preop_simdir: SimulationDirectory opbject for the preoperative simulation
        :param postop_simdir: SimulationDirectory object for the postoperative simulation
        :param adapted_simdir: SimulationDirectory object for the adapted simulation
        :param tree_params: csv file optimized_params.csv
        :param method: adaptation method, default is 'cwss'. options ['cwss', 'wss-ims']
        :param location: location of the adaptation, default is 'uniform'
        '''
        self.preop_simdir = preop_simdir
        self.postop_simdir = postop_simdir
        self.adapted_simdir = adapted_simdir

        self.simple_pa = ConfigHandler.from_json(reduced_order_pa, is_pulmonary=True)

        # grab tree params from csv, of form [k1, k2, k3, lrr, diameter]
        opt_params = pd.read_csv(os.path.join(tree_params))
        self.tree_params = {
            'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], opt_params['diameter'][opt_params.pa=='lpa'].values[0]],
            'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], opt_params['diameter'][opt_params.pa=='rpa'].values[0]]
        }

        self.clinical_targets = clinical_targets

        if method not in ['cwss', 'wss-ims']:
            raise ValueError(f"adaptation method {method} not recognized, please use 'cwss' or 'wss-ims'")
        else:
            print(f"using adaptation method {method}")
        self.method = method
        self.location = location

        self.convert_to_cm = convert_to_cm

        # construct lpa and rpa trees
        self.lpa_tree, self.rpa_tree = self.constructTrees()

    def adapt(self, fig_dir: str = None):
        '''
        adapt the microvasculature based on the constant wall shear stress assumption

        TODO: implement cwss-ims adaptation method here
        '''

        if fig_dir is not None:
            print("computing preop impedance! \n")
            Z_t_l_pre, time = self.lpa_tree.compute_olufsen_impedance(self.tree_params['lpa'][0], self.tree_params['lpa'][1], self.tree_params['lpa'][2], n_procs=24)
            Z_t_r_pre, time = self.rpa_tree.compute_olufsen_impedance(self.tree_params['rpa'][0], self.tree_params['rpa'][1], self.tree_params['rpa'][2],n_procs=24)
            self.lpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'lpa_stiffness_preop.png'))
            self.rpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'rpa_stiffness_preop.png'))

        sum_flows = lambda d: tuple(map(lambda x: sum(x.values()), d.flow_split(get_mean=True)))

        preop_lpa_flow, preop_rpa_flow = sum_flows(self.preop_simdir)
        postop_lpa_flow, postop_rpa_flow = sum_flows(self.postop_simdir)

        if self.method == 'cwss':
            self.lpa_tree.adapt_constant_wss(Q=preop_lpa_flow, Q_new=postop_lpa_flow, n_itern=self.n_iter)
            self.rpa_tree.adapt_constant_wss(Q=preop_rpa_flow, Q_new=postop_rpa_flow, n_iter=self.n_iter)
        elif self.method == 'wss-ims':
            self.lpa_tree.adapt_wss_ims(Q=preop_lpa_flow, Q_new=postop_lpa_flow, n_iter=self.n_iter)
            self.rpa_tree.adapt_wss_ims(Q=preop_rpa_flow, Q_new=postop_rpa_flow, n_iter=self.n_iter)

        print("computing adapted impedance! \n")
        Z_t_l_adapt, time = self.lpa_tree.compute_olufsen_impedance(self.tree_params['lpa'][0], self.tree_params['lpa'][1], self.tree_params['lpa'][2], n_procs=24)
        Z_t_r_adapt, time = self.rpa_tree.compute_olufsen_impedance(self.tree_params['rpa'][0], self.tree_params['rpa'][1], self.tree_params['rpa'][2],n_procs=24)

        if fig_dir is not None:
            # plot the preop and postop impedance
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].plot(time, Z_t_l_pre, label='preop')
            ax[0].plot(time, Z_t_l_adapt, label='postop')
            ax[0].set_title('LPA impedance')
            ax[0].legend()
            ax[0].set_xlabel('time [s]')
            ax[0].set_ylabel('Z')
            ax[0].set_yscale('log')
            ax[1].plot(time, Z_t_r_pre, label='preop')
            ax[1].plot(time, Z_t_r_adapt, label='postop')
            ax[1].set_title('RPA impedance')
            ax[1].legend()
            ax[1].set_xlabel('time [s]')
            ax[1].set_ylabel('Z')
            ax[1].set_yscale('log')
            plt.savefig(os.path.join(fig_dir, 'impedance_adaptation.png'))
        
            self.lpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'lpa_stiffness_adapted.png'))
            self.rpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'rpa_stiffness_adapted.png'))

        # distribute the impedance to lpa and rpa specifically
        self.createImpedanceBCs()
        
        self.adapted_simdir.svzerod_3Dcoupling = self.postop_simdir.svzerod_3Dcoupling
        # change path
        self.adapted_simdir.svzerod_3Dcoupling.path = os.path.join(self.adapted_simdir.path, 'svzerod_3Dcoupling.json')
        # update the config with the adapted trees
        self.adapted_simdir.svzerod_3Dcoupling.tree_params[self.lpa_tree.name] = self.lpa_tree.to_dict()
        self.adapted_simdir.svzerod_3Dcoupling.tree_params[self.rpa_tree.name] = self.rpa_tree.to_dict()

        print("saving adapted config to " + self.adapted_simdir.svzerod_3Dcoupling.path)
        self.adapted_simdir.svzerod_3Dcoupling.to_json(self.adapted_simdir.svzerod_3Dcoupling.path)

    def constructTrees(self):
        '''
        construct the trees for the preop and postop simulations
        '''

        k1_l, k2_l, k3_l, lrr_l, d_l = self.tree_params['lpa']
        k1_r, k2_r, k3_r, lrr_r, d_r = self.tree_params['rpa']

        time_array = self.preop_simdir.svzerod_3Dcoupling.bcs['INFLOW'].t
        
        lpa_tree = StructuredTree(name='LPA', time=time_array, simparams=None)
        print(f'building LPA tree with lpa parameters: {self.tree_params["lpa"]}')
        lpa_tree.build_tree(initial_d=d_l, d_min=0.01, lrr=lrr_l)

        rpa_tree = StructuredTree(name='RPA', time=time_array, simparams=None)
        print(f'building RPA tree with rpa parameters: {self.tree_params["rpa"]}')
        rpa_tree.build_tree(initial_d=d_r, d_min=0.01, lrr=lrr_r)

        return lpa_tree, rpa_tree

    def constructTreesFromConfig(self):
        
        time_array = self.preop_simdir.svzerod_3Dcoupling.bcs['INFLOW'].t

        for name, params in self.preop_simdir.svzerod_3Dcoupling.tree_params.items():
            tree = StructuredTree(name=name, time=time_array, simparams=None)
            print(f'building {name} tree with parameters: {params}')
            tree.build_tree(initial_d=params['initial_d'], d_min=params['d_min'], lrr=params['lrr'])

    def createImpedanceBCs(self):
        '''
        create the impedance boundary conditions for the adapted trees
        '''
        # if self.location == 'uniform':
        
        cap_info = vtp_info(self.postop_simdir.mesh_complete.mesh_surfaces_dir, convert_to_cm=self.convert_to_cm, pulmonary=False)

        outlet_bc_names = [name for name, bc in self.postop_simdir.svzerod_3Dcoupling.bcs.items() if 'inflow' not in bc.name.lower()]

        # assumed that cap and boundary condition orders match, TODO: UPDATE THIS TO BE USED with SIMULATIONDIRECTORY CLASS
        if len(outlet_bc_names) != len(cap_info):
            print('number of outlet boundary conditions does not match number of cap surfaces, automatically assigning bc names...')
            for i, name in enumerate(outlet_bc_names):
                # delete the unused bcs
                del self.postop_simdir.svzerod_3Dcoupling.bcs[name]
            outlet_bc_names = [f'IMPEDANCE_{i}' for i in range(len(cap_info))]
        
        cap_to_bc = {list(cap_info.keys())[i]: outlet_bc_names[i] for i in range(len(outlet_bc_names))}

        for idx, (cap_name, area) in enumerate(cap_info.items()):
                    print(f'generating tree {idx + 1} of {len(cap_info)} for cap {cap_name}...')
                    if 'lpa' in cap_name.lower():
                        self.postop_simdir.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.lpa_tree.create_impedance_bc(cap_to_bc[cap_name], 0, self.clinicalTargets.wedge_p * 1333.2)
                    elif 'rpa' in cap_name.lower():
                        self.postop_simdir.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.rpa_tree.create_impedance_bc(cap_to_bc[cap_name], 1, self.clinicalTargets.wedge_p * 1333.2)
                    else:
                        raise ValueError('cap name not recognized')
                    
    
    def adapt_cwss_ims(self, K_arr, fig_dir: str = None):
        # TODO: Implement later as here we are using the 3D model to adapt the trees

        preop_pa = self.simple_pa

        # compute difference in pressure drop to get postop nonlinear resistance
        S_lpa_preop, S_rpa_preop = self.preop_simdir.compute_pressure_drop(steady=False)
        S_lpa_postop, S_rpa_postop = self.postop_simdir.compute_pressure_drop(steady=False)

        postop_pa = copy.deepcopy(preop_pa)

        # rescale postop stenosis coefficient
        postop_pa.lpa.stenosis_coefficient *= S_lpa_postop / S_lpa_preop
        postop_pa.rpa.stenosis_coefficient *= S_rpa_postop / S_rpa_preop

        # save pa jsons
        preop_pa.to_json(os.path.join(self.preop_simdir.path, 'preop_simple_pa.json'))
        postop_pa.to_json(os.path.join(self.postop_simdir.path, 'postop_simple_pa.json'))

        # initialize preop and postop PAConfig objects
        preop_pa, postop_pa = initialize_from_paths(
            os.path.join(self.preop_simdir.path, 'preop_simple_pa.json'),
            os.path.join(self.postop_simdir.path, 'postop_simple_pa.json'),
            os.path.dirname(self.preop_simdir.path) + '/optimized_params.csv',
            os.path.dirname(self.preop_simdir.path) + '/clinical_targets.csv'
        )
        # run adaptation
        result, flow_log, sol, postop_pa = run_adaptation(preop_pa, postop_pa, CWSSIMSAdaptation, K_arr)

        print(f"Adaptation result: {result}")

        # assign BCs from postop_pa
        self.lpa_tree = postop_pa.lpa_tree
        self.rpa_tree = postop_pa.rpa_tree

        # distribute the impedance to lpa and rpa specifically
        self.createImpedanceBCs()
        
        self.adapted_simdir.svzerod_3Dcoupling = self.postop_simdir.svzerod_3Dcoupling
        # change path
        self.adapted_simdir.svzerod_3Dcoupling.path = os.path.join(self.adapted_simdir.path, 'svzerod_3Dcoupling.json')
        # update the config with the adapted trees
        self.adapted_simdir.svzerod_3Dcoupling.tree_params[self.lpa_tree.name] = self.lpa_tree.to_dict()
        self.adapted_simdir.svzerod_3Dcoupling.tree_params[self.rpa_tree.name] = self.rpa_tree.to_dict()

        print("saving adapted config to " + self.adapted_simdir.svzerod_3Dcoupling.path)
        self.adapted_simdir.svzerod_3Dcoupling.to_json(self.adapted_simdir.svzerod_3Dcoupling.path)




        pass