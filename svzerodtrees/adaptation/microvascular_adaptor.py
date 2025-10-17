import json
import csv

from ..io import ConfigHandler
import matplotlib.pyplot as plt
import os
from ..microvasculature import StructuredTree
from ..utils import *
from ..simulation.threedutils import vtp_info
from ..simulation.simulation_directory import SimulationDirectory
from ..tune_bcs.clinical_targets import ClinicalTargets
from .setup import *
from .integrator import run_adaptation
from .models import CWSSIMSAdaptation

# ---- at module top-level (same .py file), not inside a class ----
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import copy
from typing import Tuple

def _adapt_single_bc_worker(
        bc_name: str,
        R_preop: float,
        t,                     # 1D array-like
        preop_flow,            # 1D array-like
        postop_flow,           # 1D array-like
        n_iter: int,
        d_min: float
    ) -> Tuple[str, float]:
        """
        Top-level, picklable worker: builds & adapts a StructuredTree and returns (bc_name, R_adapt).
        """
        # Import here to avoid pickling large modules on executor startup if not needed elsewhere
        # from .structured_tree import StructuredTree  # adjust import to your project structure

        tree = StructuredTree(name=bc_name, time=t, simparams=None)
        tree.optimize_tree_diameter(resistance=R_preop, d_min=d_min)
        tree.adapt_constant_wss(preop_flow, postop_flow, n_iter=n_iter)
        R_adapt = tree.root.R_eq
        return bc_name, R_adapt

class MicrovascularAdaptor: 
    '''
    class for computing microvascular adaptation from a preop and postop result
    '''

    def __init__(self, 
                 preop_simdir: SimulationDirectory, 
                 postop_simdir: SimulationDirectory, 
                 adapted_simdir: SimulationDirectory,
                 clinical_targets: ClinicalTargets,
                 reduced_order_pa: json = None,
                 tree_params: csv = None, 
                 method: str = 'cwss', 
                 location: str = 'uniform',
                 bc_type: str = 'impedance',
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
            

        # grab tree params from csv, of form [k1, k2, k3, lrr, diameter]

        self.convert_to_cm = convert_to_cm

        self.clinical_targets = clinical_targets

        if method not in ['cwss', 'wss-ims']:
            raise ValueError(f"adaptation method {method} not recognized, please use 'cwss' or 'wss-ims'")
        else:
            print(f"using adaptation method {method}")
        self.method = method
        self.location = location
        self.bc_type = bc_type

        if self.bc_type == 'impedance':
            print("adaptating impedance boundary conditions")
            self.simple_pa = ConfigHandler.from_json(reduced_order_pa, is_pulmonary=True)
            opt_params = pd.read_csv(os.path.join(tree_params))
            self.tree_params = {
                'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], opt_params['diameter'][opt_params.pa=='lpa'].values[0]],
                'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], opt_params['diameter'][opt_params.pa=='rpa'].values[0]]
            }
            # construct lpa and rpa trees
            self.lpa_tree, self.rpa_tree = self.construct_impedance_trees()
        elif self.bc_type == 'resistance':
            print("adapting resistance boundary conditions")
            if tree_params is not None:
                opt_params = pd.read_csv(os.path.join(tree_params))
                self.tree_params = {
                    'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], opt_params['diameter'][opt_params.pa=='lpa'].values[0]],
                    'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], opt_params['diameter'][opt_params.pa=='rpa'].values[0]]
                }
        else:
            raise ValueError(f"bc_type {bc_type} not recognized, please use 'impedance' or 'resistance'")
            

        

        

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

    def construct_impedance_trees(self):
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
    
    def adapt_resistance(
        self,
        n_iter: int = 1,
        d_min: float = 0.01,
        coupler_path: str = 'svzerod_3Dcoupling.json',
        max_workers: int | None = None,
        parallel: bool = True,
    ) -> None:
        """
        Construct structured trees for preop/postop, adapt resistances based on flow changes,
        and write an updated 3D coupling file.

        Parameters
        ----------
        n_iter : int
            Iterations for tree adaptation.
        d_min : float
            Minimum terminal diameter passed into StructuredTree.optimize_tree_diameter.
        coupler_path : str
            Where to write the adapted coupling JSON (relative to adapted_simdir.path).
        max_workers : int | None
            Number of worker processes. Defaults to os.cpu_count() if None.
        parallel : bool
            Toggle parallel execution (useful for debugging).
        """
        preop_svzerod_coupler = self.preop_simdir.svzerod_3Dcoupling
        preop_svzerod_data = self.preop_simdir.svzerod_data
        postop_svzerod_data = self.postop_simdir.svzerod_data

        adapted_svzerod_coupler = copy.deepcopy(preop_svzerod_coupler)

        # 1) Gather tasks (avoid sending big objects—extract only arrays and scalars)
        tasks = []
        for bc_name, coupling_block in preop_svzerod_coupler.coupling_blocks.items():
            if 'branch' in bc_name.lower():
                continue

            R_preop = preop_svzerod_coupler.bcs[bc_name].R
            t = coupling_block.values['t']  # assume 1D array-like
            # Extract flows now (arrays), so workers don't need access to the data managers
            preop_flow = preop_svzerod_data.get_flow(coupling_block)
            postop_flow = postop_svzerod_data.get_flow(coupling_block)

            # Optional: small log message here if you like
            print(f'building structured tree for resistance {R_preop} at BC "{bc_name}"...')

            tasks.append((bc_name, R_preop, t, preop_flow, postop_flow, n_iter, d_min))

        # 2) Execute (parallel or sequential)
        results: dict[str, float] = {}

        if parallel and len(tasks) > 1:
            if max_workers is None:
                max_workers = os.cpu_count() or 1

            # Mac/Windows note:
            # Ensure this code runs under if __name__ == "__main__": when called from a script to avoid spawn issues.
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                future_map = {
                    ex.submit(_adapt_single_bc_worker, *args): args[0]  # bc_name
                    for args in tasks
                }
                for fut in as_completed(future_map):
                    bc = future_map[fut]
                    try:
                        bc_name, R_adapt = fut.result()
                    except Exception as e:
                        # Fail fast with context; you could also choose to continue on error
                        raise RuntimeError(f'Adaptation failed for BC "{bc}": {e}') from e
                    results[bc_name] = R_adapt
        else:
            # Sequential fallback (useful for debugging/pdb)
            for args in tasks:
                bc_name, R_adapt = _adapt_single_bc_worker(*args)
                results[bc_name] = R_adapt

        # 3) Apply results back to the (deep-copied) coupler in the main process
        for bc_name, R_adapt in results.items():
            R_preop = adapted_svzerod_coupler.bcs[bc_name].R
            print(f'updating resistance for {bc_name} from {R_preop} to {R_adapt}')
            adapted_svzerod_coupler.bcs[bc_name].R = R_adapt

        # 4) Persist to disk
        self.adapted_simdir.svzerod_3Dcoupling = adapted_svzerod_coupler
        self.adapted_simdir.svzerod_3Dcoupling.path = os.path.join(self.adapted_simdir.path, coupler_path)
        print("saving adapted config to " + self.adapted_simdir.svzerod_3Dcoupling.path)
        self.adapted_simdir.svzerod_3Dcoupling.to_json(self.adapted_simdir.svzerod_3Dcoupling.path)
    
    def adapt_resistance_nonparallel(self, n_iter: int = 1, d_min=0.01, coupler_path='svzerod_3Dcoupling.json'):
        '''
        construct the trees for the preop and postop simulations from resistance values

        '''

        preop_svzerod_coupler = self.preop_simdir.svzerod_3Dcoupling
        preop_svzerod_data = self.preop_simdir.svzerod_data
        postop_svzerod_data = self.postop_simdir.svzerod_data
        adapted_svzerod_coupler = copy.deepcopy(preop_svzerod_coupler)

        for bc_name, coupling_block in preop_svzerod_coupler.coupling_blocks.items():
            if 'branch' not in bc_name.lower():
                # get the resistance from the preop simulation
                R_preop  = preop_svzerod_coupler.bcs[bc_name].R
                # build tree for preop resistance
                print(f'building structured tree for resistance {R_preop}...')
                tree = StructuredTree(name=bc_name, time=coupling_block.values['t'], simparams=None)
                tree.optimize_tree_diameter(resistance=R_preop, d_min=d_min)
                # get the flow from the preop, postop simulation
                preop_flow = preop_svzerod_data.get_flow(coupling_block)
                postop_flow = postop_svzerod_data.get_flow(coupling_block)

                # adapt the tree based on the flow change
                tree.adapt_constant_wss(preop_flow, postop_flow, n_iter=n_iter)

                # update the bc with the new resistance
                R_adapt = tree.root.R_eq
                print(f'updating resistance for {bc_name} from {R_preop} to {R_adapt}')
                adapted_svzerod_coupler.bcs[bc_name].R = R_adapt

        
        self.adapted_simdir.svzerod_3Dcoupling = adapted_svzerod_coupler
        # change path
        self.adapted_simdir.svzerod_3Dcoupling.path = os.path.join(self.adapted_simdir.path, coupler_path)
        print("saving adapted config to " + self.adapted_simdir.svzerod_3Dcoupling.path)
        self.adapted_simdir.svzerod_3Dcoupling.to_json(self.adapted_simdir.svzerod_3Dcoupling.path)


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

        Z_t_l_adapt, time = self.lpa_tree.compute_olufsen_impedance(self.tree_params['lpa'][0], self.tree_params['lpa'][1], self.tree_params['lpa'][2], n_procs=24, tsteps=2000)
        Z_t_r_adapt, time = self.rpa_tree.compute_olufsen_impedance(self.tree_params['rpa'][0], self.tree_params['rpa'][1], self.tree_params['rpa'][2], n_procs=24, tsteps=2000)

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
                        self.postop_simdir.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.lpa_tree.create_impedance_bc(cap_to_bc[cap_name], 0, self.clinical_targets.wedge_p * 1333.2)
                    elif 'rpa' in cap_name.lower():
                        self.postop_simdir.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.rpa_tree.create_impedance_bc(cap_to_bc[cap_name], 1, self.clinical_targets.wedge_p * 1333.2)
                    else:
                        raise ValueError('cap name not recognized')
                    
    
    def adapt_cwss_ims(self, K_arr, fig_dir: str = None):

        preop_pa = self.simple_pa

        # preop_pa.inflows["INFLOW"].q = [q / (self.postop_simdir.mesh_complete.n_outlets // 2) for q in preop_pa.inflows["INFLOW"].q]

        # compute difference in pressure drop to get postop nonlinear resistance
        S_lpa_preop, S_rpa_preop = self.preop_simdir.compute_pressure_drop(steady=False)
        S_lpa_postop, S_rpa_postop = self.postop_simdir.compute_pressure_drop(steady=False)

        postop_pa = copy.deepcopy(preop_pa)

        # rescale postop stenosis coefficient
        postop_pa.lpa.stenosis_coefficient *= S_lpa_postop / S_lpa_preop
        postop_pa.vessel_map[2].stenosis_coefficient *= S_lpa_postop / S_lpa_preop
        postop_pa.rpa.stenosis_coefficient *= S_rpa_postop / S_rpa_preop
        postop_pa.vessel_map[4].stenosis_coefficient *= S_rpa_postop / S_rpa_preop

        print("rescaled postop lpa stenosis coefficient to " + str(postop_pa.lpa.stenosis_coefficient))
        print("rescaled postop rpa stenosis coefficient to " + str(postop_pa.rpa.stenosis_coefficient))

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

        # rescale inflow

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

