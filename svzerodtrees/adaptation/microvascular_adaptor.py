import json
import csv
import copy
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

from ..io import ConfigHandler
from ..microvasculature import StructuredTree, TreeParameters
from ..simulation.simulation_directory import SimulationDirectory
from ..simulation.threedutils import vtp_info
from ..tune_bcs.clinical_targets import ClinicalTargets
from ..utils import *
from .integrator import run_adaptation
from .models import CWSSIMSAdaptation
from .setup import *


_DEFAULT_ALPHA = 0.9
_DEFAULT_BETA = 0.6
_DEFAULT_LRR = 10.0
_MIN_FLOW_EPS = 1e-8


def _optimize_tree_for_resistance(
    tree: StructuredTree,
    R_target: float,
    d_min: float,
    *,
    alpha: float = _DEFAULT_ALPHA,
    beta: float = _DEFAULT_BETA,
    lrr: float = _DEFAULT_LRR,
) -> Tuple[float, float]:
    """
    Tune the root diameter so that the structured tree matches the target resistance.
    Returns (optimized_root_diameter, matched_resistance).
    """
    if R_target <= 0.0:
        raise ValueError(f"Target resistance must be positive, got {R_target}")

    print(f'[{tree.name}] optimizing root diameter to match R={R_target:.6g}')

    lower = max(d_min * 1.02, 1e-4)
    upper = max(tree.diameter, lower * 2.0)
    upper_cap = max(upper, lower) * 1e3

    eval_counter = {"count": 0}

    def objective(diameter: float) -> float:
        tree.build(initial_d=float(diameter), d_min=d_min, alpha=alpha, beta=beta, lrr=lrr)
        current_R = tree.equivalent_resistance()
        eval_counter["count"] += 1
        print(f'[{tree.name}]   eval {eval_counter["count"]:02d}: d={float(diameter):.6f} cm → R={current_R:.6g}')
        return abs(current_R - R_target)

    # Expand upper bound until resistance is below target or limit reached
    for _ in range(12):
        tree.build(initial_d=upper, d_min=d_min, alpha=alpha, beta=beta, lrr=lrr)
        current_R = tree.equivalent_resistance()
        print(f'[{tree.name}]   bound check: d={upper:.6f} cm → R={current_R:.6g}')
        if current_R <= R_target or upper >= upper_cap:
            break
        upper *= 1.5

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-5})
    if not result.success:
        raise RuntimeError(f"Failed to match resistance {R_target} for tree {tree.name}: {result.message}")

    optimized_diameter = float(result.x)
    tree.build(initial_d=optimized_diameter, d_min=d_min, alpha=alpha, beta=beta, lrr=lrr)
    matched_resistance = tree.equivalent_resistance()
    print(f'[{tree.name}] optimized diameter={optimized_diameter:.6f} cm, matched R={matched_resistance:.6g}')
    return optimized_diameter, matched_resistance


def _apply_constant_wss_scaling(store, preop_flow: float, postop_flow: float, n_iter: int) -> None:
    """
    Uniformly scale diameters to enforce constant wall shear stress assumption.
    """
    if n_iter <= 0:
        return

    Q_old = float(preop_flow)
    Q_new = float(postop_flow)

    if not np.isfinite(Q_old) or abs(Q_old) < _MIN_FLOW_EPS:
        return

    ratio = abs(Q_new) / max(abs(Q_old), _MIN_FLOW_EPS)
    if ratio <= 0.0:
        return

    scale = ratio ** (1.0 / 3.0)
    total_scale = scale ** n_iter
    orig_dtype = np.asarray(store.d).dtype
    scaled = np.asarray(store.d, dtype=np.float64) * total_scale
    store.d = scaled.astype(orig_dtype, copy=False)

def _adapt_single_bc_worker(
        bc_name: str,
        R_preop: float,
        t,                     # 1D array-like
        preop_flow,            # 1D array-like
        postop_flow,           # 1D array-like
        n_iter: int,
        d_min: float
    ) -> Tuple[str, float, float]:
        """
        Top-level, picklable worker: builds & adapts a StructuredTree and returns
        (bc_name, R_adapt, optimized_root_diameter).
        """
        tree = StructuredTree(name=bc_name, time=t, simparams=None)

        optimized_root_diameter, _ = _optimize_tree_for_resistance(
            tree,
            R_target=float(R_preop),
            d_min=d_min,
        )

        _apply_constant_wss_scaling(tree.store, preop_flow=float(preop_flow), postop_flow=float(postop_flow), n_iter=n_iter)
        R_adapt = tree.equivalent_resistance()
        adapted_root_diameter = float(tree.store.d[0]) if tree.store.n_nodes() else optimized_root_diameter
        return bc_name, R_adapt, adapted_root_diameter

class MicrovascularAdaptor: 
    '''
    class for computing microvascular adaptation from a preop and postop result
    '''

    def __init__(self, 
                 preop_simdir: SimulationDirectory, 
                 postop_simdir: SimulationDirectory, 
                 adapted_simdir: SimulationDirectory,
                 clinical_targets: ClinicalTargets,
                 reduced_order_pa: json,
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
        self.reduced_order_pa_path = reduced_order_pa
            

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
        self.tree_params: Dict[str, TreeParameters] = {}
        self.lpa_tree = None
        self.rpa_tree = None

        def _load_tree_parameters(df, side: str) -> TreeParameters:
            subset = df[df["pa"].str.lower() == side.lower()]
            if subset.empty:
                raise ValueError(f"Could not find tree parameters for '{side}' in {tree_params}.")
            cols_lower = {col.lower() for col in subset.columns}
            if {"compliance model", "alpha", "beta"}.issubset(cols_lower):
                return TreeParameters.from_row(subset)
            else:
                raise ValueError(f"Tree parameters for '{side}' are missing required columns alpha, beta, compliance model in {tree_params}.")

        if self.bc_type == 'impedance':
            print("adaptating impedance boundary conditions")
            self.simple_pa = ConfigHandler.from_json(reduced_order_pa, is_pulmonary=True)
            opt_params = pd.read_csv(tree_params)
            self.tree_params['lpa'] = _load_tree_parameters(opt_params, 'lpa')
            self.tree_params['rpa'] = _load_tree_parameters(opt_params, 'rpa')
            # construct lpa and rpa trees
            self.lpa_tree, self.rpa_tree = self.construct_impedance_trees()
        elif self.bc_type == 'resistance':
            print("adapting resistance boundary conditions")
            if tree_params is not None:
                opt_params = pd.read_csv(tree_params)
                self.tree_params['lpa'] = _load_tree_parameters(opt_params, 'lpa')
                self.tree_params['rpa'] = _load_tree_parameters(opt_params, 'rpa')
        else:
            raise ValueError(f"bc_type {bc_type} not recognized, please use 'impedance' or 'resistance'")
            

        

        

    def adapt(self, fig_dir: str = None):
        '''
        adapt the microvasculature based on the constant wall shear stress assumption

        TODO: implement cwss-ims adaptation method here
        '''

        if fig_dir is not None:
            print("computing preop impedance! \n")
            Z_t_l_pre, time = self.lpa_tree.compute_olufsen_impedance(n_procs=24)
            Z_t_r_pre, time = self.rpa_tree.compute_olufsen_impedance(n_procs=24)
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
        Z_t_l_adapt, time = self.lpa_tree.compute_olufsen_impedance(n_procs=24)
        Z_t_r_adapt, time = self.rpa_tree.compute_olufsen_impedance(n_procs=24)

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

        lpa_params = self.tree_params.get('lpa')
        rpa_params = self.tree_params.get('rpa')
        if lpa_params is None or rpa_params is None:
            raise RuntimeError("Tree parameters must be loaded before constructing impedance trees.")

        time_array = self.preop_simdir.svzerod_3Dcoupling.bcs['INFLOW'].t
        
        lpa_tree = StructuredTree(
            name='LPA',
            time=time_array,
            simparams=None,
            compliance_model=lpa_params.compliance_model,
        )
        lpa_summary = lpa_params.summary() if hasattr(lpa_params.compliance_model, "description") else lpa_params.name
        print(f'building LPA tree with parameters: {lpa_summary}')
        lpa_tree.build(
            initial_d=lpa_params.diameter,
            d_min=lpa_params.d_min,
            lrr=lpa_params.lrr,
            alpha=lpa_params.alpha,
            beta=lpa_params.beta,
        )

        rpa_tree = StructuredTree(
            name='RPA',
            time=time_array,
            simparams=None,
            compliance_model=rpa_params.compliance_model,
        )
        rpa_summary = rpa_params.summary() if hasattr(rpa_params.compliance_model, "description") else rpa_params.name
        print(f'building RPA tree with parameters: {rpa_summary}')
        rpa_tree.build(
            initial_d=rpa_params.diameter,
            d_min=rpa_params.d_min,
            lrr=rpa_params.lrr,
            alpha=rpa_params.alpha,
            beta=rpa_params.beta,
        )

        return lpa_tree, rpa_tree
    
    def adapt_resistance(
        self,
        n_iter: int = 1,
        d_min: float = 0.01,
        coupler_path: str = 'svzerod_3Dcoupling.json',
        max_workers: int = None,
        parallel: bool = True,
    ) -> None:
        """
        Construct structured trees for preop/postop, adapt resistances based on flow changes,
        and write an updated 3D coupling file. Execution now proceeds sequentially for easier
        debugging; `parallel` and `max_workers` are accepted for backward compatibility but ignored.
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

        if parallel:
            print("Parallel execution disabled; proceeding sequentially for all outlets.")
        if max_workers is not None:
            print("max_workers ignored: adaptation runs sequentially.")

        # Sequential execution
        results: Dict[str, Tuple[float, float]] = {}
        for args in tasks:
            bc_name, R_adapt, adapted_root_diameter = _adapt_single_bc_worker(*args)
            results[bc_name] = (R_adapt, adapted_root_diameter)

        # 3) Apply results back to the (deep-copied) coupler in the main process
        for bc_name, (R_adapt, root_diameter) in results.items():
            R_preop = adapted_svzerod_coupler.bcs[bc_name].R
            if root_diameter is not None:
                print(f'updating resistance for {bc_name} from {R_preop} to {R_adapt} (root d = {root_diameter:.4f} cm)')
            else:
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
            if 'branch' in bc_name.lower():
                continue

            R_preop = preop_svzerod_coupler.bcs[bc_name].R
            print(f'building structured tree for resistance {R_preop}...')
            t = coupling_block.values['t']
            preop_flow = preop_svzerod_data.get_flow(coupling_block)
            postop_flow = postop_svzerod_data.get_flow(coupling_block)

            _, R_adapt, root_diameter = _adapt_single_bc_worker(
                bc_name,
                R_preop,
                t,
                preop_flow,
                postop_flow,
                n_iter,
                d_min,
            )

            if root_diameter is not None:
                print(f'updating resistance for {bc_name} from {R_preop} to {R_adapt} (root d = {root_diameter:.4f} cm)')
            else:
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
            tree.build(
                initial_d=params['initial_d'],
                d_min=params['d_min'],
                lrr=params.get('lrr', _DEFAULT_LRR),
                alpha=params.get('alpha'),
                beta=params.get('beta'),
                xi=params.get('xi'),
                eta_sym=params.get('eta_sym'),
            )

    def createImpedanceBCs(self):
        '''
        create the impedance boundary conditions for the adapted trees
        '''
        # if self.location == 'uniform':

        Z_t_l_adapt, time = self.lpa_tree.compute_olufsen_impedance(n_procs=24, tsteps=2000)
        Z_t_r_adapt, time = self.rpa_tree.compute_olufsen_impedance(n_procs=24, tsteps=2000)

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
        # TODO: use optimize nonlinear resistance instead as this is what I previously did
        # S_lpa_preop, S_rpa_preop = self.preop_simdir.optimize_nonlinear_resistance(self.reduced_order_pa_path, initial_guess=[100, 100])
        # S_lpa_postop, S_rpa_postop = self.postop_simdir.optimize_nonlinear_resistance(self.reduced_order_pa_path, initial_guess=[100, 100])
        S_lpa_preop, S_rpa_preop = self.preop_simdir.compute_pressure_drop(steady=False)
        S_lpa_postop, S_rpa_postop = self.postop_simdir.compute_pressure_drop(steady=False)

        # fixed values for testing
        # S_lpa_preop = 225.59599016730704
        # S_rpa_preop = 2174.5744993285525
        # S_lpa_postop = 133.44002892612093
        # S_rpa_postop = 1639.2238005469198


        postop_pa = copy.deepcopy(preop_pa)

        # rescale postop stenosis coefficient
        # postop_pa.lpa.stenosis_coefficient *= S_lpa_postop / S_lpa_preop
        # postop_pa.vessel_map[2].stenosis_coefficient *= S_lpa_postop / S_lpa_preop
        # postop_pa.rpa.stenosis_coefficient *= S_rpa_postop / S_rpa_preop
        # postop_pa.vessel_map[4].stenosis_coefficient *= S_rpa_postop / S_rpa_preop

        # fixed values for testing
        postop_pa.lpa.stenosis_coefficient = 19.7
        postop_pa.vessel_map[2].stenosis_coefficient = 19.7
        postop_pa.rpa.stenosis_coefficient = 7.6
        postop_pa.vessel_map[4].stenosis_coefficient = 7.6

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
        result, flow_log, sol, postop_pa, hists = run_adaptation(preop_pa, postop_pa, CWSSIMSAdaptation, K_arr)

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
