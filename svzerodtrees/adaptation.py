from svzerodtrees.utils import *
import copy
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees.result_handler import ResultHandler
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.simulation_directory import *
import numpy as np


def adapt_pries_secomb(config_handler: ConfigHandler, result_handler: ResultHandler, log_file: str = None, tol: float = .01):
    '''
    adapt structured tree microvasculature model based on Pries et al. 1998

    :param postop_config: config dict from the stenosis repair
    :param trees: list of StructuredTree instances, corresponding to the outlets of the 0D model
    :param preop_result: preoperative result array
    :param postop_result: postoperative result array, calculated after the stenosis repair
    :param log_file: path to log file, for writing important messages for debugging purposes

    :return: config dict post-adaptation, flow result post-adaptation, list of StructuredTree instances
    '''
    # get the preop and postop outlet flowrate and pressure
    preop_q = get_outlet_data(config_handler.config, result_handler.results['preop'], 'flow_out', steady=True)
    postop_q = get_outlet_data(config_handler.config, result_handler.results['postop'], 'flow_out', steady=True)
    postop_p = get_outlet_data(config_handler.config, result_handler.results['postop'], 'pressure_out', steady=True)

    # initialize R_old and R_new for pre- and post-adaptation comparison
    R_old = [tree.root.R_eq for tree in config_handler.trees]
    R_adapt = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on Pries and Secomb model **")

    # loop through the vessels and create StructuredTree instances at the outlets, from the pre-adaptation tree instances
    for vessel in config_handler.vessel_map.values():
        if vessel.bc is not None:
            if "outlet" in vessel.bc:
                config_handler.trees[outlet_idx].block_dict["P_in"] = [np.mean(postop_p[outlet_idx]), ] * 2
                config_handler.trees[outlet_idx].block_dict["Q_in"] =[np.mean(postop_q[outlet_idx]), ] * 2

                config_handler.trees[outlet_idx].pries_n_secomb.integrate()

                # append the adapted equivalent resistance to the list of adapted resistances
                R_adapt.append(config_handler.trees[outlet_idx].root.R_eq)

                # write results to log file for debugging
                write_to_log(log_file, "** adaptation results for " + str(config_handler.trees[outlet_idx].name) + " **")
                write_to_log(log_file, "    R_new = " + str(config_handler.trees[outlet_idx].root.R_eq) + ", R_old = " + str(R_old[outlet_idx]))
                write_to_log(log_file, "    The change in resistance is " + str(config_handler.trees[outlet_idx].root.R_eq - R_old[outlet_idx]))

                outlet_idx += 1

    # write adapted tree R_eq to the adapted_config
    write_resistances(config_handler.config, R_adapt)

    # get the adapted flow and pressure result
    adapted_result = run_svzerodplus(config_handler.config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')

    write_to_log(log_file, 'pries and secomb adaptation completed for all trees. R_old = ' + str(R_old) + ' R_new = ' + str(R_adapt))


def adapt_constant_wss(config_handler: ConfigHandler, result_handler: ResultHandler, log_file: str = None):
    '''
    adapt structured trees based on the constant wall shear stress assumption

    :param postop_config: config dict from the stenosis repair, with StructuredTree instances at the outlets
    :param trees: list of StructuredTree instances, corresponding to the outlets of the 0D model
    :param preop_result: preoperative result array
    :param postop_result: postoperative result array, calculated after the stenosis repair
    :param log_file: path to log file, for writing important messages for debugging purposes

    :return: config dict post-adaptation, flow result post-adaptation, list of StructuredTree instances
    '''

    # get the preop and postop outlet flowrates
    preop_q = get_outlet_data(config_handler.config, result_handler.results['preop'], 'flow_out', steady=True)
    postop_q = get_outlet_data(config_handler.config, result_handler.results['postop'], 'flow_out', steady=True)

    # intialize the adpated resistance list
    R_adapt = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on constant wall shear stress assumption **")


    for vessel in config_handler.vessel_map.values():
        if vessel.bc is not None:
            if "outlet" in vessel.bc:

                R_old, R_new = config_handler.trees[outlet_idx].adapt_constant_wss(Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx])

                # append the adapted equivalent resistance to the list of adapted resistances
                R_adapt.append(R_new)

                # write results to log file for debugging
                write_to_log(log_file, "** adaptation results for " + str(config_handler.trees[outlet_idx].name) + " **")
                write_to_log(log_file, "    R_new = " + str(config_handler.trees[outlet_idx].root.R_eq) + ", R_old = " + str(R_old))
                write_to_log(log_file, "    The change in resistance is " + str(config_handler.trees[outlet_idx].root.R_eq - R_old))

                config_handler.bcs[vessel.bc["outlet"]].R = R_new

                outlet_idx += 1

    # write the adapted resistances to the config resistance boundary conditions
    # config_handler.to_json('experiments/AS1_no_repair/postop_config.json')
    write_resistances(config_handler.config, R_adapt)
    # config_handler.to_json('experiments/AS1_no_repair/adapted_config_no_trees.json')

    config_handler.simulate(result_handler, 'adapted')


def adapt_constant_wss_threed_OLD(config_handler: ConfigHandler, preop_q, postop_q, log_file: str = None):
    '''
    adapt structured trees coupled to a 3d simulation based on the constant wall shear stress assumption
    
    :param config_handler: ConfigHandler instance
    :param preop_q: a list of preoperative flowrates at the outlets
    :param postop_q: a list of postoperative flowrates at the outlets
    :param log_file: path to log file, for writing important messages for debugging purposes
    '''

    outlet_idx = 0 # linear search, i know. its bad. will fix later
    for bc in config_handler.bcs.values():
        # we assume that an inlet location indicates taht this is an outlet bc and threfore undergoes adaptation
        if config_handler.coupling_blocks[bc.name].location == 'inlet':
            # adapt the corresponding tree
            R_old, R_new = config_handler.trees[outlet_idx].adapt_constant_wss(Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx])

            if np.isnan(R_new) or np.isnan(R_old):
                raise ValueError('nan resistance encountered')
            
            print(R_old, R_new)

            
            # add the updated resistance to the boundary condition
            if bc.type == 'RESISTANCE':
                bc.R = R_new
            elif bc.type == 'RCR':
                bc.Rp = 0.1 * R_new
                bc.Rd = 0.9 * R_new
            else:
                raise ValueError('unknown boundary condition type')

            outlet_idx += 1


def adapt_constant_wss_threed(preop_sim_dir, postop_sim_dir, location: str = 'uniform'):
    '''
    adapt structured trees coupled to a 3d simulation based on the constant wall shear stress assumption
    
    :param preop_sim_dir: SimulationDirectory instance for the preoperative simulation
    :param postop_sim_dir: SimulationDirectory instance for the postoperative simulation
    '''

    # get the preop and postop outlet flowrates
    preop_q = get_outlet_data(preop_sim_dir.config, preop_sim_dir.results['steady'], 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_sim_dir.config, postop_sim_dir.results['steady'], 'flow_out', steady=True)

    if location == 'uniform':
            # adapt one tree each for left and right based on flow split
            preop_lpa_flow, preop_rpa_flow = preop_sim_dir.flow_split()
            postop_lpa_flow, postop_rpa_flow = postop_sim_dir.flow_split()
    elif location == 'lobe':
        # adapt one tree for upper, lower, middle lobes
        pass
    elif location == 'all':
        # adapt a tree for each individual outlet
        pass

    # adapt the trees
    adapt_constant_wss_threed(postop_sim_dir.config_handler, preop_q, postop_q, log_file=postop_sim_dir.log_file)

    # simulate the adapted trees
    postop_sim_dir.simulate('adapted')
    postop_sim_dir.save_results('adapted')


def adapt_threed(preop_sim_dir, postop_sim_dir, adapted_sim_path, location: str = 'uniform', method: str = 'cwss'):
    '''
    adapt structured trees coupled to a 3d simulation based on the constant wall shear stress assumption
    
    :param preop_coupler_path: path to the preoperative coupling file
    :param postop_coupler_path: path to the postoperative coupling file
    :param preop_svzerod_data: path to preop svZeroD_data
    :param postop_svzerod_data: path to postop svZeroD_data
    :param location: str indicating the location of the adaptation
    :param method: str indicating the adaptation method
    '''

    # get the preop and postop outlet flowrates
    if location == 'uniform':
        # adapt one tree each for left and right based on flow split
        preop_lpa_flow, preop_rpa_flow = [sum(flow.values()) for flow in preop_sim_dir.flow_split()]
        postop_lpa_flow, postop_rpa_flow = [sum(flow.values()) for flow in postop_sim_dir.flow_split()]
    elif location == 'lobe':
        preop_lpa_flow, preop_rpa_flow = preop_sim_dir.flow_split()
        postop_lpa_flow, postop_rpa_flow = postop_sim_dir.flow_split()
    elif location == 'all':
        # adapt a tree for each individual outlet
        pass

    print(f"preop_lpa_flow: {preop_lpa_flow}, preop_rpa_flow: {preop_rpa_flow}")
    print(f"postop LPA flow: {postop_lpa_flow}, postop RPA flow: {postop_rpa_flow}")


    adapted_sim_dir = SimulationDirectory.from_directory(adapted_sim_path, mesh_complete=preop_sim_dir.mesh_complete.path)


    return adapted_sim_dir


class MicrovascularAdaptor: 
    '''
    class for computing microvascular adaptation from a preop and postop result
    '''

    def __init__(self, 
                 preopSimulationDirectory, 
                 postopSimulationDirectory, 
                 adaptedSimulationDirectory,
                 reducedOrderPA: json,
                 treeParams: csv, 
                 clinicalTargets,
                 method: str = 'cwss', 
                 location: str = 'uniform',
                 n_iter: int = 100,
                 convert_to_cm: bool = False):
        '''
        initialize the MicrovascularAdaptation class
        
        :param preopSimulationDirectory: SimulationDirectory opbject for the preoperative simulation
        :param postopSimulationDirectory: SimulationDirectory object for the postoperative simulation
        :param adaptedSimulationDirectory: SimulationDirectory object for the adapted simulation
        :param tree_params: csv file optimized_params.csv
        :param method: adaptation method, default is 'cwss'. options ['cwss', 'wss-ims']
        :param location: location of the adaptation, default is 'uniform'
        '''
        self.preopSimulationDirectory = preopSimulationDirectory
        self.postopSimulationDirectory = postopSimulationDirectory
        self.adaptedSimulationDirectory = adaptedSimulationDirectory

        self.simple_pa = ConfigHandler.from_json(reducedOrderPA, is_pulmonary=True)

        # grab tree params from csv, of form [k1, k2, k3, lrr, diameter]
        opt_params = pd.read_csv(os.path.join(treeParams))
        self.treeParams = {
            'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], opt_params['diameter'][opt_params.pa=='lpa'].values[0]],
            'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], opt_params['diameter'][opt_params.pa=='rpa'].values[0]]
        }

        self.clinicalTargets = clinicalTargets

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
            Z_t_l_pre, time = self.lpa_tree.compute_olufsen_impedance(self.treeParams['lpa'][0], self.treeParams['lpa'][1], self.treeParams['lpa'][2], n_procs=24)
            Z_t_r_pre, time = self.rpa_tree.compute_olufsen_impedance(self.treeParams['rpa'][0], self.treeParams['rpa'][1], self.treeParams['rpa'][2],n_procs=24)
            self.lpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'lpa_stiffness_preop.png'))
            self.rpa_tree.plot_stiffness(path=os.path.join(fig_dir, 'rpa_stiffness_preop.png'))

        sum_flows = lambda d: tuple(map(lambda x: sum(x.values()), d.flow_split(get_mean=True)))

        preop_lpa_flow, preop_rpa_flow = sum_flows(self.preopSimulationDirectory)
        postop_lpa_flow, postop_rpa_flow = sum_flows(self.postopSimulationDirectory)

        if self.method == 'cwss':
            self.lpa_tree.adapt_constant_wss(Q=preop_lpa_flow, Q_new=postop_lpa_flow, n_itern=self.n_iter)
            self.rpa_tree.adapt_constant_wss(Q=preop_rpa_flow, Q_new=postop_rpa_flow, n_iter=self.n_iter)
        elif self.method == 'wss-ims':
            self.lpa_tree.adapt_wss_ims(Q=preop_lpa_flow, Q_new=postop_lpa_flow, n_iter=self.n_iter)
            self.rpa_tree.adapt_wss_ims(Q=preop_rpa_flow, Q_new=postop_rpa_flow, n_iter=self.n_iter)

        print("computing adapted impedance! \n")
        Z_t_l_adapt, time = self.lpa_tree.compute_olufsen_impedance(self.treeParams['lpa'][0], self.treeParams['lpa'][1], self.treeParams['lpa'][2], n_procs=24)
        Z_t_r_adapt, time = self.rpa_tree.compute_olufsen_impedance(self.treeParams['rpa'][0], self.treeParams['rpa'][1], self.treeParams['rpa'][2],n_procs=24)

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
        
        self.adaptedSimulationDirectory.svzerod_3Dcoupling = self.postopSimulationDirectory.svzerod_3Dcoupling
        # change path
        self.adaptedSimulationDirectory.svzerod_3Dcoupling.path = os.path.join(self.adaptedSimulationDirectory.path, 'svzerod_3Dcoupling.json')
        # update the config with the adapted trees
        self.adaptedSimulationDirectory.svzerod_3Dcoupling.tree_params[self.lpa_tree.name] = self.lpa_tree.to_dict()
        self.adaptedSimulationDirectory.svzerod_3Dcoupling.tree_params[self.rpa_tree.name] = self.rpa_tree.to_dict()

        print("saving adapted config to " + self.adaptedSimulationDirectory.svzerod_3Dcoupling.path)
        self.adaptedSimulationDirectory.svzerod_3Dcoupling.to_json(self.adaptedSimulationDirectory.svzerod_3Dcoupling.path)

    def constructTrees(self):
        '''
        construct the trees for the preop and postop simulations
        '''

        k1_l, k2_l, k3_l, lrr_l, d_l = self.treeParams['lpa']
        k1_r, k2_r, k3_r, lrr_r, d_r = self.treeParams['rpa']

        time_array = self.preopSimulationDirectory.svzerod_3Dcoupling.bcs['INFLOW'].t
        
        lpa_tree = StructuredTree(name='LPA', time=time_array, simparams=None)
        print(f'building LPA tree with lpa parameters: {self.treeParams["lpa"]}')
        lpa_tree.build_tree(initial_d=d_l, d_min=0.01, lrr=lrr_l)

        rpa_tree = StructuredTree(name='RPA', time=time_array, simparams=None)
        print(f'building RPA tree with rpa parameters: {self.treeParams["rpa"]}')
        rpa_tree.build_tree(initial_d=d_r, d_min=0.01, lrr=lrr_r)

        return lpa_tree, rpa_tree

    def constructTreesFromConfig(self):
        
        time_array = self.preopSimulationDirectory.svzerod_3Dcoupling.bcs['INFLOW'].t

        for name, params in self.preopSimulationDirectory.svzerod_3Dcoupling.tree_params.items():
            tree = StructuredTree(name=name, time=time_array, simparams=None)
            print(f'building {name} tree with parameters: {params}')
            tree.build_tree(initial_d=params['initial_d'], d_min=params['d_min'], lrr=params['lrr'])

    def createImpedanceBCs(self):
        '''
        create the impedance boundary conditions for the adapted trees
        '''
        # if self.location == 'uniform':
        
        cap_info = vtp_info(self.postopSimulationDirectory.mesh_complete.mesh_surfaces_dir, convert_to_cm=self.convert_to_cm, pulmonary=False)

        outlet_bc_names = [name for name, bc in self.postopSimulationDirectory.svzerod_3Dcoupling.bcs.items() if 'inflow' not in bc.name.lower()]

        # assumed that cap and boundary condition orders match, TODO: UPDATE THIS TO BE USED with SIMULATIONDIRECTORY CLASS
        if len(outlet_bc_names) != len(cap_info):
            print('number of outlet boundary conditions does not match number of cap surfaces, automatically assigning bc names...')
            for i, name in enumerate(outlet_bc_names):
                # delete the unused bcs
                del self.postopSimulationDirectory.svzerod_3Dcoupling.bcs[name]
            outlet_bc_names = [f'IMPEDANCE_{i}' for i in range(len(cap_info))]
        
        cap_to_bc = {list(cap_info.keys())[i]: outlet_bc_names[i] for i in range(len(outlet_bc_names))}

        for idx, (cap_name, area) in enumerate(cap_info.items()):
                    print(f'generating tree {idx + 1} of {len(cap_info)} for cap {cap_name}...')
                    if 'lpa' in cap_name.lower():
                        self.postopSimulationDirectory.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.lpa_tree.create_impedance_bc(cap_to_bc[cap_name], 0, self.clinicalTargets.wedge_p * 1333.2)
                    elif 'rpa' in cap_name.lower():
                        self.postopSimulationDirectory.svzerod_3Dcoupling.bcs[cap_to_bc[cap_name]] = self.rpa_tree.create_impedance_bc(cap_to_bc[cap_name], 1, self.clinicalTargets.wedge_p * 1333.2)
                    else:
                        raise ValueError('cap name not recognized')
                    
    
    def adaptCWSS_IMS(self, fig_dir: str = None):
        # TODO: Implement later as here we are using the 3D model to adapt the trees
        pass