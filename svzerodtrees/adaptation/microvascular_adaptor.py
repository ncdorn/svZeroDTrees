import json
import csv
from ..io import ConfigHandler
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