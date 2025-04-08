from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.preop import *
from svzerodtrees.inflow import *
from svzerodtrees.simulation_directory import *
from svzerodtrees.structuredtree import *
import json
import pandas as pd
import pickle
import os
import vtk

class Simulation:
    '''
    the overall class for submitting an svZeroDTrees Simulation
    '''

    def __init__(self,
                 path='.',
                 clinical_targets: csv = 'clinical_targets.csv',
                 preop_dir='preop',
                 postop_dir='postop',
                 adapted_dir='adapted',
                 steady_dir='steady',
                 adaptation='cwss',
                 adapt_location='uniform',
                 zerod_config='zerod_config.json',
                 convert_to_cm=False,
                 optimized=False):
        
        self.path = os.path.abspath(path)

        # zerod configs
        self.zerod_config_path = os.path.join(self.path, zerod_config)
        self.simplified_zerod_config = os.path.join(self.path, 'simplified_nonlinear_zerod.json')

        self.preop_dir = SimulationDirectory.from_directory(path=os.path.join(self.path, preop_dir), zerod_config=self.zerod_config_path, convert_to_cm=convert_to_cm)
        self.postop_dir = SimulationDirectory.from_directory(path=os.path.join(self.path, postop_dir), zerod_config=self.zerod_config_path, convert_to_cm=convert_to_cm)
        self.adapted_dir = os.path.join(self.path, adapted_dir) # just a path initially
        self.adaptation = adaptation
        self.adapt_location = adapt_location
        self.steady_dir = os.path.join(self.path, steady_dir)

        # simulation parameters
        self.n_tsteps = 2000

        ## Bools
        self.optimized = optimized
        self.convert_to_cm = convert_to_cm

        self.clinical_targets = ClinicalTargets.from_csv(clinical_targets)

        # use a generic inflow profile
        self.inflow = Inflow.periodic()
        self.inflow.rescale(cardiac_output=self.clinical_targets.q, tsteps=self.n_tsteps)

        # make a figures directory
        self.figures_dir = os.path.join(self.path, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    

    @classmethod
    def from_config(cls, config_file):
        '''
        create simulation from config file containing directories, adaptation and clinical targets'''
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        ## TO BE IMPLEMENTED LATER
        return cls(**config)
    
    def run_pipeline(self, run_steady=True, bcs_optimized=False):
        '''
        run the entire pipeline
        '''
        if run_steady:
            # run the steady simulations
            self.run_steady_sims()

            # generate the simplified zerod config
            self.generate_simplified_nonlinear_zerod()

            # optimize preop BCs
            reduced_config = ConfigHandler.from_json(self.simplified_zerod_config, is_pulmonary=True)
        else:
            reduced_config = ConfigHandler.from_json(self.simplified_zerod_config, is_pulmonary=True)
        
        if not bcs_optimized:
            optimize_impedance_bcs(reduced_config, self.preop_dir.mesh_complete.mesh_surfaces_dir, self.clinical_targets, opt_config_path=self.zerod_config_path, d_min=0.01, convert_to_cm=self.convert_to_cm, n_procs=24)
            # need to create coupling config and add to preop/postop directories
                # build trees for LPA/RPA

        # construct trees
        opt_params = pd.read_csv(os.path.join(self.path, 'optimized_params.csv'))
        tree_params = {
            'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], opt_params['diameter'][opt_params.pa=='lpa'].values[0]],
            'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], opt_params['diameter'][opt_params.pa=='rpa'].values[0]]
        }

        # generate blank threed coupler
        # blank_threed_coupler = ConfigHandler.blank_threed_coupler(path=os.path.join(self.path, 'svzerod_3Dcoupling.json'))
        if os.path.exists(self.zerod_config_path):
            self.zerod_config = ConfigHandler.from_json(self.zerod_config_path)
        else:
            self.zerod_config = ConfigHandler.from_json(self.simplified_zerod_config)

        # rescale the inflow, in this case the first element in the inflows dict
        self.zerod_config.inflows[next(iter(self.zerod_config.inflows))].rescale(tsteps=2000)

        # create the trees
        construct_impedance_trees(self.zerod_config, self.preop_dir.mesh_complete.mesh_surfaces_dir, self.clinical_targets.wedge_p, d_min=0.01, convert_to_cm=self.convert_to_cm, use_mean=True, specify_diameter=True, tree_params=tree_params)

        impedance_threed_coupler, coupling_block_list = self.zerod_config.generate_threed_coupler(self.preop_dir.path, mesh_complete=self.preop_dir.mesh_complete)
        self.zerod_config.to_json(self.zerod_config_path)
        # run preop + postop simulations
        sim_config = {
            'n_tsteps': 10000,
            'dt': 0.0005,
            'nodes': 4,
            'procs_per_node': 24,
            'memory': 16,
            'hours': 16
        }
        preop_sim = SimulationDirectory.from_directory(self.preop_dir.path, self.zerod_config_path, convert_to_cm=self.convert_to_cm)
        preop_sim.write_files(simname='Preop Simulation', user_input=False, sim_config=sim_config)
        preop_sim.run()

        postop_sim = SimulationDirectory.from_directory(self.postop_dir.path, self.zerod_config_path, convert_to_cm=self.convert_to_cm)
        postop_sim.write_files(simname='Postop Simulation', user_input=False, sim_config=sim_config)
        postop_sim.run()

        # compute adaptation
        self.compute_adaptation


        # run adapted simulation


        # postprocess results

    def compute_adaptation(self):
        '''
        compute the adaptation based on a method and location
        '''

        if self.adaptation == 'cwss':
            pass
        elif self.adaptation == 'szafron':
            pass
        elif self.adaptation == 'szafron_simple':
            pass
        elif self.adaptation == 'ps':
            pass
        else:
            raise ValueError('Invalid adaptation method')
        
        if self.adapt_location == 'uniform':
            # adapt one tree each for left and right based on flow split
            pass
        elif self.adapt_location == 'lobe':
            # adapt one tree for upper, lower, middle lobes
            pass
        elif self.adapt_location == 'all':
            # adapt a tree for each individual outlet
            pass

    def run_steady_sims(self):
        '''
        run steady state simulations
        '''

        print('setting up and running steady simulations...')
        # create the steady directory if it doesn't exist
        if not os.path.exists(self.steady_dir):
            os.makedirs(self.steady_dir)

        # make the steady simulations
        flow_dict = {
            'sys': max(self.inflow.q),
            'dia': 2.0,
            'mean': np.mean(self.inflow.q)
        }
        steady_sims = {}
        for label, q in flow_dict.items():
            dir_path = os.path.join(self.steady_dir, label)
            os.makedirs(dir_path, exist_ok=True)
            # create the steady simulation
            steady_sims[label] = SimulationDirectory.from_directory(path=dir_path, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm)
            steady_sims[label].generate_steady_sim(flow_rate=q)
            # cd into directory to sumit simulation
            os.chdir(steady_sims[label].path)
            steady_sims[label].run()
            # cd back to the original directory
            os.chdir(self.path)
        
        # now check simulations
        for label, sim in steady_sims.items():
            sim.check_simulation()
    
    def generate_simplified_nonlinear_zerod(self, sys_dir=None, dia_dir=None, mean_dir=None, plot=True):
        '''
        generate a simplified zerod config with nonlinear resistors, using data from sys/dia/mean flow steady simulations'''
        cardiac_output = self.clinical_targets.q
        
        if sys_dir is None:
            sys_dir = os.path.join(self.steady_dir, 'sys')
        if dia_dir is None:
            dia_dir = os.path.join(self.steady_dir, 'dia')
        if mean_dir is None:
            mean_dir = os.path.join(self.steady_dir, 'mean')

        sys_sim = SimulationDirectory.from_directory(sys_dir, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm, is_pulmonary=True)
        dia_sim = SimulationDirectory.from_directory(dia_dir, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm, is_pulmonary=True)
        mean_sim = SimulationDirectory.from_directory(mean_dir, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm, is_pulmonary=True)

        R_sys_lpa, R_sys_rpa = sys_sim.compute_pressure_drop()
        R_dia_lpa, R_dia_rpa = dia_sim.compute_pressure_drop()
        R_mean_lpa, R_mean_rpa = mean_sim.compute_pressure_drop()

        # get the flow for each simulation
        Q_sys_lpa, Q_sys_rpa = [sum(q.values()) for q in sys_sim.flow_split(verbose=False)]
        Q_dia_lpa, Q_dia_rpa = [sum(q.values()) for q in dia_sim.flow_split(verbose=False)]
        Q_mean_lpa, Q_mean_rpa = [sum(q.values()) for q in mean_sim.flow_split(verbose=False)]

        # compute the linear relationship between resistance and flow
        S_lpa = np.polyfit([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], [R_sys_lpa, R_dia_lpa, R_mean_lpa], 1)
        S_rpa = np.polyfit([Q_sys_rpa, Q_dia_rpa, Q_mean_rpa], [R_sys_rpa, R_dia_rpa, R_mean_rpa], 1)

        if plot:
            # plot the data and the fit
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))

            q = np.linspace(0, 100, 100)

            ax[0].scatter([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], [R_sys_lpa, R_dia_lpa, R_mean_lpa], label='LPA')
            ax[0].plot(q, np.polyval(S_lpa, q), label='LPA fit')
            ax[0].set_xlabel('Flow (mL/s)')
            ax[0].set_ylabel('Resistance (dyn/cm5/s)')
            ax[0].set_title('LPA flow vs resistance')
            ax[0].legend()

            ax[1].scatter([Q_sys_rpa, Q_dia_rpa, Q_mean_rpa], [R_sys_rpa, R_dia_rpa, R_mean_rpa], label='RPA')
            ax[1].plot(q, np.polyval(S_rpa, q), label='RPA fit')
            ax[1].set_xlabel('Flow (mL/s)')
            ax[1].set_ylabel('Resistance (dyn/cm5/s)')
            ax[1].set_title('RPA flow vs resistance')
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'resistance_fit.png'))


        # create the config
        # need to rescale the inflow and make it periodic with a generic shape (see Inflow class)
        inflow = Inflow.periodic(path=None)
        # inflow.rescale(cardiac_output=mean_sim.svzerod_3Dcoupling.bcs['inflow'].Q[0])
        inflow.rescale(cardiac_output=cardiac_output)

        config = ConfigHandler({
            "boundary_conditions": [
                inflow.to_dict(),
                {
                    "bc_name": "LPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "Pd": 6.0,
                        "R": 100.0
                    }
                },
                {
                    "bc_name": "RPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "Pd": 6.0,
                        "R": 100.0
                    }
                }
            ],
            "simulation_parameters": {
                "output_all_cycles": False,
                "steady_initial": False,
                "density": 1.06,
                "model_name": "pa_reduced",
                "number_of_cardiac_cycles": 8,
                "number_of_time_pts_per_cardiac_cycle": 500,
                "viscosity": 0.04
            },
            "junctions": [
                {
                    "junction_name": "J0",
                    "junction_type": "NORMAL_JUNCTION",
                    "inlet_vessels": [
                        0
                    ],
                    "outlet_vessels": [
                        1,
                        3
                    ]
                },
                {
                    "junction_name": "J1",
                    "junction_type": "NORMAL_JUNCTION",
                    "inlet_vessels": [
                        1
                    ],
                    "outlet_vessels": [
                        2
                    ]
                },
                {
                    "junction_name": "J2",
                    "junction_type": "NORMAL_JUNCTION",
                    "inlet_vessels": [
                        3
                    ],
                    "outlet_vessels": [
                        4
                    ]
                }
            ],
            "vessels": [
                {
                    "boundary_conditions": {
                        "inlet": "INFLOW"
                    },
                    "vessel_id": 0,
                    "vessel_length": 1.0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": 0.0
                    }
                },
                {
                    "vessel_id": 1,
                    "vessel_length": 1.0,
                    "vessel_name": "branch1_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": S_lpa[0] / 2
                    }
                },
                {
                    "boundary_conditions": {
                        "outlet": "LPA_BC"
                    },
                    "vessel_id": 2,
                    "vessel_length": 1.0,
                    "vessel_name": "branch2_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": S_lpa[0] / 2
                    }
                },
                {
                    "vessel_id": 3,
                    "vessel_length": 1.0,
                    "vessel_name": "branch3_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": S_rpa[0] / 2
                    }
                },
                {
                    "boundary_conditions": {
                        "outlet": "RPA_BC"
                    },
                    "vessel_id": 4,
                    "vessel_length": 1.0,
                    "vessel_name": "branch4_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": S_rpa[0] / 2
                    }
                }
            ]
        })


        config.to_json(self.simplified_zerod_config)

        config.simulate()

        print('simplified zerod config is simulatable')
        

    def plot_optimized_pa_config(self, path='pa_config_test_tuning.json'):
        '''
        plot the optimized zerod config
        '''

        # load the config
        config_handler = ConfigHandler.from_json(path)
        optimized_pa_config = PAConfig.from_pa_config(config_handler, self.clinical_targets)

        # plot the config
        optimized_pa_config.plot_mpa()


    def postprocess_3d_results(self):
        '''
        postprocess the 3d results
        '''

        pass