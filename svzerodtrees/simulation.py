from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.preop import *
from svzerodtrees.inflow import *
from svzerodtrees.simulation_directory import *
from svzerodtrees.structuredtree import *
import json
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
                 convert_to_cm=False,
                 optimized=False):
        
        self.path = os.path.abspath(path)

        # zerod configs
        self.zerod_config = os.path.join(self.path, 'optimized_zerod_config.json')
        self.simplified_zerod_config = os.path.join(self.path, 'simplified_nonlinear_zerod.json')

        self.preop_dir = SimulationDirectory.from_directory(path=os.path.join(self.path, preop_dir), zerod_config=self.zerod_config, convert_to_cm=convert_to_cm)
        self.postop_dir = SimulationDirectory.from_directory(path=os.path.join(self.path, postop_dir), zerod_config=self.zerod_config, convert_to_cm=convert_to_cm)
        self.adapted_dir = os.path.join(self.path, adapted_dir) # just a path initially
        self.adaptation = adaptation
        self.steady_dir = os.path.join(self.path, steady_dir)

        ## Bools
        self.optimized = optimized
        self.convert_to_cm = convert_to_cm

        self.clinical_targets = ClinicalTargets.from_csv(clinical_targets)

        # use a generic inflow profile
        self.inflow = Inflow.periodic()
        self.inflow.rescale(cardiac_output=self.clinical_targets.q)

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
            self.generate_simplified_nonlinear_zerod(cardiac_output=self.clinical_targets.q)

            # optimize preop BCs
            reduced_config = ConfigHandler.from_json(self.simplified_zerod_config)
        
        if not bcs_optimized:
            optimize_impedance_bcs(reduced_config, self.preop_dir.mesh_surfaces.path, self.clinical_targets, opt_config_path=self.zerod_config, d_min=0.01, convert_to_cm=self.convert_to_cm, n_procs=24)

        # run preop + postop simulations
        sim_config = {
            'n_tsteps': 10000,
            'dt': 0.0005,
            'nodes': 4,
            'procs_per_node': 24,
            'memory': 16,
            'hours': 16
        }
        preop_sim = SimulationDirectory.from_directory(self.preop_dir.path, self.zerod_config, convert_to_cm=self.convert_to_cm)
        preop_sim.write_files()
        preop_sim.run(simname='Preop Simulation', user_input=False, sim_config=sim_config)

        postop_sim = SimulationDirectory.from_directory(self.postop_dir.path, self.zerod_config, convert_to_cm=self.convert_to_cm)
        postop_sim.write_files()
        postop_sim.run(simname='Postop Simulation', user_input=False, sim_config=sim_config)

        # compute adaptation
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


        # run adapted simulation


        # postprocess results



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
            'dia': 1.0,
            'mean': np.mean(self.inflow.q)
        }

        for label, q in flow_dict.items():
            dir_path = os.path.join(self.steady_dir, label)
            os.makedirs(dir_path, exist_ok=True)
            # create the steady simulation
            sim = SimulationDirectory.from_directory(path=dir_path, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm)
            sim.generate_steady_sim(flow_rate=q, wedge_p=self.clinical_targets.wedge_p)
            sim.run()
        
        # now check simulations
        for label, q in flow_dict.items():
            dir_path = os.path.join(self.steady_dir, label)
            sim = SimulationDirectory.from_directory(path=dir_path, mesh_complete=self.preop_dir.mesh_complete.path, convert_to_cm=self.convert_to_cm)
            sim.check_simulation()
    
    def generate_simplified_nonlinear_zerod(self, cardiac_output, sys_dir=None, dia_dir=None, mean_dir=None, plot=True):
        '''
        generate a simplified zerod config with nonlinear resistors, using data from sys/dia/mean flow steady simulations'''

        if sys_dir is None:
            sys_dir = os.path.join(self.steady_dir, 'sys')
        if dia_dir is None:
            dia_dir = os.path.join(self.steady_dir, 'dia')
        if mean_dir is None:
            mean_dir = os.path.join(self.steady_dir, 'mean')

        sys_sim = SimulationDirectory.from_directory(sys_dir, is_pulmonary=False)
        dia_sim = SimulationDirectory.from_directory(dia_dir, is_pulmonary=False)
        mean_sim = SimulationDirectory.from_directory(mean_dir, is_pulmonary=False)

        R_sys_lpa, R_sys_rpa = sys_sim.compute_pressure_drop()
        R_dia_lpa, R_dia_rpa = dia_sim.compute_pressure_drop()
        R_mean_lpa, R_mean_rpa = mean_sim.compute_pressure_drop()

        # get the flow for each simulation
        Q_sys_lpa, Q_sys_rpa = sys_sim.flow_split()
        Q_dia_lpa, Q_dia_rpa = dia_sim.flow_split()
        Q_mean_lpa, Q_mean_rpa = mean_sim.flow_split()

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
        

    def postprocess_3d_results(self):
        '''
        postprocess the 3d results
        '''

        pass