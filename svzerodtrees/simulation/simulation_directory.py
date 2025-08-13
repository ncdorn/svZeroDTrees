from ..utils import *
from .threedutils import *
from ..io import *
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import copy
import time
import os
import math
from .input_builders import *
from ..tune_bcs import ClinicalTargets
from ..io.blocks import *
from ..tune_bcs import construct_impedance_trees, ClinicalTargets
from .input_builders.simulation_file import SimulationFile
from ..microvasculature import StructuredTree

class SimulationDirectory:
    '''
    a class for handling simulation directories of pulmonary artery simulations'''
    def __init__(self, path, 
                 zerod_config=None,
                 mesh_complete=None, 
                 svzerod_interface=None, 
                 svzerod_3Dcoupling=None, 
                 svFSIxml=None, 
                 solver_runscript=None, 
                 svzerod_data=None, 
                 results_dir=None,
                 clinical_target=None,
                 fig_dir=None,
                 convert_to_cm=False):
        '''
        initialize the simulation handler which handles threed simulation data'''

        # path to the simulation directory
        self.path = path

        # sim name
        self.simname = f"simulation {os.path.basename(path)}"

        # zerod model
        self.zerod_config = zerod_config

        # mesh complete directory
        self.mesh_complete = mesh_complete

        # svZeroD_interface.dat file
        self.svzerod_interface = svzerod_interface

        # svzerod_3Dcoupling.json file
        self.svzerod_3Dcoupling = svzerod_3Dcoupling

        self.solver_runscript = solver_runscript

        # svFSI.xml file
        self.svFSIxml = svFSIxml

        # simulation results
        ## svZeroD_data
        self.svzerod_data = svzerod_data

        ## result*.vtu files
        self.results_dir = results_dir

        self.clinical_targets = clinical_target

        # figures directory
        self.fig_dir = fig_dir

        self.results_file = os.path.join(self.path, 'results.txt')

        self.convert_to_cm = convert_to_cm

    @classmethod
    def from_directory(cls, path='.', zerod_config: str =None, mesh_complete: str ='mesh-complete', threed_coupler=None, results_dir: str =None, convert_to_cm: bool =False, is_pulmonary=True):
        '''
        create a simulation directory object from the path to the simulation directory
        and search for the necessary files within the path'''

        path = os.path.abspath(path)

        print(f'\n\n *** INITIALIZING SIMULATION DIRECTORY: {os.path.basename(path)} *** \n\n')

        # check for zerod model
        if zerod_config is not None and os.path.exists(zerod_config):
            print('zerod model found')
            zerod_config = ConfigHandler.from_json(zerod_config, is_pulmonary=False)
            if os.path.dirname(zerod_config.path) != path:
                print('copying zerod model to simulation directory')
                os.system(f'cp {zerod_config.path} {path}')
                zerod_config.path = os.path.join(path, os.path.basename(zerod_config.path))
        else:
            print('zerod model not found! you will need to create one or add one...')
            zerod_config = None

        # check for mesh complete
        mesh_complete = os.path.join(path, mesh_complete)
        if os.path.exists(mesh_complete):
            print('mesh-complete found')
            mesh_complete = MeshComplete(mesh_complete)
            mesh_complete.rename_vtps()
        else:
            print('mesh-complete not found')
            mesh_complete = None
        
        # check for svZeroD_interface.dat
        svzerod_interface = os.path.join(path, 'svZeroD_interface.dat')
        if os.path.exists(svzerod_interface):
            print('svZeroD_interface.dat found')
            svzerod_interface = SvZeroDInterface(svzerod_interface)
        else:
            print('svZeroD_interface.dat not found')
            svzerod_interface = SvZeroDInterface(svzerod_interface)

        # check for svzerod_3Dcoupling.json
        svzerod_3Dcoupling = os.path.join(path, 'svzerod_3Dcoupling.json')
        if threed_coupler is not None and zerod_config is not None and mesh_complete is not None:
            print('generating svzerod_3Dcoupling.json from zerod model...')
            svzerod_3Dcoupling, coupling_blocks = zerod_config.generate_threed_coupler(path, 
                                                                                        inflow_from_0d=True, 
                                                                                        mesh_complete=mesh_complete)
        elif os.path.exists(svzerod_3Dcoupling):
            print('svzerod_3Dcoupling.json found')
            svzerod_3Dcoupling = ConfigHandler.from_json(svzerod_3Dcoupling, is_pulmonary=False)
        else:
            print('zerod model not found or mesh-complete not foun, generating blank svzerod_3Dcoupling.json')
            svzerod_3Dcoupling = ConfigHandler.blank_threed_coupler(path=os.path.join(path, 'svzerod_3Dcoupling.json'))

        # check for svFSI.xml
        svFSIxml = os.path.join(path, 'svFSIplus.xml')
        if os.path.exists(svFSIxml):
            print('svFSI.xml found')
            svFSIxml = SvMPxml(svFSIxml)
        else:
            print('svFSI.xml not found')
            svFSIxml = SvMPxml(svFSIxml)

        # check for solver runscript
        solver_runscript = os.path.join(path, 'run_solver.sh')
        if os.path.exists(solver_runscript):
            print('solver runscript found')
            solver_runscript = SolverRunscript(solver_runscript)
        else:
            print('solver runscript not found')
            solver_runscript = SolverRunscript(solver_runscript)

        # check for svZeroD_data
        zerod_data = os.path.join(path, 'svZeroD_data')
        if os.path.exists(zerod_data):
            print('svZeroD_data result found, adding to threed_coupler')
            svzerod_data = SvZeroDdata(zerod_data)
            svzerod_3Dcoupling.add_result(svzerod_data=svzerod_data)
        else:
            print('svZeroD_data result not found')
            svzerod_data = SvZeroDdata(zerod_data)

        # check for results directory
        if results_dir is not None:
            results_dir = os.path.join(path, results_dir)
            if os.path.exists(results_dir):
                print('results directory found')
            else:
                print('results directory not found')
                results_dir = SimResults(results_dir)


        if os.path.exists(os.path.join(path, 'figures')):
            print('figures directory found!')
            fig_dir = os.path.join(path, 'figures')
        else:
            print('creating figures directory...')
            fig_dir = os.path.join(path, 'figures')
            os.system(f'mkdir {fig_dir}')

        if os.path.exists(os.path.join(path, 'clinical_targets.csv')):
            print('clinical targets found!')
            clinical_targets = ClinicalTargets.from_csv(os.path.join(path, 'clinical_targets.csv'))
        else:
            clinical_targets = None
        return cls(path,
                   zerod_config,
                   mesh_complete, 
                   svzerod_interface, 
                   svzerod_3Dcoupling, 
                   svFSIxml, 
                   solver_runscript, 
                   svzerod_data, 
                   results_dir,
                   clinical_targets,
                   fig_dir,
                   convert_to_cm)
    
    def duplicate(self, new_path):
        '''
        copy the simulation directory object do a new directory'''

        os.system(f'cp -r {self.path} {new_path}')

        return SimulationDirectory.from_directory(new_path)

    def run(self):
        '''
        run the simulation'''

        os.chdir(self.path)

        self.check_files(verbose=False)

        os.system('clean')
        os.system(f'sbatch {self.solver_runscript.path}')

        os.chdir("..")

    def check_files(self, verbose=True):
        '''
        check if the simulation directory has all the necessary files'''

        if self.mesh_complete.is_written:
            if verbose:
                print('mesh-complete exists')
        else:
            raise FileNotFoundError('mesh-complete does not exist')
        
        if self.svzerod_interface.is_written:
            if verbose:
                print('svZeroD_interface.dat written')
        else:
            raise FileNotFoundError('svZeroD_interface.dat does not exist')
        
        if self.svFSIxml.is_written:
            if verbose:
                print('svFSI.xml written')
        else:
            raise FileNotFoundError('svFSI.xml does not exist')
        
        if self.solver_runscript.is_written:
            if verbose:
                print('solver runscript written')
        else:
            raise FileNotFoundError('solver runscript does not exist')
        
        if self.svzerod_3Dcoupling.is_written:
            if verbose:
                print('svzerod_3Dcoupling.json written')
        else:
            raise FileNotFoundError('svzerod_3Dcoupling.json does not exist')
        
        print('ready to run simulation!')

        return True

    def write_files(self, simname='SIMULATION', user_input=True, sim_config=None):
        '''
        write simulation files to the simulation directory'''

        # write the 3d-0d coupling json file

        print(f'writing files for {simname}...')

        self.svzerod_interface.write(self.svzerod_3Dcoupling.path)

        def write_svfsixml_input_params(user_input=user_input, sim_config=sim_config):
            if user_input:
                n_tsteps = int(input('number of time steps (default 5000): ') or 5000)
                dt = float(input('time step size (default 0.001): ') or 0.001)
            else:
                if sim_config is None:
                    raise ValueError('sim_config is None, cannot write svFSI.xml')
                n_tsteps = sim_config['n_tsteps']
                dt = sim_config['dt']
            if self.convert_to_cm:
                print("scaling mesh to cm...")
                mesh_scale_factor = 0.1
            else:
                mesh_scale_factor = 1.0
            self.svFSIxml.write(self.mesh_complete, n_tsteps=n_tsteps, dt=dt, scale_factor=mesh_scale_factor)
        
        def write_runscript_input_params(user_input=user_input, sim_config=sim_config):
            if user_input:
                nodes = int(input('number of nodes (default 3): ') or 3)
                procs_per_node = int(input('number of processors per node ( default 24): ') or 24)
                memory = int(input('memory per node in GB (default 16): ') or 16)
                hours = int(input('number of hours (default 6): ') or 12)
            else:
                nodes = sim_config['nodes']
                procs_per_node = sim_config['procs_per_node']
                memory = sim_config['memory']
                hours = sim_config['hours']
            self.solver_runscript.write(nodes=nodes, procs_per_node=procs_per_node, hours=hours, memory=memory)


        if self.svFSIxml.is_written:
            if user_input:
                rewrite = input('\nsvFSI.xml already exists, overwrite? y/n: ')
                if rewrite.lower() == 'y':
                    write_svfsixml_input_params()
            else:
                # automatically rewrite
                write_svfsixml_input_params()
        else:
            write_svfsixml_input_params()

        if self.solver_runscript.is_written:
            if user_input:
                rewrite = input('solver runscript already exists, overwrite? y/n: ')
                if rewrite.lower() == 'y':
                    write_runscript_input_params()
            else:
                # automatically rewrite
                write_runscript_input_params()
        else:
            write_runscript_input_params()

        self.check_files()

    def generate_steady_sim(self, flow_rate=None):
        '''
        generate simulation files for a steady simulation'''

        wedge_p = 0.0

        # add the inflows to the svzerod_3Dcoupling
        tsteps = 100
        self.svzerod_3Dcoupling.simparams.number_of_time_pts_per_cardiac_cycle = tsteps
        bc_idx = 0
        for vtp in self.mesh_complete.mesh_surfaces.values():
            if 'inflow' in vtp.filename.lower():
                # need to get inflow path or steady flow rate
                if flow_rate is None:
                    flow_rate = float(input(f'input steady flow rate for {vtp.filename}: '))
                else:
                    flow_rate = flow_rate

                try:
                    inflow = Inflow.steady(flow_rate, name=vtp.filename.split('.')[0].upper())
                    inflow.rescale(tsteps=tsteps)
                except:
                    print('invalid input, please provide a valid path to a flow file or a steady flow rate')
                    return

                self.svzerod_3Dcoupling.set_inflow(inflow, vtp.filename.split('.')[0].upper(), threed_coupled=False)
            else:

                bc_name = f'RESISTANCE_{bc_idx}'

                self.svzerod_3Dcoupling.bcs[bc_name] = BoundaryCondition({
                    "bc_name": bc_name,
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "Pd": wedge_p,
                        "R": 0.0
                    }
                })

                bc_idx += 1

        self.svzerod_3Dcoupling.to_json('blank_zerod_config.json')
        self.svzerod_3Dcoupling, coupling_blocks = self.svzerod_3Dcoupling.generate_threed_coupler(self.path, inflow_from_0d=True, mesh_complete=self.mesh_complete)

        sim_config = {
            'n_tsteps': 100,
            'dt': 0.0005,
            'nodes': 1,
            'procs_per_node': 24,
            'memory': 16,
            'hours': 6
        }

        self.write_files(simname='Steady Simulation', user_input=False, sim_config=sim_config)
    
    def compute_pressure_drop(self, steady=True):
        '''
        compute the pressure drop across the LPA and RPA based on the simulation results'''


        # compute the pressure drop
        if steady:
            print("computing steady pressure drop...")
            # get lpa, rpa flow
            lpa_flow, rpa_flow = self.flow_split(get_mean=True)

            # get the MPA pressure
            mpa_pressure = np.mean(self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])[2][-100:])

            lpa_outlet_pressures = []
            rpa_outlet_pressures = []
            for block in self.svzerod_3Dcoupling.coupling_blocks.values():
                if 'lpa' in block.surface.lower():
                    lpa_outlet_pressures.append(np.mean(self.svzerod_data.get_result(block)[2][-100:]))
                if 'rpa' in block.surface.lower():
                    rpa_outlet_pressures.append(np.mean(self.svzerod_data.get_result(block)[2][-100:]))

            lpa_outlet_mean_pressure = np.mean(lpa_outlet_pressures)
            rpa_outlet_mean_pressure = np.mean(rpa_outlet_pressures)

            lpa_pressure_drop = mpa_pressure - lpa_outlet_mean_pressure
            rpa_pressure_drop = mpa_pressure - rpa_outlet_mean_pressure

            lpa_resistance = lpa_pressure_drop / sum(lpa_flow.values())
            rpa_resistance = rpa_pressure_drop / sum(rpa_flow.values())

            print(f'LPA pressure drop: {lpa_pressure_drop / 1333.2} mmHg, \n RPA pressure drop: {rpa_pressure_drop / 1333.2} mmHg')
            print(f'LPA resistance: {lpa_resistance} dyn/cm5/s, \n RPA resistance: {rpa_resistance} dyn/cm5/s')
        
        else:
            print("computing systolic/diastolic/mean pressure drop...")
            lpa_flow, rpa_flow = self.flow_split(get_mean=False)

            # get the MPA mean, systolic, diastolic pressure
            time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
            time = time[time > time.max() - 1.0]
            pressure = pressure[time.index]
            sys_p = np.max(pressure)
            dia_p = pressure.iloc[0]
            mean_p = np.mean(pressure)

            lpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
            rpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
            for block in self.svzerod_3Dcoupling.coupling_blocks.values():
                if 'lpa' in block.surface.lower():
                    time, flow, pressure = self.svzerod_data.get_result(block)
                    time = time[time > time.max() - 1.0]
                    pressure = pressure[time.index]
                    lpa_outlet_pressures['sys'].append(np.max(pressure))
                    lpa_outlet_pressures['dia'].append(pressure.iloc[0])
                    lpa_outlet_pressures['mean'].append(np.mean(pressure))
                if 'rpa' in block.surface.lower():
                    time, flow, pressure = self.svzerod_data.get_result(block)
                    time = time[time > time.max() - 1.0]
                    pressure = pressure[time.index]
                    rpa_outlet_pressures['sys'].append(np.max(pressure))
                    rpa_outlet_pressures['dia'].append(pressure.iloc[0])
                    rpa_outlet_pressures['mean'].append(np.mean(pressure))

            lpa_pressure_drops = {
                'sys': sys_p - np.mean(lpa_outlet_pressures['sys']),
                'dia': dia_p - np.mean(lpa_outlet_pressures['dia']),
                'mean': mean_p - np.mean(lpa_outlet_pressures['mean'])
            }
            rpa_pressure_drops = {
                'sys': sys_p - np.mean(rpa_outlet_pressures['sys']),
                'dia': dia_p - np.mean(rpa_outlet_pressures['dia']),
                'mean': mean_p - np.mean(rpa_outlet_pressures['mean'])
            }

            Q_sys_lpa = sum(lpa_flow['sys'].values())
            Q_dia_lpa = sum(lpa_flow['dia'].values())
            Q_mean_lpa = sum(lpa_flow['mean'].values())
            Q_sys_rpa = sum(rpa_flow['sys'].values())
            Q_dia_rpa = sum(rpa_flow['dia'].values())
            Q_mean_rpa = sum(rpa_flow['mean'].values())

            lpa_resistance = {
                'sys': lpa_pressure_drops['sys'] / Q_sys_lpa,
                'dia': lpa_pressure_drops['dia'] / Q_dia_lpa,
                'mean': lpa_pressure_drops['mean'] / Q_mean_lpa
            }
            rpa_resistance = {
                'sys': rpa_pressure_drops['sys'] / Q_sys_rpa,
                'dia': rpa_pressure_drops['dia'] / Q_dia_rpa,
                'mean': rpa_pressure_drops['mean'] / Q_mean_rpa
            }

            print(f'\nLPA pressure drop: {lpa_pressure_drops["sys"] / 1333.2} mmHg, {lpa_pressure_drops["dia"] / 1333.2} mmHg, {lpa_pressure_drops["mean"] / 1333.2} mmHg')
            print(f'RPA pressure drop: {rpa_pressure_drops["sys"] / 1333.2} mmHg, {rpa_pressure_drops["dia"] / 1333.2} mmHg, {rpa_pressure_drops["mean"] / 1333.2} mmHg')

            print(f'\nLPA resistance:  sys {lpa_resistance["sys"]} dyn/cm5/s, dia {lpa_resistance["dia"]} dyn/cm5/s, mean {lpa_resistance["mean"]} dyn/cm5/s')
            print(f'RPA resistance: sys {rpa_resistance["sys"]} dyn/cm5/s, dia {rpa_resistance["dia"]} dyn/cm5/s, mean {rpa_resistance["mean"]} dyn/cm5/s')

            print(f'\nLPA PVR: {lpa_resistance["mean"] / 80.0} Wood units')
            print(f'RPA PVR: {rpa_resistance["mean"] / 80.0} Wood units')

            print(f'\nLPA flow: {Q_sys_lpa} dyn/cm5/s, {Q_dia_lpa} dyn/cm5/s, {Q_mean_lpa} dyn/cm5/s')
            print(f'RPA flow: {Q_sys_rpa} dyn/cm5/s, {Q_dia_rpa} dyn/cm5/s, {Q_mean_rpa} dyn/cm5/s')
            
            # compute nonlinear resistance coefficient by fitting resistance vs flows
            S_lpa = np.polyfit([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], [lpa_resistance["sys"], lpa_resistance["dia"], lpa_resistance["mean"]], 1)
            S_rpa = np.polyfit([Q_sys_lpa, Q_dia_rpa, Q_mean_rpa], [rpa_resistance["sys"], rpa_resistance["dia"], rpa_resistance["mean"]], 1)

            # plot the resistance fit
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))

            q = np.linspace(0, 100, 100)

            ax[0].scatter([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], [lpa_resistance["sys"], lpa_resistance["dia"], lpa_resistance["mean"]], label='LPA')
            ax[0].plot(q, np.polyval(S_lpa, q), label='LPA fit')
            ax[0].set_xlabel('Flow (mL/s)')
            ax[0].set_ylabel('Resistance (dyn/cm5/s)')
            ax[0].set_title('LPA flow vs resistance')
            ax[0].legend()

            ax[1].scatter([Q_sys_rpa, Q_dia_rpa, Q_mean_rpa], [rpa_resistance["sys"], rpa_resistance["dia"], rpa_resistance["mean"]], label='RPA')
            ax[1].plot(q, np.polyval(S_rpa, q), label='RPA fit')
            ax[1].set_xlabel('Flow (mL/s)')
            ax[1].set_ylabel('Resistance (dyn/cm5/s)')
            ax[1].set_title('RPA flow vs resistance')
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, 'resistance_fit.png'))

            lpa_resistance = S_lpa[0]
            rpa_resistance = S_rpa[0]



        return lpa_resistance, rpa_resistance

    def generate_simplified_zerod(self, path='simplified_nonlinear_zerod.json', nonlinear=True, optimize=False):
        '''
        compute the simplified 0D model for a 3D pulmonary model from the steady simulation result'''

        if optimize:
            print("Optimizing nonlinear resistance coefficients against 3D result...")
            lpa_resistance, rpa_resistance = self.optimize_nonlinear_resistance('simplified_zerod_config.json')
        else:
            lpa_resistance, rpa_resistance = self.compute_pressure_drop(steady=not nonlinear)

        # need to rescale the inflow and make it periodic with a generic shape (see Inflow class)
        inflow = Inflow.periodic(path=None)
        inflow.rescale(cardiac_output=self.svzerod_3Dcoupling.bcs['INFLOW'].Q[0])

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
                        "R_poiseuille": 1.0 if nonlinear else lpa_resistance / 2,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": lpa_resistance / 2 if nonlinear else 0.0
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
                        "R_poiseuille": 1.0 if nonlinear else lpa_resistance / 2,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": lpa_resistance / 2 if nonlinear else 0.0
                    }
                },
                {
                    "vessel_id": 3,
                    "vessel_length": 1.0,
                    "vessel_name": "branch3_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "R_poiseuille": 1.0 if nonlinear else rpa_resistance / 2,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": rpa_resistance / 2 if nonlinear else 0.0
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
                        "R_poiseuille": 1.0 if nonlinear else rpa_resistance / 2,
                        "C": 0.0,
                        "L": 0.0,
                        "stenosis_coefficient": rpa_resistance / 2 if nonlinear else 0.0
                    }
                }
            ]
        })

        config.to_json(path)
 
    def optimize_nonlinear_resistance(self, tuned_pa_config, initial_guess=[500, 500]):
        '''
        Get the nonlinear resistance coefficients for the LPA and RPA by optimizing against the pressure drop in the unsteady result
        This function assumes that the simulation has been run and the results are available in svZeroD_data
        '''
        if self.svzerod_data is None:
            raise ValueError("svZeroD_data not found. Please run the simulation first.")

        # get the MPA pressure
        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
        time = time[time > time.max() - 1.0]
        pressure = pressure[time.index]
        flow = flow[time.index]

        # get rpa split
        lpa_flow, rpa_flow = self.flow_split()
        rpa_split = sum(rpa_flow['mean'].values()) / (sum(lpa_flow['mean'].values()) + sum(rpa_flow['mean'].values()))

        targets = {'mean': np.mean(pressure) / 1333.2, 'sys': np.max(pressure) / 1333.2, 'dia': np.min(pressure) / 1333.2, 'rpa_split': rpa_split}
        # targets = {'mean': 34, 'sys': 68, 'dia': 8, 'rpa_split': targets['rpa_split']}

        # compute a loss function of a nonlinear resistance model with impedance boundary conditions
        # self.generate_simplified_zerod(nonlinear=True)  # generate the simplified 0D model with nonlinear resistance
        nonlinear_config = ConfigHandler.from_json(tuned_pa_config) # pa config with tuned boundary conditions
        # rescale inflow back up to the original cardiac output
        # nonlinear_config.inflows['INFLOW'].rescale(scalar = 2)

        def loss_function(nonlinear_resistance, targets, nonlinear_config):
            # Update the nonlinear resistance values in the simplified 0D model
            # nonlinear resistance in format [lpa, rpa]
            nonlinear_config.vessel_map[1].stenosis_coefficient = nonlinear_resistance[0] / 2
            nonlinear_config.vessel_map[2].stenosis_coefficient = nonlinear_resistance[0] / 2
            nonlinear_config.vessel_map[3].stenosis_coefficient = nonlinear_resistance[1] / 2
            nonlinear_config.vessel_map[4].stenosis_coefficient = nonlinear_resistance[1] / 2

            result = pysvzerod.simulate(nonlinear_config.config) # Run the simulation with the updated nonlinear resistance

            mpa_result = result[result.name == 'branch0_seg0']
            mpa_result = mpa_result[mpa_result.time > mpa_result.time.max() - 1.0]
            flow = mpa_result.flow_in
            pressure = mpa_result.pressure_in
            mean_pressure = np.mean(pressure) / 1333.2
            sys_pressure = np.max(pressure) / 1333.2
            dia_pressure = np.min(pressure) / 1333.2

            # get rpa split
            rpa_result = result[result.name == 'branch3_seg0']
            rpa_result = rpa_result[rpa_result.time > rpa_result.time.max() - 1.0]
            rpa_flow = rpa_result.flow_in
            rpa_split = np.trapz(rpa_flow, rpa_result.time) / np.trapz(flow, mpa_result.time)

            # compute loss
            lamb = 0.00001  # small constant to penalize large resistances
            loss = (abs((mean_pressure - targets['mean'])/targets['mean']) ** 2 +
                    abs((sys_pressure - targets['sys'])/targets['sys']) ** 2 +
                    abs((dia_pressure - targets['dia'])/targets['dia']) ** 2 +
                    abs((rpa_split - targets['rpa_split'])/targets['rpa_split'] * 1) ** 2 + 
                    lamb * (nonlinear_resistance[0]**2 +  # L2 regularization
                    nonlinear_resistance[1]**2))
            print(f"pressures: {int(sys_pressure * 100) / 100} / {int(dia_pressure * 100) / 100}/{int(mean_pressure * 100) / 100} mmHg, target: {int(targets['sys'] * 100) / 100}/{int(targets['dia'] * 100) / 100}/{int(targets['mean'] * 100) / 100} mmHg")
            print(f"RPA split: {rpa_split}, target: {targets['rpa_split']}")
            print(f"Current nonlinear resistances: LPA = {nonlinear_resistance[0]}, RPA = {nonlinear_resistance[1]}, Loss = {loss}")
            
            return loss

        # initial_guess = self.compute_pressure_drop(steady=False)  # get the initial guess for nonlinear resistance
        print(f"Starting optimization with initial guess for nonlinear resistances: LPA = {initial_guess[0]}, RPA = {initial_guess[1]}")
        bounds = Bounds(lb=[0, 0])  # set bounds for the nonlinear resistances to be positive and non-zero
        result = minimize(loss_function, initial_guess, args=(targets, nonlinear_config),
                          method='Nelder-Mead', options={'disp': True}, bounds=bounds)

        print("Optimization complete.")
        optimized_resistances = result.x
        print(f"Optimized LPA nonlinear resistance: {optimized_resistances[0]}")
        print(f"Optimized RPA nonlinear resistance: {optimized_resistances[1]}")

        # save the config with half of the tuned resistances
        print('saving config with 0.5 * the tuned resistances...')
        nonlinear_config.vessel_map[1].stenosis_coefficient = optimized_resistances[0] / 4
        nonlinear_config.vessel_map[2].stenosis_coefficient = optimized_resistances[0] / 4
        nonlinear_config.vessel_map[3].stenosis_coefficient = optimized_resistances[1] / 4
        nonlinear_config.vessel_map[4].stenosis_coefficient = optimized_resistances[1] / 4

        # rescale inflow back to 500 tsteps
        nonlinear_config.inflows['INFLOW'].rescale(tsteps=500)

        nonlinear_config.to_json(os.path.join(self.path, 'simplified_zerod_tuned.json'))

        return optimized_resistances.tolist()  # return as a list for easier handling
        
        

    def generate_impedance_bcs(self):

        wedge_p = float(input('input wedge pressure (default 6.0): ') or 6.0)
        d_min = float(input('minimum diameter for impedance tree (default 0.01): ') or 0.01)

        if self.zerod_config is not None:
            construct_impedance_trees(self.zerod_config, self.mesh_complete.mesh_surfaces_dir, wedge_pressure=wedge_p, d_min=d_min, convert_to_cm=self.convert_to_cm)

            self.svzerod_3Dcoupling, coupling_blocks = self.zerod_config.generate_threed_coupler(self.path, 
                                                                                                inflow_from_0d=True, 
                                                                                                mesh_complete=self.mesh_complete)
        else:
            # we will need to create the svzerod_3Dcoupling.json file from scratch and choose the inflows
            # self.svzerod_3Dcoupling is a blank config at this point and bc's corresponding to each surface will need to be added

            # add the inflows to the svzerod_3Dcoupling
            tsteps = int(input('number of time steps for inflow (default 512): ') or 512)
            self.svzerod_3Dcoupling.simparams.number_of_time_pts_per_cardiac_cycle = tsteps
            bc_idx = 0
            for vtp in self.mesh_complete.mesh_surfaces.values():
                if 'inflow' in vtp.filename.lower():
                    # need to get inflow path or steady flow rate
                    flow_file_path = input(f'path to flow file for {vtp.filename} OR steady flow rate: ')
                    if os.path.exists(flow_file_path):
                        inflow = Inflow.periodic(flow_file_path, name=vtp.filename.split('.')[0])
                        inflow.rescale(tsteps=tsteps)
                    else:
                        try:
                            flow_rate = float(flow_file_path)
                            inflow = Inflow.steady(flow_rate, name=vtp.filename.split('.')[0])
                            inflow.rescale(tsteps=tsteps)
                        except:
                            print('invalid input, please provide a valid path to a flow file or a steady flow rate')
                            return

                    self.svzerod_3Dcoupling.set_inflow(inflow, vtp.filename.split('.')[0], threed_coupled=False)
                else:
                    # this is not an inflow so we will make an impedance boundary conditiosn for this surface
                    cap_d = (vtp.area / np.pi)**(1/2) * 2

                    if self.convert_to_cm:
                        cap_d = cap_d / 10
                    
                    print(f'generating tree {bc_idx} for cap {vtp.filename} with diameter {cap_d}...')
                    tree = StructuredTree(name=vtp.filename, time=self.svzerod_3Dcoupling.bcs['INFLOW'].t, simparams=self.svzerod_3Dcoupling.simparams)

                    tree.build_tree(initial_d=cap_d, d_min=d_min)

                    # compute the impedance in frequency domain
                    tree.compute_olufsen_impedance()

                    bc_name = f'IMPEDANCE_{bc_idx}'

                    self.svzerod_3Dcoupling.bcs[bc_name] = tree.create_impedance_bc(bc_name, wedge_p * 1333.2)

                    bc_idx += 1

            self.svzerod_3Dcoupling.to_json('blank_edited_config.json')
            self.svzerod_3Dcoupling, coupling_blocks = self.svzerod_3Dcoupling.generate_threed_coupler(self.path, inflow_from_0d=True, mesh_complete=self.mesh_complete)

    def flow_split(self, get_mean=False, verbose=True):
        '''
        get the flow split between the LPA and RPA

        :param get_mean: boolean, whether to return mean flow or sys/dia/mean flow
        
        :return (lpa_flow, rpa_flow)'''

        # get the LPA and RPA boundary conditions based on surface name
        if get_mean:
            lpa_flow = {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0
            }
            rpa_flow = {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0
            }
            for block in self.svzerod_3Dcoupling.coupling_blocks.values():
                if 'inflow' in block.surface.lower():
                    continue
                outlet = self.mesh_complete.mesh_surfaces[block.surface]
                if outlet.lpa:
                    if outlet.lobe == 'upper':
                        lpa_flow['upper'] += self.svzerod_data.get_flow(block)
                    elif outlet.lobe == 'middle':
                        lpa_flow['middle'] += self.svzerod_data.get_flow(block)
                    elif outlet.lobe == 'lower':
                        lpa_flow['lower'] += self.svzerod_data.get_flow(block)
                elif outlet.rpa:
                    if outlet.lobe == 'upper':
                        rpa_flow['upper'] += self.svzerod_data.get_flow(block)
                    elif outlet.lobe == 'middle':
                        rpa_flow['middle'] += self.svzerod_data.get_flow(block)
                    elif outlet.lobe == 'lower':
                        rpa_flow['lower'] += self.svzerod_data.get_flow(block)
           
            # get the total flow
            total_flow = sum(lpa_flow.values()) + sum(rpa_flow.values())
            lpa_pct = math.trunc(sum(lpa_flow.values()) / total_flow * 1000) / 10
            rpa_pct = math.trunc(sum(rpa_flow.values()) / total_flow * 1000) / 10

            # get upper/middle/lower flow split
            if verbose:
                print(f'LPA flow: {sum(lpa_flow.values())} ({lpa_pct}%) | upper: {math.trunc(lpa_flow["upper"] / total_flow * 1000) / 10}% | middle: {math.trunc(lpa_flow["middle"] / total_flow * 1000) / 10}% | lower: {math.trunc(lpa_flow["lower"] / total_flow * 1000) / 10}%')
                print(f'RPA flow: {sum(rpa_flow.values())} ({rpa_pct}%) | upper: {math.trunc(rpa_flow["upper"] / total_flow * 1000) / 10}% | middle: {math.trunc(rpa_flow["middle"] / total_flow * 1000) / 10}% | lower: {math.trunc(rpa_flow["lower"] / total_flow * 1000) / 10}%')
            
            with open(self.results_file, 'a') as f:
                f.write(f'LPA flow: {sum(lpa_flow.values())} ({lpa_pct}%) | upper: {math.trunc(lpa_flow["upper"] / total_flow * 1000) / 10}% | middle: {math.trunc(lpa_flow["middle"] / total_flow * 1000) / 10}% | lower: {math.trunc(lpa_flow["lower"] / total_flow * 1000) / 10}%\n')
                f.write(f'RPA flow: {sum(rpa_flow.values())} ({rpa_pct}%) | upper: {math.trunc(rpa_flow["upper"] / total_flow * 1000) / 10}% | middle: {math.trunc(rpa_flow["middle"] / total_flow * 1000) / 10}% | lower: {math.trunc(rpa_flow["lower"] / total_flow * 1000) / 10}%\n\n')
        
        else:
            # unsteady case, need to compute sys, dia, mean flows
            lpa_flow = {
                "sys": {
                    'upper': 0.0,
                    'middle': 0.0,
                    'lower': 0.0
                },
                "dia": {
                    'upper': 0.0,
                    'middle': 0.0,
                    'lower': 0.0
                },
                "mean": {
                    'upper': 0.0,
                    'middle': 0.0,
                    'lower': 0.0
                }
            }
            rpa_flow = copy.deepcopy(lpa_flow)

            for block in self.svzerod_3Dcoupling.coupling_blocks.values():
                if 'inflow' in block.surface.lower():
                    continue
                outlet = self.mesh_complete.mesh_surfaces[block.surface]
                time, flow, pressure = self.svzerod_data.get_result(block)
                time = time[time > time.max() - 1.0]
                last_half_time = time[time > time.max() - 0.5]
                # use the indices of the time to get the flow
                if outlet.lpa:
                    lpa_flow['sys'][outlet.lobe] += np.max(flow[time.index])
                    lpa_flow['dia'][outlet.lobe] += flow[time.index].iloc[0]
                    lpa_flow['mean'][outlet.lobe] += np.mean(flow[time.index])
                elif outlet.rpa:
                    rpa_flow['sys'][outlet.lobe] += np.max(flow[time.index])
                    rpa_flow['dia'][outlet.lobe] += flow[time.index].iloc[0]
                    rpa_flow['mean'][outlet.lobe] += np.mean(flow[time.index])
        
        return lpa_flow, rpa_flow
    
    def plot_mpa(self, clinical_targets=None, plot_pf_loop=True):
        '''
        plot the MPA pressure
        
        :param clinical_targets: csv of clinical targets'''

        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])

        
        if time.max() > 1.0:
            # remove the 1st period of results
            time = time[time > 1.0]
            print(f'length of time: {len(time)}')
            flow = flow[time.index]
            pressure = pressure[time.index]
        else:
            print('Taking result from 1st period! results may not be converged')
            print(f'length of time: {len(time)}')
            print(f'pressure at t=0.2: {pressure[time[time == 0.2].index].values[0] / 1333.2} mmHg')

        pressure = pressure / 1333.2

        if plot_pf_loop:
            fig, ax = plt.subplots(3, figsize=(10, 10))
        else:
            fig, ax = plt.subplots(2, figsize=(10, 10))

        # plot clinical targets as horizontal lines, if available
        if clinical_targets is not None:
            clinical_targets = ClinicalTargets.from_csv(clinical_targets)
            clinical_systolic = clinical_targets.mpa_p[0]
            clinical_diastolic = clinical_targets.mpa_p[1]
            clinical_mean = clinical_targets.mpa_p[2]

            ax[0].axhline(y=clinical_mean, color='green', linestyle='--', linewidth=1.5, label='Clinical Mean Pressure')
            ax[0].axhline(y=clinical_systolic, color='red', linestyle='--', linewidth=1.5, label='Clinical Systolic Pressure')
            ax[0].axhline(y=clinical_diastolic, color='purple', linestyle='--', linewidth=1.5, label='Clinical Diastolic Pressure')


        ax[0].plot(time, pressure, label='MPA pressure')
        ax[0].set_title("Simulated Pressure Waveform vs. Clinical Targets", fontsize=14)
        ax[0].set_xlabel("Cardiac Cycle (normalized time)", fontsize=12)
        ax[0].set_ylabel("Pressure (mmHg)", fontsize=12)
        ax[0].grid(True, linestyle='--', alpha=0.6)

        # add mean pressure as a horizontal line
        ax[0].axhline(y=np.mean(pressure), color='b', linestyle='-', label='Simulated Mean Pressure')
        ax[0].legend(loc='upper right')

        ax[1].plot(time, flow, label='MPA flow')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Flow (mL/s)')
        ax[1].set_title('MPA flow')
        
        if plot_pf_loop:
            ax[2].plot(flow, pressure, label='MPA pressure vs flow')
            ax[2].set_xlabel('Flow (mL/s)')
            ax[2].set_ylabel('Pressure (mmHg)')
            ax[2].set_title('MPA pressure vs flow')

        plt.tight_layout()
        if plot_pf_loop:
            plt.savefig(os.path.join(self.path, 'figures', 'mpa_w_pf.png'))
        else:
            plt.savefig(os.path.join(self.path, 'figures', 'mpa.png'))

        # get the time over the last period
        time = time[time > time.max() - 1.0]
        # use the indices of the time to get the flow
        pressure = pressure[time.index]

        sys_p = np.max(pressure)
        dias_p = np.min(pressure)
        mean_p = np.mean(pressure)

        print(f'MPA systolic pressure: {sys_p} mmHg')
        print(f'MPA diastolic pressure: {dias_p} mmHg')
        print(f'MPA mean pressure: {mean_p} mmHg')

        with open(self.results_file, 'a') as f:
            f.write(f'MPA systolic pressure: {sys_p} mmHg\n')
            f.write(f'MPA diastolic pressure: {dias_p} mmHg\n')
            f.write(f'MPA mean pressure: {mean_p} mmHg\n\n')

    def plot_outlet(self):
        '''
        plot the outlet pressures'''

        fig, ax = plt.subplots(1, 3, figsize=(10, 10))

        # block = self.svzerod_3Dcoupling.coupling_blocks['RESISTANCE_0']

        for block in self.svzerod_3Dcoupling.coupling_blocks.values():

            time, flow, pressure = self.svzerod_data.get_result(block)

            if 'lpa' in block.surface.lower():
                color='r'
            elif 'rpa' in block.surface.lower():
                color='b'
            else:
                color='g'
                continue


            pressure = pressure / 1333.2

            ax[0].plot(time, pressure, label=f'{block.surface} pressure',color=color)
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Pressure (mmHg)')
            ax[0].set_title(f'outlet pressure')
            ax[0].set_ylim(bottom=0)


            ax[1].plot(time, flow, label=f'{block.surface} flow', color=color)
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Flow (mL/s)')
            ax[1].set_title(f'outlet flows')
            
            ax[2].plot(flow, pressure, label=f'{block.surface} pressure vs flow')
            ax[2].set_xlabel('Flow (mL/s)')
            ax[2].set_ylabel('Pressure (mmHg)')
            ax[2].set_title(f'outlet pressure vs flow')

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'figures', f'outlets.png'))

    def get_pvr(self):
        '''
        get the pulmonary vascular resistance from the svZeroD_data file'''

        # get the MPA pressure
        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
        time = time[time > time.max() - 1.0]
        pressure = pressure[time.index]
        flow = flow[time.index]
        mean_pressure = np.mean(pressure)

        # get rpa flow
        lpa_flow, rpa_flow = self.flow_split()
        rpa_flow = sum(rpa_flow['mean'].values())

        # get lpa flow
        lpa_flow = sum(lpa_flow['mean'].values())

        # get the wedge pressure
        if self.clinical_targets is not None:
            wedge_pressure = self.clinical_targets.wedge_p * 1333.2
        else:
            print('no clinical targets found, using 5.0 mmHg as wedge pressure')
            wedge_pressure = 5.0 * 1333.2
        
        # get the pvr
        pvr = (mean_pressure - wedge_pressure) / (lpa_flow + rpa_flow) / 80.0
        print(f'PVR: {pvr} WU')
        with open(self.results_file, 'a') as f:
            f.write(f'PVR: {pvr} WU\n\n')


    def check_simulation(self, poll_interval=60):
        '''
        check the simulation status'''

        n_procs = self.solver_runscript.nodes * self.solver_runscript.procs_per_node
        started_running = False
        is_completed = False
        while True:
            # check for n_procs folder
            if os.path.exists(os.path.join(self.path, f'{n_procs}-procs')):
                if not started_running:
                    print(f'{self.simname} has started running...')
                    started_running = True
                else:
                    with open(os.path.join(self.path, f'{n_procs}-procs', 'histor.dat'), 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            if str(self.svFSIxml.n_tsteps) in lines[-1].split(' ')[2]:
                                print(f'{self.simname} has completed!')
                                return
            
            time.sleep(poll_interval)

class SimResults(SimulationFile):
    '''
    class to handle 3D simulation results in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the simulation results object'''
        super().__init__(path)
    
    def initialize(self):
        '''
        initialize SimResults object'''

        self.vtus = glob.glob(os.path.join(self.path, 'result*.vtu'))

    def write(self):
        '''
        write the simulation results'''
            
        pass
    
    def do_stuff_with_vtus(self):
        '''
        script with the vtu files using paraview python interface perhaps'''

        pass
    

if __name__ == '__main__':
    '''
    test the simulation directory class code'''
    os.chdir('../../Sheep/cassian/preop')

    sim = SimulationDirectory.from_directory()

    sim.optimize_nonlinear_resistance('../pa_config_test_tuning.json')


    

