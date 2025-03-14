from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.preop import *
from svzerodtrees.inflow import *
import matplotlib.pyplot as plt
import json
import pickle
import copy
import time
import os
import vtk
import math
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET




class SimFile(ABC):
    '''
    abstract super class for simulation files'''

    def __init__(self, path):
        '''
        initialize the simulation file'''

        self.path = os.path.abspath(path)

        self.directory = os.path.dirname(path)

        self.filename = os.path.basename(path)

        if os.path.exists(path):
            self.initialize()
            self.is_written = True
        else:
            self.is_written = False

    @abstractmethod
    def initialize(self):
        '''
        initialize the file from some pre-existing file'''
        raise NotImplementedError

    @abstractmethod
    def write(self):
        '''
        write the file'''
        raise NotImplementedError


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

        # figures directory
        self.fig_dir = fig_dir

        self.convert_to_cm = convert_to_cm

    @classmethod
    def from_directory(cls, path='.', zerod_config=None, mesh_complete='mesh-complete', results_dir=None, convert_to_cm=True, is_pulmonary=True):
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
        else:
            print('mesh-complete not found')
            mesh_complete = None
        
        # check for svZeroD_interface.dat
        svzerod_interface = os.path.join(path, 'svZeroD_interface.dat')
        if os.path.exists(svzerod_interface):
            print('svZeroD_interface.dat found')
            svzerod_interface = SVZeroDInterface(svzerod_interface)
        else:
            print('svZeroD_interface.dat not found')
            svzerod_interface = SVZeroDInterface(svzerod_interface)

        # check for svzerod_3Dcoupling.json
        svzerod_3Dcoupling = os.path.join(path, 'svzerod_3Dcoupling.json')
        if zerod_config is not None and mesh_complete is not None:
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
            svFSIxml = SvFSIxml(svFSIxml)
        else:
            print('svFSI.xml not found')
            svFSIxml = SvFSIxml(svFSIxml)

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

        return cls(path,
                   zerod_config,
                   mesh_complete, 
                   svzerod_interface, 
                   svzerod_3Dcoupling, 
                   svFSIxml, 
                   solver_runscript, 
                   svzerod_data, 
                   results_dir,
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

        self.check_files(verbose=False)

        os.system('clean')
        os.system(f'sbatch {self.solver_runscript.path}')

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
                nodes = int(input('number of nodes (default 4): ') or 4)
                procs_per_node = int(input('number of processors per node ( default 24): ') or 24)
                memory = int(input('memory per node in GB (default 16): ') or 16)
                hours = int(input('number of hours (default 6): ') or 6)
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

    def generate_steady_sim(self, flow_rate=None, wedge_p=None):
        '''
        generate simulation files for a steady simulation'''

        if wedge_p is None:
            wedge_p = float(input('input wedge pressure in mmHg (default 5.0): ') or 5.0) * 1333.2
        else:
            wedge_p = wedge_p * 1333.2

        # add the inflows to the svzerod_3Dcoupling
        tsteps = 2
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
                        "R": 100.0
                    }
                })

                bc_idx += 1

        self.svzerod_3Dcoupling.to_json('blank_zerod_config.json')
        self.svzerod_3Dcoupling, coupling_blocks = self.svzerod_3Dcoupling.generate_threed_coupler(self.path, inflow_from_0d=True, mesh_complete=self.mesh_complete)

        sim_config = {
            'n_tsteps': 100,
            'dt': 0.0005,
            'nodes': 2,
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
            lpa_flow, rpa_flow = self.flow_split()

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
            lpa_flow, rpa_flow = self.flow_split(steady=False)

            # get the MPA mean, systolic, diastolic pressure
            time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
            time = time[time > time.max() - 1.0]
            pressure = pressure[time.index]
            sys_p = np.max(pressure)
            dia_p = np.min(pressure)
            mean_p = np.mean(pressure)

            lpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
            rpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
            for block in self.svzerod_3Dcoupling.coupling_blocks.values():
                if 'lpa' in block.surface.lower():
                    time, flow, pressure = self.svzerod_data.get_result(block)
                    time = time[time > time.max() - 1.0]
                    pressure = pressure[time.index]
                    lpa_outlet_pressures['sys'].append(np.max(pressure))
                    lpa_outlet_pressures['dia'].append(np.min(pressure))
                    lpa_outlet_pressures['mean'].append(np.mean(pressure))
                if 'rpa' in block.surface.lower():
                    time, flow, pressure = self.svzerod_data.get_result(block)
                    time = time[time > time.max() - 1.0]
                    pressure = pressure[time.index]
                    rpa_outlet_pressures['sys'].append(np.max(pressure))
                    rpa_outlet_pressures['dia'].append(np.min(pressure))
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

            print(f'LPA pressure drop: {lpa_pressure_drops["sys"] / 1333.2} mmHg, {lpa_pressure_drops["dia"] / 1333.2} mmHg, {lpa_pressure_drops["mean"] / 1333.2} mmHg')
            print(f'RPA pressure drop: {rpa_pressure_drops["sys"] / 1333.2} mmHg, {rpa_pressure_drops["dia"] / 1333.2} mmHg, {rpa_pressure_drops["mean"] / 1333.2} mmHg')

            print(f'LPA resistance: {lpa_resistance["sys"]} dyn/cm5/s, {lpa_resistance["dia"]} dyn/cm5/s, {lpa_resistance["mean"]} dyn/cm5/s')
            print(f'RPA resistance: {rpa_resistance["sys"]} dyn/cm5/s, {rpa_resistance["dia"]} dyn/cm5/s, {rpa_resistance["mean"]} dyn/cm5/s')

            print(f'LPA flow: {Q_sys_lpa} dyn/cm5/s, {Q_dia_lpa} dyn/cm5/s, {Q_mean_lpa} dyn/cm5/s')
            print(f'RPA flow: {Q_sys_rpa} dyn/cm5/s, {Q_dia_rpa} dyn/cm5/s, {Q_mean_rpa} dyn/cm5/s')
            
            # compute nonlinear resistance coefficient by fitting resistance vs flows
            S_lpa = np.polyfit([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], [lpa_resistance["sys"], lpa_resistance["dia"], lpa_resistance["mean"]], 1)
            S_rpa = np.polyfit([Q_sys_lpa, Q_dia_rpa, Q_mean_rpa], [rpa_resistance["sys"], rpa_resistance["dia"], rpa_resistance["mean"]], 1)
            lpa_resistance = S_lpa[0]
            rpa_resistance = S_rpa[0]

            # plot the resistance fit
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))

            print(f'length of ax is {len(ax)}')

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
            plt.savefig(os.path.join(self.figures_dir, 'resistance_fit.png'))



        return lpa_resistance, rpa_resistance

    def generate_simplified_zerod(self, nonlinear=True):
        '''
        compute the simplified 0D model for a 3D pulmonary model from the steady simulation result'''



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

        config.to_json('simplified_zerod_config.json')
 
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

    def flow_split(self, steady=True, verbose=True):
        '''
        get the flow split between the LPA and RPA
        
        :return (lpa_flow, rpa_flow)'''

        # get the LPA and RPA boundary conditions based on surface name
        if steady:
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
                    lpa_flow['dia'][outlet.lobe] += np.min(flow[time.index])
                    lpa_flow['mean'][outlet.lobe] += np.mean(flow[time.index])
                elif outlet.rpa:
                    rpa_flow['sys'][outlet.lobe] += np.max(flow[time.index])
                    rpa_flow['dia'][outlet.lobe] += np.min(flow[time.index])
                    rpa_flow['mean'][outlet.lobe] += np.mean(flow[time.index])
        
        return lpa_flow, rpa_flow
    
    def plot_mpa(self):
        '''
        plot the MPA pressure'''

        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])

        # remove the 1st period of results
        time = time[time > 1.0]
        print(f'length of time: {len(time)}')
        flow = flow[time.index]
        pressure = pressure[time.index]

        pressure = pressure / 1333.2

        fig, ax = plt.subplots(3, figsize=(10, 10))

        ax[0].plot(time, pressure, label='MPA pressure')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Pressure (mmHg)')
        ax[0].set_title('MPA pressure')

        # add mean pressure as a horizontal line
        ax[0].axhline(y=np.mean(pressure), color='r', linestyle='--', label='mean pressure')
        ax[0].legend()

        ax[1].plot(time, flow, label='MPA flow')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Flow (mL/s)')
        ax[1].set_title('MPA flow')
        
        ax[2].plot(flow, pressure, label='MPA pressure vs flow')
        ax[2].set_xlabel('Flow (mL/s)')
        ax[2].set_ylabel('Pressure (mmHg)')
        ax[2].set_title('MPA pressure vs flow')

        plt.tight_layout()
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
    

class MeshComplete(SimFile):
    '''
    class to handle the mesh complete directory
    '''

    def __init__(self, path, 
                 mesh_surfaces_dir='mesh-surfaces', 
                 volume_mesh='mesh-complete.mesh.vtu', 
                 exterior_mesh='mesh-complete.exterior.vtp', 
                 walls_combined='walls_combined.vtp'):
        '''
        initialize the mesh complete directory'''

        super().__init__(path)

        self.mesh_surfaces_dir = os.path.join(path, mesh_surfaces_dir)

        self.volume_mesh = os.path.join(path, volume_mesh)

        self.walls_combined = VTPFile(os.path.join(path, walls_combined))

        self.exterior_mesh = VTPFile(os.path.join(path, exterior_mesh))


    def initialize(self):
        '''
        get the mesh surfaces in the mesh complete directory as VTP objects'''

        filelist_raw = glob.glob(os.path.join(self.path, 'mesh-surfaces', '*.vtp'))

        filelist = [file for file in filelist_raw if 'wall' not in file]

        filelist.sort()

        # find the inflow vtp
        if 'inflow' in filelist[-1].lower():
            # inflow must be the last element, move to the front
            inflow = filelist.pop(-1)
            filelist.insert(0, inflow)
        elif 'inflow' in filelist[0].lower():
            # inflow is the first element
            pass
        else:
            # find inflow.vtp in the list of files, pop it and move it to the front
            inflow = [file for file in filelist if 'inflow' in file.lower()][0]
            filelist.remove(inflow)
            filelist.insert(0, inflow)
        
        self.mesh_surfaces = {}
        for file in filelist:
            self.mesh_surfaces[os.path.basename(file)] = VTPFile(file)

        self.assign_lobe()
    
    def write(self):
        '''
        write the mesh complete directory'''

        pass

    def scale(self, scale_factor=0.1):
        '''
        scale the mesh complete directory by a scale factor
        '''

        print(f'scaling mesh complete by factor {scale_factor}...')

        # scale the volume mesh
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.volume_mesh)
        reader.Update()

        transform = vtk.vtkTransform()
        transform.Scale(scale_factor, scale_factor, scale_factor)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(transform_filter.GetOutput())
        writer.SetFileName(self.volume_mesh)
        writer.Write()

        # scale the walls combined
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.walls_combined)
        reader.Update()

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(transform_filter.GetOutput())
        writer.SetFileName(self.walls_combined)
        writer.Write()

        # scale the mesh surfaces
        for surface in self.mesh_surfaces:
            surface.scale(scale_factor=scale_factor)

    def assign_lobe_old(self):
        '''
        assign upper, middle or lower lobe location to left and right outlets, except the inlet, based on the center of mass y coourdinate'''

        # get the y coord of lpa and rpa outlets
        lpa_locs = [vtp.get_location()[1] for vtp in self.mesh_surfaces.values() if vtp.lpa]
        rpa_locs = [vtp.get_location()[1] for vtp in self.mesh_surfaces.values() if vtp.rpa]
        
        # get the lobe size (1/3 of the y range)
        lpa_lobe_size = (max(lpa_locs) - min(lpa_locs)) / 3
        rpa_lobe_size = (max(rpa_locs) - min(rpa_locs)) / 3

        # assign outlet lobe location
        for vtp in self.mesh_surfaces.values():
            if vtp.lpa:
                if vtp.get_location()[1] < min(lpa_locs) + lpa_lobe_size:
                    vtp.lobe = 'lower'
                elif vtp.get_location()[1] > max(lpa_locs) - lpa_lobe_size:
                    vtp.lobe = 'upper'
                else:
                    vtp.lobe = 'middle'
            elif vtp.rpa:
                if vtp.get_location()[1] < min(rpa_locs) + rpa_lobe_size:
                    vtp.lobe = 'lower'
                elif vtp.get_location()[1] > max(rpa_locs) - rpa_lobe_size:
                    vtp.lobe = 'upper'
                else:
                    vtp.lobe = 'middle'
        
        # count the number of outlets in each lobe
        lpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'upper'])
        lpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'middle'])
        lpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'lower'])

        rpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'upper'])
        rpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'middle'])
        rpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'lower'])

        print(f'outlets by lobe: LPA upper: {lpa_upper}, middle: {lpa_middle}, lower: {lpa_lower}')
        print(f'outlets by lobe: RPA upper: {rpa_upper}, middle: {rpa_middle}, lower: {rpa_lower}\n')

    def swap_lpa_rpa(self):
        '''
        swap the lpa and rpa outlets
        '''

        for filename in os.listdir(self.mesh_surfaces_dir):
            new_filename = None
            if "LPA" in filename:
                new_filename = filename.replace("LPA", "RPA_")
            elif "RPA" in filename:
                new_filename = filename.replace("RPA", "LPA_")
            
            if new_filename:
                os.rename(os.path.join(self.mesh_surfaces_dir, filename), os.path.join(self.mesh_surfaces_dir, new_filename))
                print(f"Renamed: {filename} to {new_filename}")

        self.initialize()

    def assign_lobe(self):
        '''
        assign lobes by sorting the outlets and taking top 1/3 as upper, middle 1/3 as middle and bottom 1/3 as lower
        '''
        # sort lpa and rpa outlets by y coordinates
        sorted_lpa = sorted([outlet for outlet in self.mesh_surfaces.values() if outlet.lpa], key=lambda x: x.get_location()[1])
        sorted_rpa = sorted([outlet for outlet in self.mesh_surfaces.values() if outlet.rpa], key=lambda x: x.get_location()[1])
        
        # get lobe size (1/3 of the y range)
        lpa_lobe_quarter = len(sorted_lpa) // 4
        rpa_lobe_quarter = len(sorted_rpa) // 4

        # assign outlet lobe location
        for i, vtp in enumerate(sorted_lpa):
            if i < lpa_lobe_quarter:
                vtp.lobe = 'lower'
            elif i < lpa_lobe_quarter * 3:
                vtp.lobe = 'middle'
            else:
                vtp.lobe = 'upper'
        for i, vtp in enumerate(sorted_rpa):
            if i < rpa_lobe_quarter:
                vtp.lobe = 'lower'
            elif i < rpa_lobe_quarter * 3:
                vtp.lobe = 'middle'
            else:
                vtp.lobe = 'upper'

        # count the number of outlets in each lobe
        lpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'upper'])
        lpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'middle'])
        lpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'lower'])
        rpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'upper'])
        rpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'middle'])
        rpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'lower'])

        print(f'outlets by lobe: LPA upper: {lpa_upper}, middle: {lpa_middle}, lower: {lpa_lower}')
        print(f'outlets by lobe: RPA upper: {rpa_upper}, middle: {rpa_middle}, lower: {rpa_lower}\n')



class SVZeroDInterface(SimFile):
    '''
    class to handle the svZeroD_interface.dat file in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the svZeroD_interface object'''
        super().__init__(path)


    def initialize(self):
        '''
        initialize from a pre-existing svZeroD_interface.dat file'''

        with open(self.path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]


        self.interface_library_path = lines[lines.index('interface library path:')+1]

        self.svZeroD_input_file = lines[lines.index('svZeroD input file:')+1]

        self.coupling_block_to_surf_id = {} 
        for line in lines[lines.index('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file):')+1:]:
            if line == '':
                break
            block, surf_id = line.split(' ')
            self.coupling_block_to_surf_id[block] = surf_id
        
        self.initialize_flows = lines[lines.index('Initialize external coupling block flows:')+1]

        self.initial_flow = lines[lines.index('External coupling block initial flows (one number is provided, it is applied to all coupling blocks):')+1]

        self.initialize_pressures = lines[lines.index('Initialize external coupling block pressures:')+1]

        self.initial_pressure = lines[lines.index('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks):')+1]
        

    def write(self,
              threed_coupler_path,
              interface_path='/home/users/ndorn/svZeroDSolver/Release/src/interface/libsvzero_interface.so',
              initialize_flows=0,
              initial_flow=0.0,
              initialize_pressures=1,
              initial_pressure=60.0):
        '''
        write the svZeroD_interface.dat file'''
        
        print('writing svZeroD interface file...')

        threed_coupler = ConfigHandler.from_json(threed_coupler_path, is_pulmonary=False, is_threed_interface=True)

        outlet_blocks = [block.name for block in list(threed_coupler.coupling_blocks.values())]

        with open(self.path, 'w') as ff:
            ff.write('interface library path: \n')
            ff.write(interface_path + '\n\n')

            ff.write('svZeroD input file: \n')
            ff.write(threed_coupler_path + '\n\n')
            
            ff.write('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file): \n')
            for idx, bc in enumerate(outlet_blocks):
                ff.write(f'{bc} {idx}\n')

            ff.write('\n')
            ff.write('Initialize external coupling block flows: \n')
            ff.write(f'{initialize_flows}\n\n')

            ff.write('External coupling block initial flows (one number is provided, it is applied to all coupling blocks): \n')
            ff.write(f'{initial_flow}\n\n')

            ff.write('Initialize external coupling block pressures: \n')
            ff.write(f'{initialize_pressures}\n\n')

            ff.write('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks): \n')
            ff.write(f'{initial_pressure}\n\n')

        self.initialize()

        self.is_written = True


class SvFSIxml(SimFile):
    '''
    class to handle the svFSI.xml file in the simulation directory'''

    def __init__(self, path):
        '''
        initialize the svFSIxml object'''
        super().__init__(path)

    def initialize(self):
        '''
        parse the pre-existing svFSI.xml file'''

        self.xml_tree = ET.parse(self.path)
        self.xml_root = self.xml_tree.getroot()

    def write(self, mesh_complete, scale_factor=1.0, n_tsteps=1000, dt=0.01):
        '''
        write the svFSI.xml file
        
        :param mesh_complete: MeshComplete object'''

        self.n_tsteps = n_tsteps
        self.dt = dt

        print('writing svFSIplus.xml...')

        # generate XML tree
        svfsifile = ET.Element("svMultiPhysicsFile")
        svfsifile.set("version", "0.1")

        # General Simulation Parameters
        gensimparams = ET.SubElement(svfsifile, "GeneralSimulationParameters")

        cont_prev_sim = ET.SubElement(gensimparams, "Continue_previous_simulation")
        cont_prev_sim.text = "false"

        num_spatial_dims = ET.SubElement(gensimparams, "Number_of_spatial_dimensions")
        num_spatial_dims.text = "3"

        num_time_steps = ET.SubElement(gensimparams, "Number_of_time_steps")
        num_time_steps.text = str(n_tsteps)

        time_step_size = ET.SubElement(gensimparams, "Time_step_size")
        time_step_size.text = str(dt)

        spec_radius = ET.SubElement(gensimparams, "Spectral_radius_of_infinite_time_step")
        spec_radius.text = "0.5"

        stop_trigger = ET.SubElement(gensimparams, "Searched_file_name_to_trigger_stop")
        stop_trigger.text = "STOP_SIM"

        save_results_to_vtk = ET.SubElement(gensimparams, "Save_results_to_VTK_format")
        save_results_to_vtk.text = "1"

        name_prefix = ET.SubElement(gensimparams, "Name_prefix_of_saved_VTK_files")
        name_prefix.text = "result"

        increment_vtk = ET.SubElement(gensimparams, "Increment_in_saving_VTK_files")
        increment_vtk.text = "20"

        start_saving_tstep = ET.SubElement(gensimparams, "Start_saving_after_time_step")
        start_saving_tstep.text = "1"

        incrememnt_restart = ET.SubElement(gensimparams, "Increment_in_saving_restart_files")
        incrememnt_restart.text = "10"

        convert_bin_vtk = ET.SubElement(gensimparams, "Convert_BIN_to_VTK_format")
        convert_bin_vtk.text = "0"

        verbose = ET.SubElement(gensimparams, "Verbose")
        verbose.text = "1"

        warning = ET.SubElement(gensimparams, "Warning")
        warning.text = "0"

        debug = ET.SubElement(gensimparams, "Debug")
        debug.text = "0"

        # add mesh
        add_mesh = ET.SubElement(svfsifile, "Add_mesh")
        add_mesh.set("name", "msh")

        msh_file_path = ET.SubElement(add_mesh, "Mesh_file_path")
        msh_file_path.text = mesh_complete.volume_mesh


        for vtp in mesh_complete.mesh_surfaces.values():
            add_face = ET.SubElement(add_mesh, "Add_face")
            add_face.set("name", vtp.filename.split('.')[0])

            face_file_path = ET.SubElement(add_face, "Face_file_path")
            face_file_path.text = vtp.path
        
        add_wall = ET.SubElement(add_mesh, "Add_face")
        add_wall.set("name", "wall")

        mesh_scale_Factor = ET.SubElement(add_mesh, "Mesh_scale_factor")
        mesh_scale_Factor.text = str(scale_factor)

        wall_file_path = ET.SubElement(add_wall, "Face_file_path")
        wall_file_path.text = mesh_complete.walls_combined.path

        # add equation
        add_eqn = ET.SubElement(svfsifile, "Add_equation")
        add_eqn.set("type", "fluid")

        coupled = ET.SubElement(add_eqn, "Coupled")
        coupled.text = "1"

        min_iterations = ET.SubElement(add_eqn, "Min_iterations")
        min_iterations.text = "3"

        max_iterations = ET.SubElement(add_eqn, "Max_iterations")
        max_iterations.text = "10"

        tolerance = ET.SubElement(add_eqn, "Tolerance")
        tolerance.text = "1e-3"

        backflow_stab = ET.SubElement(add_eqn, "Backflow_stabilization_coefficient")
        backflow_stab.text = "0.2"

        density = ET.SubElement(add_eqn, "Density")
        density.text = "1.06"

        viscosity = ET.SubElement(add_eqn, "Viscosity", {"model": "Constant"})
        value = ET.SubElement(viscosity, "Value")
        value.text = "0.04"

        output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
        velocity = ET.SubElement(output, "Velocity")
        velocity.text = "true"

        pressure = ET.SubElement(output, "Pressure")
        pressure.text = "true"

        traction = ET.SubElement(output, "Traction")
        traction.text = "true"

        wss = ET.SubElement(output, "WSS")
        wss.text = "true"

        vorticity = ET.SubElement(output, "Vorticity")
        vorticity.text = "true"

        divergence = ET.SubElement(output, "Divergence")
        divergence.text = "true"

        ls = ET.SubElement(add_eqn, "LS", {"type": "NS"})

        linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
        preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
        preconditioner.text = "fsils"

        ls_max_iterations = ET.SubElement(ls, "Max_iterations")
        ls_max_iterations.text = "10"

        ns_gm_max_iterations = ET.SubElement(ls, "NS_GM_max_iterations")
        ns_gm_max_iterations.text = "3"

        ns_cg_max_iterations = ET.SubElement(ls, "NS_CG_max_iterations")
        ns_cg_max_iterations.text = "500"

        ls_tolerance = ET.SubElement(ls, "Tolerance")
        ls_tolerance.text = "1e-3"

        ns_gm_tolerance = ET.SubElement(ls, "NS_GM_tolerance")
        ns_gm_tolerance.text = "1e-3"

        ns_cg_tolerance = ET.SubElement(ls, "NS_CG_tolerance")
        ns_cg_tolerance.text = "1e-3"

        krylov_space_dim = ET.SubElement(ls, "Krylov_space_dimension")
        krylov_space_dim.text = "50"



        couple_to_svzerod = ET.SubElement(add_eqn, "Couple_to_svZeroD")
        couple_to_svzerod.set("type", "SI")

        # add boundary conditions
        for vtp in mesh_complete.mesh_surfaces.values():
            add_bc = ET.SubElement(add_eqn, "Add_BC")
            add_bc.set("name", vtp.filename.split('.')[0])
            
            typ = ET.SubElement(add_bc, "Type")
            typ.text = "Neu"
            time_dep = ET.SubElement(add_bc, "Time_dependence")
            time_dep.text = "Coupled"
        
        # add wall bc
        add_wall_bc = ET.SubElement(add_eqn, "Add_BC")
        add_wall_bc.set("name", "wall")
        typ = ET.SubElement(add_wall_bc, "Type")
        typ.text = "Dir"
        time_dep = ET.SubElement(add_wall_bc, "Time_dependence")
        time_dep.text = "Steady"
        value = ET.SubElement(add_wall_bc, "Value")
        value.text = "0.0"

        # Create the XML tree
        self.xml_tree = ET.ElementTree(svfsifile)

        ET.indent(self.xml_tree.getroot())

        # def prettify(elem):
        #     """Return a pretty-printed XML string for the Element."""
        #     rough_string = ET.tostring(elem, 'utf-8')
        #     reparsed = xml.dom.minidom.parseString(rough_string)
        #     return reparsed.toprettyxml(indent="  ")
        

        # pretty_xml_str = prettify(svfsifile)

        # print(pretty_xml_str)

        # Write the XML to a file
        with open(self.path, "wb") as file:
            self.xml_tree.write(file, encoding="utf-8", xml_declaration=True)

        self.is_written = True


class SolverRunscript(SimFile):
    '''
    class to handle the solver runscript file in the simulation directory (run_solver.sh)'''

    def __init__(self, path):
        '''
        initialize the solver runscript object'''
        super().__init__(path)

    def initialize(self):
        '''
        initialize the solver runscript object'''

        with open(self.path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        pass

    def write(self, 
              nodes=4, 
              procs_per_node=24, 
              hours=6, 
              memory=16,
              svfsiplus_path='/home/users/ndorn/svMP-procfix/svMP-build/svMultiPhysics-build/bin/svmultiphysics'):
        '''
        write the solver runscript file'''

        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.hours = hours
        self.memory = memory
        self.svfsiplus_path = svfsiplus_path

        print('writing solver runscript file...')

        with open(self.path, 'w') as ff:
            ff.write("#!/bin/bash\n\n")
            ff.write("#name of your job \n")
            ff.write("#SBATCH --job-name=svFlowSolver\n")
            ff.write("#SBATCH --partition=amarsden\n\n")
            ff.write("# Specify the name of the output file. The %j specifies the job ID\n")
            ff.write("#SBATCH --output=svFlowSolver.o%j\n\n")
            ff.write("# Specify the name of the error file. The %j specifies the job ID \n")
            ff.write("#SBATCH --error=svFlowSolver.e%j\n\n")
            ff.write("# The walltime you require for your job \n")
            ff.write(f"#SBATCH --time={hours}:00:00\n\n")
            ff.write("# Job priority. Leave as normal for now \n")
            ff.write("#SBATCH --qos=normal\n\n")
            ff.write("# Number of nodes are you requesting for your job. You can have 24 processors per node \n")
            ff.write(f"#SBATCH --nodes={nodes} \n\n")
            ff.write("# Amount of memory you require per node. The default is 4000 MB per node \n")
            ff.write(f"#SBATCH --mem={memory}G\n\n")
            ff.write("# Number of processors per node \n")
            ff.write(f"#SBATCH --ntasks-per-node={procs_per_node} \n\n")
            ff.write("# Send an email to this address when your job starts and finishes \n")
            ff.write("#SBATCH --mail-user=ndorn@stanford.edu \n")
            ff.write("#SBATCH --mail-type=begin \n")
            ff.write("#SBATCH --mail-type=end \n")
            ff.write("module --force purge\n\n")
            ff.write("ml devel\n")
            ff.write("ml math\n")
            ff.write("ml openmpi\n")
            ff.write("ml openblas\n")
            ff.write("ml boost\n")
            ff.write("ml system\n")
            ff.write("ml x11\n")
            ff.write("ml mesa\n")
            ff.write("ml qt\n")
            ff.write("ml gcc/14.2.0\n")
            ff.write("ml cmake\n\n")
            ff.write(f"srun {svfsiplus_path} svFSIplus.xml\n")
        
        self.is_written = True


class VTPFile(SimFile):
    '''
    class to handle vtp files in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the vtp object'''
        super().__init__(path)

        self.lobe = None # to be assigned later

        if 'lpa' in self.filename.lower():
            self.lpa = True
            self.rpa = False
            self.inflow = False
        elif 'rpa' in self.filename.lower():
            self.rpa = True
            self.lpa = False
            self.inflow = False
        elif 'inflow' or 'mpa' in self.filename.lower():
            self.inflow = True
            self.lpa = False
            self.rpa = False
    
    def initialize(self):
        '''
        initialize the vtp file'''
        self.get_area()

    def write(self):
        '''
        write the vtp file'''

        pass
    
    def get_area(self):
        # with open(infile):
        # print('file able to be opened!')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()
        poly = reader.GetOutputPort()
        masser = vtk.vtkMassProperties()
        masser.SetInputConnection(poly)
        masser.Update()

        self.area = masser.GetSurfaceArea()

    def get_location(self):
        '''
        get the center of mass of the outlet'''
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()
        poly = reader.GetOutputPort()
        com = vtk.vtkCenterOfMass()
        com.SetInputConnection(poly)
        com.Update()

        self.center = com.GetCenter()

        return self.center

    def scale(self, scale_factor=0.1):
        '''
        scale a vtp file from mm to cm (multiply by 0.1) using vtkTransform
        '''

        print(f'scaling {self.filename} by factor {scale_factor}...')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.filename)
        reader.Update()

        # get the area before scaling
        print(f'area before scaling: {self.area}')

        transform = vtk.vtkTransform()
        transform.Scale(scale_factor, scale_factor, scale_factor)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(transform_filter.GetOutput())   
        writer.SetFileName(self.filename)
        writer.Write()

        # get the area after scaling
        self.get_area()
        print(f'area after scaling: {self.area}')


class SvZeroDdata(SimFile):
    '''
    class to handle the svZeroD_data file in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the svZeroD_data object'''
        super().__init__(path)

    def initialize(self):
        '''
        initialize the svZeroD_data object'''
        self.df = pd.read_csv(self.path, sep='\s+')

        self.df.rename({self.df.columns[0]: 'time'}, axis=1, inplace=True)

    def write(self):
        '''
        write the svZeroD_data file'''

        pass

    def get_result(self, block):
        '''
        get the pressure and flow from the svZeroD_data DataFrame for a given CouplingBlock
        
        :returns: time, flow, pressure'''

        if block.location == 'inlet':
            return self.df['time'], self.df[f'flow:{block.name}:{block.connected_block}'], self.df[f'pressure:{block.name}:{block.connected_block}']
        
        elif block.location == 'outlet':
            return self.df['time'], self.df[f'flow:{block.connected_block}:{block.name}'], self.df[f'pressure:{block.connected_block}:{block.name}']

    def get_flow(self, block):
        '''
        integrate the flow at the outlet over the last period
        
        :coupling_block: name of the coupling block
        :block_name: name of the block to integrate the flow over'''

        time, flow, pressure = self.get_result(block)

        # only get times and flows over the last cardiac period 1.0s
        if time.max() > 1.0:
            # unsteady simulation, get last period of the pandas dataframd
            time = time[time > time.max() - 1.0]
            # use the indices of the time to get the flow
            flow = flow[time.index]
            return np.trapz(flow, time)
        else:
            # steady simulation, only get last flow value in the pandas dataframe
            flow = flow.iloc[-1]
            return flow


        
        
    

class SimResults(SimFile):
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


    

