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

_DEFAULT_TREE_LRR = 10.0

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

        print(f'\n\n *** INITIALIZING SIMULATION DIRECTORY: {path} *** \n\n')

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

        if os.path.exists(os.path.join(path, mesh_complete)):
            # relative path is specified
            mesh_complete = MeshComplete(os.path.join(path, mesh_complete))
            mesh_complete.rename_vtps()
            print('mesh-complete found and loaded')
        elif os.path.exists(mesh_complete):
            # absolute path is specified
            mesh_complete = MeshComplete(mesh_complete)
            mesh_complete.rename_vtps()
            print('mesh-complete found and loaded')
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
            'n_tsteps': 400,
            'dt': 0.0005,
            'nodes': 2,
            'procs_per_node': 24,
            'memory': 16,
            'hours': 6
        }

        self.write_files(simname='Steady Simulation', user_input=False, sim_config=sim_config)

    @staticmethod
    def _cycle_statistics(signal, dia_window_fraction: float = 0.05):
        """
        Compute systolic (maximum), diastolic (local minimum averaged over a window),
        and mean values for a periodic waveform.

        Parameters
        ----------
        signal : array-like
            Waveform samples across one cardiac cycle.
        dia_window_fraction : float
            Fraction of the cycle length used to average around the minimum; guards
            against noise from a single sample.
        """
        if signal is None:
            return 0.0, 0.0, 0.0

        arr = np.asarray(signal, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, 0.0, 0.0

        sys_val = float(np.max(arr))
        mean_val = float(np.mean(arr))

        min_idx = int(np.argmin(arr))
        window = max(1, int(round(arr.size * dia_window_fraction)))
        half = window // 2
        start = max(0, min_idx - half)
        end = min(arr.size, start + window)
        if end <= start:
            dia_val = float(arr[min_idx])
        else:
            dia_val = float(np.mean(arr[start:end]))

        return sys_val, dia_val, mean_val

    def _compute_pressure_drops(self, get_mean=False):
        lpa_flow, rpa_flow = self.flow_split(get_mean=False)

        # get the MPA mean, systolic, diastolic pressure
        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
        if time.size == 0:
            raise RuntimeError("No MPA data available to compute pressure drops.")

        mask_last_period = time > time.max() - 1.0
        time_last_period = time[mask_last_period]
        pressure_last_period = pressure[mask_last_period]

        if time_last_period.size == 0:
            time_last_period = time
            pressure_last_period = pressure

        if pressure_last_period.size == 0:
            raise RuntimeError("Pressure array is empty after masking for the last period.")

        sys_p, dia_p, mean_p = self._cycle_statistics(pressure_last_period)

        lpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
        rpa_outlet_pressures = {'sys': [], 'dia': [], 'mean': []}
        for block in self.svzerod_3Dcoupling.coupling_blocks.values():
            time, flow, pressure = self.svzerod_data.get_result(block)
            if time.size == 0:
                continue

            mask_last_period = time > time.max() - 1.0
            pressure_last_period = pressure[mask_last_period]
            if pressure_last_period.size == 0:
                pressure_last_period = pressure

            sys_val, dia_val, mean_val = self._cycle_statistics(pressure_last_period)

            if 'lpa' in block.surface.lower():
                lpa_outlet_pressures['sys'].append(sys_val)
                lpa_outlet_pressures['dia'].append(dia_val)
                lpa_outlet_pressures['mean'].append(mean_val)
            elif 'rpa' in block.surface.lower():
                rpa_outlet_pressures['sys'].append(sys_val)
                rpa_outlet_pressures['dia'].append(dia_val)
                rpa_outlet_pressures['mean'].append(mean_val)

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

        print(f'\nLPA flow: {Q_sys_lpa} cm3/s, {Q_dia_lpa} cm3/s, {Q_mean_lpa} cm3/s')
        print(f'RPA flow: {Q_sys_rpa} cm3/s, {Q_dia_rpa} cm3/s, {Q_mean_rpa} cm3/s')

        print(f'\nLPA pressure drop: {lpa_pressure_drops["sys"] / 1333.2} mmHg, {lpa_pressure_drops["dia"] / 1333.2} mmHg, {lpa_pressure_drops["mean"] / 1333.2} mmHg')
        print(f'RPA pressure drop: {rpa_pressure_drops["sys"] / 1333.2} mmHg, {rpa_pressure_drops["dia"] / 1333.2} mmHg, {rpa_pressure_drops["mean"] / 1333.2} mmHg')

        print(f'\nLPA resistance:  sys {lpa_resistance["sys"]} dyn/cm5/s, dia {lpa_resistance["dia"]} dyn/cm5/s, mean {lpa_resistance["mean"]} dyn/cm5/s')
        print(f'RPA resistance: sys {rpa_resistance["sys"]} dyn/cm5/s, dia {rpa_resistance["dia"]} dyn/cm5/s, mean {rpa_resistance["mean"]} dyn/cm5/s')

        print(f'\nLPA PVR: {lpa_resistance["mean"] / 80.0} Wood units')
        print(f'RPA PVR: {rpa_resistance["mean"] / 80.0} Wood units')

        if get_mean:
            return mean_p, Q_mean_lpa, Q_mean_rpa, lpa_resistance['mean'], rpa_resistance['mean']
        else:
            return Q_sys_lpa, Q_dia_lpa, Q_mean_lpa, Q_sys_rpa, Q_dia_rpa, Q_mean_rpa, lpa_resistance, rpa_resistance

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
            Q_sys_lpa, Q_dia_lpa, Q_mean_lpa, Q_sys_rpa, Q_dia_rpa, Q_mean_rpa, lpa_resistance, rpa_resistance = self._compute_pressure_drops(get_mean=False)

            # compute nonlinear resistance coefficient by fitting resistance vs flows
            lpa_flows = np.array([Q_sys_lpa, Q_dia_lpa, Q_mean_lpa], dtype=float)
            lpa_res_vals = np.array([lpa_resistance["sys"], lpa_resistance["dia"], lpa_resistance["mean"]], dtype=float)
            rpa_flows = np.array([Q_sys_lpa, Q_dia_rpa, Q_mean_rpa], dtype=float)
            rpa_res_vals = np.array([rpa_resistance["sys"], rpa_resistance["dia"], rpa_resistance["mean"]], dtype=float)

            S_lpa = np.polyfit(lpa_flows, lpa_res_vals, 1)
            S_rpa = np.polyfit(rpa_flows, rpa_res_vals, 1)

            def _compute_r_squared(x_vals, y_vals, coeffs):
                y_pred = np.polyval(coeffs, x_vals)
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                if np.isclose(ss_tot, 0.0):
                    return 1.0 if np.isclose(ss_res, 0.0) else 0.0
                return 1.0 - ss_res / ss_tot

            lpa_r2 = _compute_r_squared(lpa_flows, lpa_res_vals, S_lpa)
            rpa_r2 = _compute_r_squared(rpa_flows, rpa_res_vals, S_rpa)
            print(f"LPA resistance fit R^2: {lpa_r2:.4f}")
            print(f"RPA resistance fit R^2: {rpa_r2:.4f}")

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

    def compute_nonlinear_resistance_sampled(self, num_samples=50, flow_tolerance=1e-6, plot_name='resistance_fit_sampled.png'):
        '''
        Estimate nonlinear resistance coefficients for the LPA and RPA using a dense sampling of the last cardiac cycle.

        Parameters
        ----------
        num_samples : int
            Number of evenly spaced samples to take across the final cardiac cycle (>=3).
        flow_tolerance : float
            Minimum absolute flow magnitude (in cm3/s) to include in the regression; points below this are skipped.
        plot_name : str
            Filename for the diagnostic plot stored in self.fig_dir.
        '''
        if self.svzerod_data is None:
            raise ValueError("svZeroD_data not found. Please run the simulation first.")
        if self.svzerod_3Dcoupling is None:
            raise ValueError("svzerod_3Dcoupling configuration is required to gather outlet data.")
        if self.mesh_complete is None:
            raise ValueError("mesh_complete data is required to categorize LPA/RPA outlets.")
        if num_samples < 3:
            raise ValueError("num_samples must be at least 3 to perform a linear fit.")

        # pull MPA pressure and restrict to the final period
        try:
            mpa_block = self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0']
        except KeyError as exc:
            raise KeyError("branch0_seg0 MPA block not found in coupling_blocks.") from exc
        time, _, pressure = self.svzerod_data.get_result(mpa_block)
        time = np.asarray(time, dtype=float)
        pressure = np.asarray(pressure, dtype=float)

        if time.size == 0 or pressure.size == 0:
            raise RuntimeError("No MPA data available to compute nonlinear resistance.")

        mask_last_period = time > time.max() - 1.0
        time_last = time[mask_last_period] if mask_last_period.any() else time
        pressure_last = pressure[mask_last_period] if mask_last_period.any() else pressure

        valid = np.isfinite(time_last) & np.isfinite(pressure_last)
        time_last = time_last[valid]
        pressure_last = pressure_last[valid]

        if time_last.size < 2:
            raise RuntimeError("Insufficient MPA samples in the last cardiac cycle.")

        sort_idx = np.argsort(time_last)
        time_last = time_last[sort_idx]
        pressure_last = pressure_last[sort_idx]

        sample_times = np.linspace(time_last.min(), time_last.max(), num_samples)
        mpa_pressure_samples = np.interp(sample_times, time_last, pressure_last)

        # initialize accumulators for outlet data
        lpa_flow_samples = np.zeros(num_samples)
        lpa_pressure_sum = np.zeros(num_samples)
        lpa_pressure_counts = np.zeros(num_samples)

        rpa_flow_samples = np.zeros(num_samples)
        rpa_pressure_sum = np.zeros(num_samples)
        rpa_pressure_counts = np.zeros(num_samples)

        for block in self.svzerod_3Dcoupling.coupling_blocks.values():
            if 'inflow' in block.surface.lower():
                continue

            outlet = self.mesh_complete.mesh_surfaces.get(block.surface)
            if outlet is None:
                continue

            bt, bf, bp = self.svzerod_data.get_result(block)
            bt = np.asarray(bt, dtype=float)
            bf = np.asarray(bf, dtype=float)
            bp = np.asarray(bp, dtype=float)

            if bt.size == 0 or bf.size == 0 or bp.size == 0:
                continue

            outlet_mask = bt > bt.max() - 1.0
            bt_last = bt[outlet_mask] if outlet_mask.any() else bt
            bf_last = bf[outlet_mask] if outlet_mask.any() else bf
            bp_last = bp[outlet_mask] if outlet_mask.any() else bp

            outlet_valid = np.isfinite(bt_last) & np.isfinite(bf_last) & np.isfinite(bp_last)
            bt_last = bt_last[outlet_valid]
            bf_last = bf_last[outlet_valid]
            bp_last = bp_last[outlet_valid]

            if bt_last.size < 2:
                continue

            outlet_sort = np.argsort(bt_last)
            bt_last = bt_last[outlet_sort]
            bf_last = bf_last[outlet_sort]
            bp_last = bp_last[outlet_sort]

            flow_interp = np.interp(sample_times, bt_last, bf_last)
            pressure_interp = np.interp(sample_times, bt_last, bp_last)

            if getattr(outlet, 'lpa', False):
                lpa_flow_samples += flow_interp
                lpa_pressure_sum += pressure_interp
                lpa_pressure_counts += 1.0
            elif getattr(outlet, 'rpa', False):
                rpa_flow_samples += flow_interp
                rpa_pressure_sum += pressure_interp
                rpa_pressure_counts += 1.0

        if not np.any(lpa_pressure_counts):
            raise RuntimeError("No LPA outlet data found to compute resistance.")
        if not np.any(rpa_pressure_counts):
            raise RuntimeError("No RPA outlet data found to compute resistance.")

        lpa_outlet_pressure = np.divide(
            lpa_pressure_sum,
            lpa_pressure_counts,
            out=np.full_like(lpa_pressure_sum, np.nan),
            where=lpa_pressure_counts > 0
        )
        rpa_outlet_pressure = np.divide(
            rpa_pressure_sum,
            rpa_pressure_counts,
            out=np.full_like(rpa_pressure_sum, np.nan),
            where=rpa_pressure_counts > 0
        )

        lpa_pressure_drop = mpa_pressure_samples - lpa_outlet_pressure
        rpa_pressure_drop = mpa_pressure_samples - rpa_outlet_pressure

        lpa_valid = (np.abs(lpa_flow_samples) > flow_tolerance) & np.isfinite(lpa_pressure_drop)
        rpa_valid = (np.abs(rpa_flow_samples) > flow_tolerance) & np.isfinite(rpa_pressure_drop)

        if lpa_valid.sum() < 2:
            raise RuntimeError("Insufficient valid LPA samples above flow_tolerance for regression.")
        if rpa_valid.sum() < 2:
            raise RuntimeError("Insufficient valid RPA samples above flow_tolerance for regression.")

        lpa_resistance_samples = lpa_pressure_drop[lpa_valid] / lpa_flow_samples[lpa_valid]
        rpa_resistance_samples = rpa_pressure_drop[rpa_valid] / rpa_flow_samples[rpa_valid]

        if np.allclose(lpa_flow_samples[lpa_valid], lpa_flow_samples[lpa_valid][0]):
            raise RuntimeError("LPA flow samples lack variation; cannot fit a line.")
        if np.allclose(rpa_flow_samples[rpa_valid], rpa_flow_samples[rpa_valid][0]):
            raise RuntimeError("RPA flow samples lack variation; cannot fit a line.")

        lpa_fit = np.polyfit(lpa_flow_samples[lpa_valid], lpa_resistance_samples, 1)
        rpa_fit = np.polyfit(rpa_flow_samples[rpa_valid], rpa_resistance_samples, 1)

        # diagnostic plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        def _plot(ax, flows, resistances, fit, title):
            ax.scatter(flows, resistances, s=25, alpha=0.7, label='samples')
            q_min, q_max = np.min(flows), np.max(flows)
            if np.isclose(q_min, q_max):
                q_min -= 1.0
                q_max += 1.0
            q = np.linspace(q_min, q_max, 200)
            ax.plot(q, np.polyval(fit, q), color='tab:red', label='linear fit')
            ax.set_xlabel('Flow (cm3/s)')
            ax.set_ylabel('Resistance (dyn/cm5/s)')
            ax.set_title(title)
            ax.legend()

        _plot(axes[0], lpa_flow_samples[lpa_valid], lpa_resistance_samples, lpa_fit, 'LPA flow vs resistance')
        _plot(axes[1], rpa_flow_samples[rpa_valid], rpa_resistance_samples, rpa_fit, 'RPA flow vs resistance')

        plt.tight_layout()
        if self.fig_dir is None:
            raise RuntimeError("fig_dir is not set; cannot save resistance plot.")
        plt.savefig(os.path.join(self.fig_dir, plot_name))
        plt.close(fig)

        print(f'LPA nonlinear resistance coefficient: slope={lpa_fit[0]}, intercept={lpa_fit[1]}')
        print(f'RPA nonlinear resistance coefficient: slope={rpa_fit[0]}, intercept={rpa_fit[1]}')

        return {'slope': lpa_fit[0], 'intercept': lpa_fit[1]}, {'slope': rpa_fit[0], 'intercept': rpa_fit[1]}

    def generate_simplified_zerod(self, path='simplified_nonlinear_zerod.json', nonlinear=True, optimize_nonlin=False, optimize_rri=False):
        '''
        compute the simplified 0D model for a 3D pulmonary model'''

        lpa_resistance = None
        rpa_resistance = None
        lpa_rri_params = None
        rpa_rri_params = None

        if optimize_rri:
            print("Optimizing RRI parameters (stenosis, R, L) against 3D result...")
            rri_result = self.optimize_RRI('simplified_zerod_config.json')
            lpa_rri_params = rri_result['LPA']
            rpa_rri_params = rri_result['RPA']
        elif optimize_nonlin:
            print("Optimizing nonlinear resistance coefficients against 3D result...")
            lpa_resistance, rpa_resistance = self.optimize_nonlinear_resistance('simplified_zerod_config.json')
        else:
            lpa_resistance, rpa_resistance = self.compute_pressure_drop(steady=not nonlinear)

        # need to rescale the inflow and make it periodic with a generic shape (see Inflow class)
        inflow = Inflow.periodic(path=None)
        inflow.rescale(cardiac_output=self.svzerod_3Dcoupling.bcs['INFLOW'].Q[0])

        def _segment_values(resistance, inductance, stenosis):
            return {
                "R_poiseuille": resistance,
                "C": 0.0,
                "L": inductance,
                "stenosis_coefficient": stenosis
            }

        vessel_segment_values = {}
        if optimize_rri and lpa_rri_params is not None and rpa_rri_params is not None:
            lpa_stenosis, lpa_R, lpa_L = lpa_rri_params
            rpa_stenosis, rpa_R, rpa_L = rpa_rri_params

            vessel_segment_values[1] = _segment_values(lpa_R / 2, lpa_L, lpa_stenosis / 2)
            vessel_segment_values[2] = _segment_values(lpa_R / 2, 0.0, lpa_stenosis / 2)
            vessel_segment_values[3] = _segment_values(rpa_R / 2, rpa_L, rpa_stenosis / 2)
            vessel_segment_values[4] = _segment_values(rpa_R / 2, 0.0, rpa_stenosis / 2)
        else:
            # fallback to legacy behavior using either nonlinear coefficients or steady resistances
            if lpa_resistance is None or rpa_resistance is None:
                raise RuntimeError("Unable to determine LPA/RPA resistance values for simplified model.")

            if nonlinear:
                vessel_segment_values[1] = _segment_values(1.0, 0.0, lpa_resistance / 2)
                vessel_segment_values[2] = _segment_values(1.0, 0.0, lpa_resistance / 2)
                vessel_segment_values[3] = _segment_values(1.0, 0.0, rpa_resistance / 2)
                vessel_segment_values[4] = _segment_values(1.0, 0.0, rpa_resistance / 2)
            else:
                vessel_segment_values[1] = _segment_values(lpa_resistance / 2, 0.0, 0.0)
                vessel_segment_values[2] = _segment_values(lpa_resistance / 2, 0.0, 0.0)
                vessel_segment_values[3] = _segment_values(rpa_resistance / 2, 0.0, 0.0)
                vessel_segment_values[4] = _segment_values(rpa_resistance / 2, 0.0, 0.0)

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
                    "zero_d_element_values": vessel_segment_values[1]
                },
                {
                    "boundary_conditions": {
                        "outlet": "LPA_BC"
                    },
                    "vessel_id": 2,
                    "vessel_length": 1.0,
                    "vessel_name": "branch2_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": vessel_segment_values[2]
                },
                {
                    "vessel_id": 3,
                    "vessel_length": 1.0,
                    "vessel_name": "branch3_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": vessel_segment_values[3]
                },
                {
                    "boundary_conditions": {
                        "outlet": "RPA_BC"
                    },
                    "vessel_id": 4,
                    "vessel_length": 1.0,
                    "vessel_name": "branch4_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": vessel_segment_values[4]
                }
            ]
        })

        config.to_json(path)
 
    def optimize_nonlinear_resistance(self, tuned_pa_config, initial_guess=[500, 500], tuning_iter=1):
        '''
        Get the nonlinear resistance coefficients for the LPA and RPA by optimizing against the pressure drop in the unsteady result
        This function assumes that the simulation has been run and the results are available in svZeroD_data
        '''
        if self.svzerod_data is None:
            raise ValueError("svZeroD_data not found. Please run the simulation first.")
        
        cycle_duration = self.svzerod_3Dcoupling.bcs['INFLOW'].t[-1]

        # get the MPA pressure
        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'])
        if time.size == 0:
            raise RuntimeError("No svZeroD results available for nonlinear resistance optimization.")

        mask_last_period = time > time.max() - cycle_duration
        time_last_period = time[mask_last_period]
        pressure_last_period = pressure[mask_last_period]
        flow_last_period = flow[mask_last_period]

        if time_last_period.size == 0:
            time_last_period = time
            pressure_last_period = pressure
            flow_last_period = flow

        # get rpa split
        lpa_flow, rpa_flow = self.flow_split()
        rpa_split = sum(rpa_flow['mean'].values()) / (sum(lpa_flow['mean'].values()) + sum(rpa_flow['mean'].values()))

        pressure_window = pressure_last_period
        targets = {
            'mean': float(np.mean(pressure_window)) / 1333.2,
            'sys': float(np.max(pressure_window)) / 1333.2,
            'dia': float(np.min(pressure_window)) / 1333.2,
            'rpa_split': rpa_split
        }
        # targets = {'mean': 34, 'sys': 68, 'dia': 8, 'rpa_split': targets['rpa_split']}

        # compute a loss function of a nonlinear resistance model with impedance boundary conditions
        # self.generate_simplified_zerod(nonlinear=True)  # generate the simplified 0D model with nonlinear resistance
        nonlinear_config = ConfigHandler.from_json(tuned_pa_config) # pa config with tuned boundary conditions
        # rescale inflow back up to the original cardiac output
        # nonlinear_config.inflows['INFLOW'].rescale(scalar = 2)

        def loss_function(nonlinear_resistance, targets, nonlinear_config, cycle_duration):
            # Update the nonlinear resistance values in the simplified 0D model
            # nonlinear resistance in format [lpa, rpa]
            nonlinear_config.vessel_map[1].stenosis_coefficient = nonlinear_resistance[0] / 2
            nonlinear_config.vessel_map[2].stenosis_coefficient = nonlinear_resistance[0] / 2
            nonlinear_config.vessel_map[3].stenosis_coefficient = nonlinear_resistance[1] / 2
            nonlinear_config.vessel_map[4].stenosis_coefficient = nonlinear_resistance[1] / 2

            result = pysvzerod.simulate(nonlinear_config.config) # Run the simulation with the updated nonlinear resistance

            mpa_result = result[result.name == 'branch0_seg0']
            mpa_result = mpa_result[mpa_result.time > mpa_result.time.max() - cycle_duration]
            flow = mpa_result.flow_in
            pressure = mpa_result.pressure_in
            mean_pressure = np.mean(pressure) / 1333.2
            sys_pressure = np.max(pressure) / 1333.2
            dia_pressure = np.min(pressure) / 1333.2

            # get rpa split
            rpa_result = result[result.name == 'branch3_seg0']
            rpa_result = rpa_result[rpa_result.time > rpa_result.time.max() - cycle_duration]
            rpa_flow = rpa_result.flow_in
            rpa_split = np.trapz(rpa_flow, rpa_result.time) / np.trapz(flow, mpa_result.time)

            # compute loss
            lamb = 1e-10  # small constant to penalize large resistances
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
        result = minimize(loss_function, initial_guess, args=(targets, nonlinear_config, cycle_duration),
                          method='Nelder-Mead', options={'disp': True}, bounds=bounds)

        print("Optimization complete.")
        optimized_resistances = result.x
        print(f"Optimized LPA nonlinear resistance: {optimized_resistances[0]}")
        print(f"Optimized RPA nonlinear resistance: {optimized_resistances[1]}")

        if tuning_iter > 1:
            print(f"Saving config with tuned resistances for tuning iteration {tuning_iter}...")
            scaling_factor = 2
        else:
            # save the config with a quarter of the tuned resistances
            print('saving config with 0.25 * the tuned resistances...')
            scaling_factor = 8
        nonlinear_config.vessel_map[1].stenosis_coefficient = optimized_resistances[0] / scaling_factor
        nonlinear_config.vessel_map[2].stenosis_coefficient = optimized_resistances[0] / scaling_factor
        nonlinear_config.vessel_map[3].stenosis_coefficient = optimized_resistances[1] / scaling_factor
        nonlinear_config.vessel_map[4].stenosis_coefficient = optimized_resistances[1] / scaling_factor

        # rescale inflow back to 500 tsteps
        nonlinear_config.inflows['INFLOW'].rescale(tsteps=500)

        nonlinear_config.to_json(os.path.join(self.path, 'simplified_zerod_tuned.json'))

        return optimized_resistances.tolist()  # return as a list for easier handling


    def optimize_RRI(self, tuned_pa_config, initial_guess=None, tuning_iter=1, output_name='simplified_zerod_tuned_RRI.json', optimizer='Nelder-Mead', nm_iter: int = 1):
        '''
        Optimize the stenosis coefficient, Poiseuille resistance, and inertance for the proximal LPA (vessel 1)
        and RPA (vessel 3) in the simplified PA config so that the 0D result matches the 3D target pressures/flow split.
        Vessels 2 and 4 are forced to zero to isolate the proximal tuning.
        '''
        if self.svzerod_data is None:
            raise ValueError("svZeroD_data not found. Please run the simulation first.")
        
        cycle_duration = self.svzerod_3Dcoupling.bcs['INFLOW'].t[-1]

        time, flow, pressure = self.svzerod_data.get_result(self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0'], cycle_duration=cycle_duration)
        if time.size == 0:
            raise RuntimeError("No svZeroD results available for RRI optimization.")

        mask_last_period = time > time.max() - cycle_duration
        time_last_period = time[mask_last_period]
        pressure_last_period = pressure[mask_last_period]
        flow_last_period = flow[mask_last_period]

        if time_last_period.size == 0:
            time_last_period = time
            pressure_last_period = pressure
            flow_last_period = flow

        lpa_flow, rpa_flow = self.flow_split()
        rpa_split = sum(rpa_flow['mean'].values()) / (sum(lpa_flow['mean'].values()) + sum(rpa_flow['mean'].values()))

        pressure_window = pressure_last_period if pressure_last_period.size else pressure
        targets = {
            'mean': float(np.mean(pressure_window)) / 1333.2,
            'sys': float(np.max(pressure_window)) / 1333.2,
            'dia': float(np.min(pressure_window)) / 1333.2,
            'rpa_split': rpa_split
        }

        rri_config = ConfigHandler.from_json(tuned_pa_config)

        def _total_side_parameters(config):
            lpa_prox = config.vessel_map[1]
            lpa_dist = config.vessel_map.get(2)
            rpa_prox = config.vessel_map[3]
            rpa_dist = config.vessel_map.get(4)

            lpa_total_stenosis = lpa_prox.stenosis_coefficient + (lpa_dist.stenosis_coefficient if lpa_dist else 0.0)
            lpa_total_R = lpa_prox.R + (lpa_dist.R if lpa_dist else 0.0)
            lpa_L = lpa_prox.L

            rpa_total_stenosis = rpa_prox.stenosis_coefficient + (rpa_dist.stenosis_coefficient if rpa_dist else 0.0)
            rpa_total_R = rpa_prox.R + (rpa_dist.R if rpa_dist else 0.0)
            rpa_L = rpa_prox.L

            return (lpa_total_stenosis, lpa_total_R, lpa_L,
                    rpa_total_stenosis, rpa_total_R, rpa_L)

        if initial_guess is None:
            lpa_sten, lpa_R, lpa_L, rpa_sten, rpa_R, rpa_L = _total_side_parameters(rri_config)
            initial_guess = [
                max(lpa_sten, 1e-6),
                100.0,
                10.0,
                max(rpa_sten, 1e-6),
                100.0,
                10.0
            ]

        def _apply_parameters(config, params):
            params = np.maximum(params, 0.0)
            lpa_stenosis, lpa_R, lpa_L, rpa_stenosis, rpa_R, rpa_L = params

            lpa_prox = config.vessel_map[1]
            lpa_dist = config.vessel_map.get(2)
            lpa_split = 0.5 if lpa_dist is not None else 1.0

            lpa_prox.stenosis_coefficient = lpa_stenosis * lpa_split
            lpa_prox.R = lpa_R * lpa_split
            lpa_prox.L = lpa_L

            if lpa_dist is not None:
                lpa_dist.stenosis_coefficient = lpa_stenosis * lpa_split
                lpa_dist.R = lpa_R * lpa_split
                lpa_dist.L = 0.0

            rpa_prox = config.vessel_map[3]
            rpa_dist = config.vessel_map.get(4)
            rpa_split = 0.5 if rpa_dist is not None else 1.0

            rpa_prox.stenosis_coefficient = rpa_stenosis * rpa_split
            rpa_prox.R = rpa_R * rpa_split
            rpa_prox.L = rpa_L

            if rpa_dist is not None:
                rpa_dist.stenosis_coefficient = rpa_stenosis * rpa_split
                rpa_dist.R = rpa_R * rpa_split
                rpa_dist.L = 0.0

            return params

        def loss_function(params, targets, config, cycle_duration):
            params = _apply_parameters(config, params)
            try:
                result = pysvzerod.simulate(config.config)
            except Exception as exc:
                print(f"pysvzerod simulation failed during RRI optimization: {exc}")
                return 1e9

            mpa_result = result[result.name == 'branch0_seg0']
            mpa_result = mpa_result[mpa_result.time > mpa_result.time.max() - cycle_duration]
            if mpa_result.empty:
                return np.inf

            flow = mpa_result.flow_in
            pressure = mpa_result.pressure_in
            mean_pressure = np.mean(pressure) / 1333.2
            sys_pressure = np.max(pressure) / 1333.2
            dia_pressure = np.min(pressure) / 1333.2

            rpa_result = result[result.name == 'branch3_seg0']
            rpa_result = rpa_result[rpa_result.time > rpa_result.time.max() - cycle_duration]
            if rpa_result.empty:
                return np.inf
            rpa_flow = rpa_result.flow_in
            rpa_split = np.trapz(rpa_flow, rpa_result.time) / np.trapz(flow, mpa_result.time)

            lamb = 1e-12
            loss = (
                abs((mean_pressure - targets['mean']) / targets['mean']) ** 2 +
                abs((sys_pressure - targets['sys']) / targets['sys']) ** 2 +
                abs((dia_pressure - targets['dia']) / targets['dia']) ** 2 +
                abs((rpa_split - targets['rpa_split']) / targets['rpa_split']) ** 2 +
                lamb * np.dot(params, params)
            )

            print(
                f"pressures: {int(sys_pressure * 100) / 100} / {int(dia_pressure * 100) / 100}/"
                f"{int(mean_pressure * 100) / 100} mmHg | target: "
                f"{int(targets['sys'] * 100) / 100}/{int(targets['dia'] * 100) / 100}/"
                f"{int(targets['mean'] * 100) / 100} mmHg"
            )
            print(f"RPA split: {rpa_split}, target: {targets['rpa_split']}")
            print(
                "Current params:"
                f" LPA = (stenosis={params[0]}, R={params[1]}, L={params[2]}),"
                f" RPA = (stenosis={params[3]}, R={params[4]}, L={params[5]}), Loss = {loss}"
            )

            return loss

        print("Starting RRI optimization with initial guess:")
        print(f"  LPA -> stenosis={initial_guess[0]}, R={initial_guess[1]}, L={initial_guess[2]}")
        print(f"  RPA -> stenosis={initial_guess[3]}, R={initial_guess[4]}, L={initial_guess[5]}")

        bounds = Bounds(lb=[0.0] * len(initial_guess))
        repeats = nm_iter if optimizer == "Nelder-Mead" else 1
        x_init = initial_guess
        result = None
        for _ in range(max(1, repeats)):
            result = minimize(
                loss_function,
                x_init,
                args=(targets, rri_config, cycle_duration),
                method=optimizer,
                bounds=bounds,
                options={'disp': True}
            )
            x_init = result.x

        optimized_params = np.maximum(result.x, 0.0)
        print("RRI optimization complete.")
        print(
            "Optimized parameters:"
            f" LPA = (stenosis={optimized_params[0]}, R={optimized_params[1]}, L={optimized_params[2]}),"
            f" RPA = (stenosis={optimized_params[3]}, R={optimized_params[4]}, L={optimized_params[5]})"
        )

        _apply_parameters(rri_config, optimized_params)

        if tuning_iter > 1:
            print(f"Saving config after tuning iteration {tuning_iter}...")

        rri_config.inflows['INFLOW'].rescale(tsteps=500)
        output_path = os.path.join(self.path, output_name)
        rri_config.to_json(output_path)

        return {
            'LPA': optimized_params[:3].tolist(),
            'RPA': optimized_params[3:].tolist(),
            'output_config': output_path
        }


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

                    tree.build(initial_d=cap_d, d_min=d_min, lrr=_DEFAULT_TREE_LRR)

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
        def _surface_from_block(block):
            if self.mesh_complete is None or self.mesh_complete.mesh_surfaces is None:
                return None
            return self.mesh_complete.mesh_surfaces.get(block.surface)

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
                outlet = _surface_from_block(block)
                if outlet is None:
                    if verbose:
                        print(f"Skipping {block.surface}: missing mesh surface metadata.")
                    continue
                time, flow, _ = self.svzerod_data.get_result(block)
                if time.size == 0 or flow.size == 0:
                    continue
                mean_flow = float(np.trapz(flow, time))
                if outlet.lpa:
                    if outlet.lobe == 'upper':
                        lpa_flow['upper'] += mean_flow
                    elif outlet.lobe == 'middle':
                        lpa_flow['middle'] += mean_flow
                    elif outlet.lobe == 'lower':
                        lpa_flow['lower'] += mean_flow
                elif outlet.rpa:
                    if outlet.lobe == 'upper':
                        rpa_flow['upper'] += mean_flow
                    elif outlet.lobe == 'middle':
                        rpa_flow['middle'] += mean_flow
                    elif outlet.lobe == 'lower':
                        rpa_flow['lower'] += mean_flow
           
            # get the total flow
            total_flow = sum(lpa_flow.values()) + sum(rpa_flow.values())
            if total_flow <= 0.0:
                if verbose:
                    print("No outlet flow data available to compute split.")
                return lpa_flow, rpa_flow
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
                outlet = _surface_from_block(block)
                if outlet is None:
                    if verbose:
                        print(f"Skipping {block.surface}: missing mesh surface metadata.")
                    continue
                time, flow, pressure = self.svzerod_data.get_result(block)
                if time.size == 0:
                    continue

                mask_last_period = time > time.max() - 1.0
                time_last_period = time[mask_last_period]
                flow_last_period = flow[mask_last_period]

                if time_last_period.size == 0:
                    time_last_period = time
                    flow_last_period = flow

                flow_window = flow_last_period if flow_last_period.size else flow
                if flow_window.size == 0:
                    continue

                peak_flow, dia_flow, mean_flow = self._cycle_statistics(flow_window)

                if outlet.lpa:
                    lpa_flow['sys'][outlet.lobe] += peak_flow
                    lpa_flow['dia'][outlet.lobe] += dia_flow
                    lpa_flow['mean'][outlet.lobe] += mean_flow
                elif outlet.rpa:
                    rpa_flow['sys'][outlet.lobe] += peak_flow
                    rpa_flow['dia'][outlet.lobe] += dia_flow
                    rpa_flow['mean'][outlet.lobe] += mean_flow
        
        return lpa_flow, rpa_flow
    
    def plot_mpa(self, clinical_targets=None, plot_pf_loop=True, last_cycle_only=True, cycle_duration=1.0):
        '''
        plot the MPA pressure
        
        :param clinical_targets: csv of clinical targets'''

        block = self.svzerod_3Dcoupling.coupling_blocks['branch0_seg0']
        time, flow, pressure = self.svzerod_data.get_result(block, 
                                                            last_cycle_only=last_cycle_only, 
                                                            cycle_duration=cycle_duration)
        if time.size == 0 or flow.size == 0 or pressure.size == 0:
            raise RuntimeError("No MPA data available to plot.")

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

        sys_p = float(np.max(pressure))
        dias_p = float(np.min(pressure))
        mean_p = float(np.mean(pressure))

        print(f'MPA systolic pressure: {sys_p} mmHg')
        print(f'MPA diastolic pressure: {dias_p} mmHg')
        print(f'MPA mean pressure: {mean_p} mmHg')

        with open(self.results_file, 'a') as f:
            f.write(f'MPA systolic pressure: {sys_p} mmHg\n')
            f.write(f'MPA diastolic pressure: {dias_p} mmHg\n')
            f.write(f'MPA mean pressure: {mean_p} mmHg\n\n')

    def plot_outlet(self, cycle_duration=1.0):
        '''
        plot the outlet pressures'''

        fig, ax = plt.subplots(1, 3, figsize=(12, 8))

        # block = self.svzerod_3Dcoupling.coupling_blocks['RESISTANCE_0']

        for block in self.svzerod_3Dcoupling.coupling_blocks.values():

            time, flow, pressure = self.svzerod_data.get_result(block, 
                                                                cycle_duration=cycle_duration)
            if time.size == 0:
                continue

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

        for axis in ax:
            axis.legend()

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


    
