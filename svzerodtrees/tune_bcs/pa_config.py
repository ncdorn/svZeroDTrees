
import copy
import json
import pysvzerod
import numpy as np
from scipy.integrate import trapz
from scipy.optimize import minimize, Bounds
import math
import matplotlib.pyplot as plt

# svzerodtrees imports
from ..io.blocks import Vessel, BoundaryCondition, SimParams
from .clinical_targets import ClinicalTargets
from ..microvasculature import StructuredTree, TreeParameters
from ..io.blocks import Junction
from ..io.utils import get_branch_result
from ..microvasculature.compliance import *
class PAConfig():
    '''
    a class to handle the reduced pa config for boundary condition optimization
    '''

    def __init__(self, 
                 simparams: SimParams, 
                 mpa: list, 
                 lpa_prox: list, 
                 rpa_prox: list, 
                 lpa_dist: Vessel, 
                 rpa_dist: Vessel, 
                 inflow: BoundaryCondition, 
                 wedge_p: float,
                 clinical_targets: ClinicalTargets,
                 steady: bool,
                 compliance_model: ComplianceModel = None):
        '''
        initialize the PAConfig object
        
        :param mpa: dict with MPA config
        :param lpa_prox: list of Vessels with LPA proximal config
        :param rpa_prox: list of Vessels with RPA proximal config
        :param lpa_dist: dict with LPA distal config
        :param rpa_dist: dict with RPA distal config
        :param inflow: dict with inflow config
        :param wedge_p: wedge pressure'''
        self.mpa = mpa
        self.rpa_prox = rpa_prox
        # edit the parameters of the prox rpa, lpa
        self.rpa_prox.length = 10.0
        # self.rpa_prox.stenosis_coefficient = 0.0
        self.lpa_prox = lpa_prox
        self.lpa_prox.length = 10.0
        # self.lpa_prox.stenosis_coefficient = 0.0
        self.rpa_dist = rpa_dist
        self.lpa_dist = lpa_dist
        self.inflow = inflow

        self.simparams = simparams

        self.simparams.number_of_cardiac_cycles = 5

        self.simparams.output_all_cycles = False

        self.clinical_targets = clinical_targets

        self.steady = steady

        self.compliance_model = compliance_model

        self._config = {}
        self.junctions = {}
        self.vessel_map = {}
        self.bcs = {'INFLOW': inflow}
        self.initialize_config_maps()

        # need to initialize boundary conditions


    @classmethod
    def from_config_handler(cls, config_handler, clinical_targets: ClinicalTargets, compliance_model: ComplianceModel = None, steady: bool=True):
        '''
        initialize from a general config handler
        '''
        mpa = copy.deepcopy(config_handler.mpa)
        rpa_prox = copy.deepcopy(config_handler.rpa)
        lpa_prox = copy.deepcopy(config_handler.lpa)
        rpa_dist = Vessel.from_config({
            "boundary_conditions":{
                "outlet": "RPA_BC"
            },
            "vessel_id": 3, # needs to be changed later
            "vessel_length": 10.0,
            "vessel_name": "branch3_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                # "C": 1 / (config_handler.rpa.C_eq ** -1 - config_handler.rpa.C ** -1), # calculates way too large of a capacitance
                "C": 0.0,
                "L": config_handler.rpa.L_eq - config_handler.rpa.L, # L_RPA_distal
                "R_poiseuille": config_handler.rpa.R_eq - config_handler.rpa.R, # R_RPA_distal
                "stenosis_coefficient": 0.0
            }
        })

        lpa_dist = Vessel.from_config({
            "boundary_conditions":{
                "outlet": "LPA_BC"
            },
            "vessel_id": 4, # needs to be changed later
            "vessel_length": 10.0,
            "vessel_name": "branch4_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                # "C": 1 / (config_handler.lpa.C_eq ** -1 - config_handler.lpa.C ** -1), # calculates way too large of a capacitance
                "C": 0.0,
                "L": config_handler.lpa.L_eq - config_handler.lpa.L, # L_LPA_distal
                "R_poiseuille": config_handler.lpa.R_eq - config_handler.lpa.R, # R_LPA_distal
                "stenosis_coefficient": 0.0
            }
        })

        return cls(config_handler.simparams, 
                   mpa, 
                   lpa_prox, 
                   rpa_prox, 
                   lpa_dist, 
                   rpa_dist, 
                   config_handler.bcs["INFLOW"], 
                   config_handler.bcs[list(config_handler.bcs.keys())[1]].values["Pd"],
                   clinical_targets,
                   steady,
                   compliance_model)

    @classmethod
    def from_pa_config(cls, pa_config_handler, clinical_targets: ClinicalTargets, compliance_model: ComplianceModel = None):
        '''
        initialize from a pre-existing pa config handler'''

        return cls(pa_config_handler.simparams,
                     pa_config_handler.mpa,
                     pa_config_handler.lpa,
                     pa_config_handler.rpa,
                     pa_config_handler.vessel_map[2],
                     pa_config_handler.vessel_map[4],
                     pa_config_handler.bcs["INFLOW"],
                     clinical_targets.wedge_p,
                     clinical_targets,
                     steady=False,
                     compliance_model=compliance_model)


    def to_json(self, output_file):
        '''
        write the config to a json file
        '''

        with open(output_file, 'w') as ff:
            json.dump(self.config, ff, indent=4)


    def simulate(self):
        '''
        run the simulation with the current config
        '''

        self.result = pysvzerod.simulate(self.config)

        self.rpa_split = np.mean(self.result[self.result.name=='branch3_seg0']['flow_in']) / (np.mean(self.result[self.result.name=='branch0_seg0']['flow_out']))

        self.P_mpa = [np.max(self.result[self.result.name=='branch0_seg0']['pressure_in']) / 1333.2, np.min(self.result[self.result.name=='branch0_seg0']['pressure_in']) / 1333.2, np.mean(self.result[self.result.name=='branch0_seg0']['pressure_in']) / 1333.2]
    

    def initialize_resistance_bcs(self):
        '''initialize the boundary conditions for the pa config
        '''

        # initialize the inflow
        if self.inflow.Q[1] - self.inflow.Q[0] == 0:
            print('steady inflow, optimizing resistance BCs')
            # assume steady
            self.bcs = {
                "INFLOW": self.inflow,

                "RPA_BC": BoundaryCondition.from_config({
                    "bc_name": "RPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "R": 1000.0,
                        "Pd": self.clinical_targets.wedge_p * 1333.2
                    }
                }),

                "LPA_BC": BoundaryCondition.from_config({
                    "bc_name": "LPA_BC",
                    "bc_type": "RESISTANCE",
                    "bc_values": {
                        "R": 1000.0,
                        "Pd": self.clinical_targets.wedge_p * 1333.2
                    }
                })
            }
        else:
            print('unsteady inflow, optimizing RCR BCs')
            # unsteady, need RCR boundary conditions
            self.bcs = {
                "INFLOW": self.inflow,

                "RPA_BC": BoundaryCondition.from_config({
                    "bc_name": "RPA_BC",
                    "bc_type": "RCR",
                    "bc_values": {
                        "Rp": 100.0,
                        "C": 1e-4,
                        "Rd": 900.0,
                        "Pd": self.clinical_targets.wedge_p * 1333.2
                    }
                }),

                "LPA_BC": BoundaryCondition.from_config({
                    "bc_name": "LPA_BC",
                    "bc_type": "RCR",
                    "bc_values": {
                        "Rp": 100.0,
                        "C": 1e-4,
                        "Rd": 900.0,
                        "Pd": self.clinical_targets.wedge_p * 1333.2
                    }
                })
            }


    def create_impedance_trees(self, lpa_params: TreeParameters, rpa_params: TreeParameters, n_procs):
        '''
        create impedance trees for the LPA and RPA distal vessels

        lpa_d: lpa mean outlet diameter
        rpa_d: rpa mean outlet diameter
        tree_params: dict with keys 'lpa', 'rpa', values list currently [k2, k3, lrr, alpha]
        '''

        self.bcs["INFLOW"] = self.inflow

        self.mpa.bc = {
            "inlet": "INFLOW"
        }
        self.lpa_dist.L = lpa_params.inductance
        self.rpa_dist.L = rpa_params.inductance

        # use same number of tsteps to build tree as simparams
        time_array = np.linspace(0, self.inflow.t[-1], self.simparams.number_of_time_pts_per_cardiac_cycle).tolist()

        self.lpa_tree = StructuredTree(name='lpa_tree', 
                                       time=time_array, 
                                       simparams=self.simparams, 
                                       compliance_model=lpa_params.compliance_model)

        self.lpa_tree.build(initial_d=lpa_params.diameter, 
                                 d_min=lpa_params.d_min, 
                                 lrr=lpa_params.lrr, 
                                 alpha=lpa_params.alpha, 
                                 beta=lpa_params.beta)

        # compute the impedance in frequency domain NEED TO SUB IN COMPLIANCE MODEL
        self.lpa_tree.compute_olufsen_impedance(n_procs=n_procs)

        self.bcs["LPA_BC"] = self.lpa_tree.create_impedance_bc(
            "LPA_BC",
            0,
            self.clinical_targets.wedge_p * 1333.2,
            inductance=lpa_params.inductance,
        )

        self.rpa_tree = StructuredTree(name='rpa_tree', 
                                       time=time_array, 
                                       simparams=self.simparams,
                                       compliance_model=rpa_params.compliance_model)

        self.rpa_tree.build(initial_d=rpa_params.diameter, 
                                 d_min=rpa_params.d_min, 
                                 lrr=rpa_params.lrr, 
                                 alpha=rpa_params.alpha, 
                                 beta=rpa_params.beta)

        # compute the impedance in frequency domain
        self.rpa_tree.compute_olufsen_impedance(n_procs=n_procs)

        self.bcs["RPA_BC"] = self.rpa_tree.create_impedance_bc(
            "RPA_BC",
            1,
            self.clinical_targets.wedge_p * 1333.2,
            inductance=rpa_params.inductance,
        )


    def create_steady_trees(self, lpa_params: TreeParameters, rpa_params: TreeParameters):
        '''
        create trees for steady simulation where we just take the tree resistance
        '''

        self.lpa_tree = StructuredTree(name='lpa_tree', time=self.inflow.t, simparams=None)
        self.lpa_tree.build(initial_d=lpa_params.diameter, 
                                 d_min=lpa_params.d_min, lrr=lpa_params.lrr, alpha=lpa_params.alpha, beta=lpa_params.beta)

        self.bcs["LPA_BC"] = self.lpa_tree.create_resistance_bc("LPA_BC", self.clinical_targets.wedge_p * 1333.2)

        self.rpa_tree = StructuredTree(name='rpa_tree', time=self.inflow.t, simparams=None)
        self.rpa_tree.build(initial_d=rpa_params.diameter, 
                                 d_min=rpa_params.d_min, lrr=rpa_params.lrr, alpha=rpa_params.alpha, beta=rpa_params.beta)

        self.bcs["RPA_BC"] = self.rpa_tree.create_resistance_bc("RPA_BC", self.clinical_targets.wedge_p * 1333.2)

    def update_bcs(self):
        '''
        update the boundary conditions in the config from a change in trees
        '''

        self.bcs["LPA_BC"] = self.lpa_tree.create_resistance_bc("LPA_BC", self.clinical_targets.wedge_p * 1333.2)
        self.bcs["RPA_BC"] = self.rpa_tree.create_resistance_bc("RPA_BC", self.clinical_targets.wedge_p * 1333.2)


    def initialize_config_maps(self):
        '''
        initialize the junctions for the pa config
        '''
        
        # change the vessel ids of the proximal vessels

        self.mpa.id = 0
        self.mpa.name = 'branch0_seg0'

        self.lpa_prox.id = 1
        self.lpa_prox.name = 'branch1_seg0'
        
        self.lpa_dist.id = 2
        self.lpa_dist.name = 'branch2_seg0'
        

        self.rpa_prox.id = 3
        self.rpa_prox.name = 'branch3_seg0'
        
        self.rpa_dist.id = 4
        self.rpa_dist.name = 'branch4' + '_seg0'

        # connect the vessels together
        self.mpa.children = [self.lpa_prox, self.rpa_prox]
        self.lpa_prox.children = [self.lpa_dist]
        self.rpa_prox.children = [self.rpa_dist]

        for vessel in [self.mpa, self.lpa_prox, self.rpa_prox, self.lpa_dist, self.rpa_dist]:
            self.vessel_map[vessel.id] = vessel
        

        for vessel in self.vessel_map.values():
            junction = Junction.from_vessel(vessel)
            if junction is not None:
                self.junctions[junction.name] = junction

        
    def assemble_config(self):
        '''
        assemble the config dict from the config maps
        '''

        # add the boundary conditions
        self._config['boundary_conditions'] = [bc.to_dict() for bc in self.bcs.values()]

        # add the junctions
        self._config['junctions'] = [junction.to_dict() for junction in self.junctions.values()]

        # add the simulation parameters
        self._config['simulation_parameters'] = self.simparams.to_dict()

        # add the vessels
        self._config['vessels'] = [vessel.to_dict() for vessel in self.vessel_map.values()]
        

    def convert_to_cm(self):
        '''
        convert vessel parameters to cm
        '''

        for vessel in self.vessel_map.values():
            vessel.convert_to_cm()
        

    def compute_steady_loss(self, R_guess, fun='L2'):
        '''
        compute loss compared to the steady inflow optimization targets
        :param R_f: list of resistances to put into the config
        '''
        blocks_to_optimize = [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]
        for block, R_g in zip(blocks_to_optimize, R_guess):
            block.R = R_g
        # run the simulation
        self.result = self.simulate()

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        # mpa pressure
        self.P_mpa = get_branch_result(self.result, 'pressure_in', 0, steady=True) /  1333.2 

        # rpa pressure
        self.P_rpa = get_branch_result(self.result, 'pressure_out', 1, steady=True) / 1333.2

        # lpa pressure
        self.P_lpa = get_branch_result(self.result, 'pressure_out', 3, steady=True) / 1333.2


        if fun == 'L2':
            loss = np.sum((self.P_mpa - self.clinical_targets.mpa_p) ** 2) + \
                np.sum((self.P_rpa - self.clinical_targets.rpa_p) ** 2) + \
                np.sum((self.P_lpa - self.clinical_targets.lpa_p) ** 2) + \
                np.sum((self.Q_rpa - self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in blocks_to_optimize]) ** 2) # penalize small resistances

        if fun == 'L1':
            loss = np.sum(np.abs(self.P_mpa - self.clinical_targets.mpa_p)) + \
                np.sum(np.abs(self.P_rpa - self.clinical_targets.rpa_p)) + \
                np.sum(np.abs(self.P_lpa - self.clinical_targets.lpa_p)) + \
                np.sum(np.abs(self.Q_rpa - self.clinical_targets.q_rpa))
        
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss
    

    def compute_unsteady_loss(self, R_guess, fun='L2'):
        '''
        compute unsteady loss by adjusting the resistances in the proximal lpa and rpa'''

        blocks_to_optimize = [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]

        self.lpa_prox.R, self.lpa_prox.C, self.rpa_prox.R, self.rpa_prox.C, self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C = R_guess
        
        # run the simulation
        self.result = self.simulate()

        self.result['time'] = np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        self.Q_rpa = trapz(get_branch_result(self.result, 'flow_in', 3, steady=False), self.result['time'])

        # mpa pressure
        P_mpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_in', 0, steady=False)]
        self.P_mpa = [np.max(P_mpa), np.min(P_mpa), np.mean(P_mpa)] # just systolic and mean pressures

        # rpa pressure
        P_rpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 1, steady=False)]
        self.P_rpa = [np.max(P_rpa), np.min(P_rpa), np.mean(P_rpa)]

        # lpa pressure
        P_lpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 3, steady=False)]
        self.P_lpa = [np.max(P_lpa), np.min(P_lpa), np.mean(P_lpa)]

        p_neg_loss = 0

        for p in self.P_mpa + self.P_rpa + self.P_lpa:
            if p < 0:
                p_neg_loss += 10000

        if fun == 'L2':
            loss = np.sum(np.subtract(self.P_mpa, self.clinical_targets.mpa_p) ** 2) + \
                np.sum(np.subtract(self.P_rpa, self.clinical_targets.rpa_p) ** 2) + \
                np.sum(np.subtract(self.P_lpa, self.clinical_targets.lpa_p) ** 2) + \
                100 * np.sum(np.subtract(self.Q_rpa, self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in [self.lpa_prox, self.rpa_prox]]) ** 2) + p_neg_loss
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss


    def compute_unsteady_loss_nonlin(self, R_guess, fun='L2'):
        '''
        compute unsteady loss by adjusting the stenosis coefficient of the proximal lpa and rpa'''

        self.lpa_prox.stenosis_coefficient, self.lpa_prox.C, self.rpa_prox.stenosis_coefficient, self.rpa_prox.C, self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C = R_guess
        
        # run the simulation
        self.result = self.simulate()

        self.result['time'] = np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                                          self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 3, steady=True)

        self.Q_rpa = trapz(get_branch_result(self.result, 'flow_in', 3, steady=False), self.result['time'])

        # mpa pressure
        P_mpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_in', 0, steady=False)]
        self.P_mpa = [np.max(P_mpa), np.min(P_mpa), np.mean(P_mpa)] # just systolic and mean pressures

        # rpa pressure
        P_rpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 1, steady=False)]
        self.P_rpa = [np.max(P_rpa), np.min(P_rpa), np.mean(P_rpa)]

        # lpa pressure
        P_lpa = [p / 1333.2 for p in get_branch_result(self.result, 'pressure_out', 3, steady=False)]
        self.P_lpa = [np.max(P_lpa), np.min(P_lpa), np.mean(P_lpa)]

        p_neg_loss = 0

        for p in self.P_mpa + self.P_rpa + self.P_lpa:
            if p < 0:
                p_neg_loss += 10000

        if fun == 'L2':
            loss = np.sum(np.subtract(self.P_mpa, self.clinical_targets.mpa_p) ** 2) + \
                np.sum(np.subtract(self.P_rpa, self.clinical_targets.rpa_p) ** 2) + \
                np.sum(np.subtract(self.P_lpa, self.clinical_targets.lpa_p) ** 2) + \
                100 * np.sum(np.subtract(self.Q_rpa, self.clinical_targets.q_rpa) ** 2) + \
                np.sum(np.array([1 / block.R for block in [self.lpa_prox, self.rpa_prox]]) ** 2) + p_neg_loss
        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss


    def optimize(self, steady=True, nonlin=False):
        '''
        optimize the resistances in the pa config
        '''

        # self.to_json('pa_config_pre_opt.json')
        # define optimization bounds [0, inf)
        bounds = Bounds(lb=0, ub=math.inf)

        if steady:
            result = minimize(self.compute_steady_loss, 
                                [obj.R for obj in [self.lpa_prox, self.rpa_prox, self.bcs['LPA_BC'], self.bcs['RPA_BC']]], 
                                method="Nelder-Mead", bounds=bounds)
        else:
            if nonlin:
                initial_guess = [self.lpa_prox.stenosis_coefficient, self.lpa_prox.C, self.rpa_prox.stenosis_coefficient, self.rpa_prox.C, 
                                 self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, 
                                 self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C]
                result = minimize(self.compute_unsteady_loss_nonlin, 
                                initial_guess, 
                                method="Nelder-Mead", bounds=bounds)
            else:
                initial_guess = [self.lpa_prox.R, self.lpa_prox.C, self.rpa_prox.R, self.rpa_prox.C, 
                                self.bcs['LPA_BC'].R, self.bcs['LPA_BC'].C, 
                                self.bcs['RPA_BC'].R, self.bcs['RPA_BC'].C]
                result = minimize(self.compute_unsteady_loss, 
                                    initial_guess, 
                                    method="Nelder-Mead", bounds=bounds)

        print([self.Q_rpa / self.clinical_targets.q, self.P_mpa, self.P_lpa, self.P_rpa])


    def plot_mpa(self, path='pa_config_mpa_plot.png'):
        '''
        plot the mpa pressure and flow
        '''

        fig, axs = plt.subplots(1, 2)

        mpa_result = self.result[self.result.name=='branch0_seg0']

        # plot flow
        axs[0].plot(mpa_result['time'], mpa_result['flow_in'])
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('flow (cm3/s)')

        # plot pressure
        axs[1].plot(mpa_result['time'], mpa_result['pressure_in'] / 1333.2)
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('pressure (mmHg)')

        if path is None:
            plt.show()
        else:
            plt.savefig(path)

            plt.close(fig)


    def plot_outlets(self, path='pa_config_outlets_plot.png'):
        '''
        plot the flow and pressure at the outlets of the pa config
        '''

        fig, axs = plt.subplots(2, 2)

        lpa_result = self.result[self.result.name=='branch2_seg0']
        rpa_result = self.result[self.result.name=='branch4_seg0']

        # plot flow
        axs[0,0].plot(lpa_result['time'], lpa_result['flow_out'])
        axs[0,0].set_xlabel('time (s)')
        axs[0,0].set_ylabel('flow (cm3/s)')
        axs[0,0].set_title('LPA flow')

        axs[0,1].plot(rpa_result['time'], rpa_result['flow_out'])
        axs[0,1].set_xlabel('time (s)')
        axs[0,1].set_ylabel('flow (cm3/s)')
        axs[0,1].set_title('RPA flow')

        # plot pressure
        axs[1,0].plot(lpa_result['time'], lpa_result['pressure_out'] / 1333.2)
        axs[1,0].set_xlabel('time (s)')
        axs[1,0].set_ylabel('pressure (mmHg)')
        axs[1,0].set_title('LPA pressure')

        axs[1,1].plot(rpa_result['time'], rpa_result['pressure_out'] / 1333.2)
        axs[1,1].set_xlabel('time (s)')
        axs[1,1].set_ylabel('pressure (mmHg)')
        axs[1,1].set_title('RPA pressure')

        if path is None:
            plt.show()
        else:
            plt.savefig(path)


    def optimize_rcrs_and_compare(self):
        '''
        create optimized RCRs against impedance trees and compare with the resistance optimization
        '''

        # optimize rcr against lpa tree
        print('optimizing RCR to match LPA')
        Rp_lpa, C_lpa, Rd_lpa = self.lpa_tree.match_RCR_to_impedance()

        # optimize rcr against rpa tree
        print('optimizing RCR to match RPA')
        Rp_rpa, C_rpa, Rd_rpa = self.rpa_tree.match_RCR_to_impedance()

        self.bcs['LPA_BC'] = BoundaryCondition.from_config({
            "bc_name": "LPA_BC",
            "bc_type": "RCR",
            "bc_values": {
                "Rp": Rp_lpa,
                "C": C_lpa,
                "Rd": Rd_lpa,
                "Pd": self.clinical_targets.wedge_p
            }
        })

        self.bcs['RPA_BC'] = BoundaryCondition.from_config({
            "bc_name": "RPA_BC",
            "bc_type": "RCR",
            "bc_values": {
                "Rp": Rp_rpa,
                "C": C_rpa,
                "Rd": Rd_rpa,
                "Pd": self.clinical_targets.wedge_p
            }
        })

        self.simulate()

        print('pa config with RCRs simulated')

        self.plot_mpa('mpa_plot_rcr.png')




        

    @property
    def config(self):
        self.assemble_config()
        return self._config
