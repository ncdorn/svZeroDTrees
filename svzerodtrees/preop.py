import copy
import csv
from pathlib import Path
import numpy as np
import json
import math
from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
from svzerodtrees.post_processing.plotting import *
from svzerodtrees.post_processing.stree_visualization import *
from scipy.optimize import minimize, Bounds
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees.adaptation import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler, Vessel, BoundaryCondition, Junction, SimParams


def optimize_outlet_bcs(input_file,
                        clinical_targets: csv,
                        log_file=None,
                        make_steady=False,
                        steady=True,
                        change_to_R=False,
                        show_optimization=True):
    '''
    optimize the outlet boundary conditions of a 0D model by conjugate gradient method

    :param input_file: 0d solver json input file name string
    :param clinical_targets: clinical targets input csv
    :param log_file: str with path to log_file
    :param make_steady: if the input file has an unsteady inflow, make the inflow steady
    :param steady: False if the input file has unsteady inflow
    :param change_to_R: True if you want to change the input config from RCR to R boundary conditions
    :param show_optimization: True if you want to display a track of the optimization results

    :return preop_config: 0D config with optimized BCs
    :return preop_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values, pressures in mmHg
    q, target_ps, rpa_ps, lpa_ps, wedge_p, rpa_split = get_clinical_targets(clinical_targets, log_file) 
    print(target_ps)

    # load input json as a config dict
    with open(input_file) as ff:
        preop_config = json.load(ff)

    # make inflow steady
    if make_steady:
        make_inflow_steady(preop_config)
        write_to_log(log_file, "inlet BCs converted to steady")

    # change boundary conditions to R
    if change_to_R:
        Pd = convert_RCR_to_R(preop_config)
        write_to_log(log_file, "RCR BCs converted to R, Pd = " + str(Pd))

    # add clinical flow values to the zerod input file
    if steady:
        config_flow(preop_config, q)

    # get resistances from the zerod input file
    if steady:
        resistance = get_resistances(preop_config)
    else:
        rcr = get_rcrs(preop_config)
        
    # initialize the data handlers
    result_handler = ResultHandler.from_config(preop_config)
    config_handler = ConfigHandler(preop_config)

    # scale the inflow
    # objective function value as global variable
    global obj_fun
    obj_fun = [] # for plotting the objective function, maybe there is a better way to do this
    # run zerod simulation to reach clinical targets
    def zerod_optimization_objective(resistances,
                                     input_config=config_handler.config,
                                     target_ps=None,
                                     steady=steady,
                                     lpa_branch= result_handler.lpa_branch, # in general, this should be [1, 2]
                                     rpa_branch = result_handler.rpa_branch,
                                     rpa_split=rpa_split
                                     ):
        '''
        objective function for 0D boundary condition optimization

        :param resistances: list of resistances or RCR values, given by the optimizer
        :param input_config: config of the simulation to be optimized
        :param target_ps: target pressures to optimize against
        :param unsteady: True if the model to be optimized has an unsteady inflow condition
        :param lpa_rpa_branch: lpa and rpa branch ids (should always be [1, 2]) 

        :return: sum of SSE of pressure targets and flow split targets
        '''
        # print("resistances: ", resistances)
        # write the optimization iteration resistances to the config
        if steady:
            write_resistances(input_config, resistances)
        else:
            write_rcrs(input_config, resistances)
            
        zerod_result = run_svzerodplus(input_config)

        # get mean, systolic and diastolic pressures
        mpa_pressures, mpa_sys_p, mpa_dia_p, mpa_mean_p  = get_pressure(zerod_result, branch=0, convert_to_mmHg=True)

        # get the MPA, RPA, LPA flow rates
        q_MPA = get_branch_result(zerod_result, branch=0, data_name='flow_in', steady=steady)
        q_RPA = get_branch_result(zerod_result, branch=rpa_branch, data_name='flow_in', steady=steady)
        q_LPA = get_branch_result(zerod_result, branch=lpa_branch, data_name='flow_in', steady=steady)

        if steady: # take the mean pressure only
            p_diff = abs(target_ps[2] - mpa_mean_p) ** 2
        else: # if unsteady, take sum of squares of mean, sys, dia pressure
            pred_p = np.array([
                -mpa_sys_p,
                -mpa_dia_p,
                -mpa_mean_p
            ])
            print("pred_p: ", pred_p)
            # p_diff = np.sum(np.square(np.subtract(pred_p, target_ps)))
            p_diff = (pred_p[0] - target_ps[0]) ** 2

        # penalty = 0
        # for i, p in enumerate(pred_p):
        #     penalty += loss_function_bound_penalty(p, target_ps[i])
            

        # add flow split to optimization by checking RPA flow against flow split
        RPA_diff = abs((np.mean(q_RPA) - (rpa_split * np.mean(q_MPA)))) ** 2
        
        # minimize sum of pressure SSE and flow split SSE
        min_obj = p_diff + RPA_diff # + penalty
        if show_optimization:
            obj_fun.append(min_obj)
            plot_optimization_progress(obj_fun)

        return min_obj

    # write to log file for debugging
    write_to_log(log_file, "Optimizing preop outlet resistance...")

    # run the optimization algorithm
    if steady:
        result = minimize(zerod_optimization_objective,
                        resistance,
                        args=(config_handler.config, target_ps, steady, result_handler.lpa_branch, result_handler.rpa_branch, rpa_split),
                        method="CG",
                        options={"disp": False},
                        )
    else:
        bounds = Bounds(lb=0, ub=math.inf)
        result = minimize(zerod_optimization_objective,
                          rcr,
                          args=(config_handler.config, target_ps, steady, result_handler.lpa_branch, result_handler.rpa_branch, rpa_split),
                          method="CG",
                          options={"disp": False},
                          bounds=bounds
                          )
        
    log_optimization_results(log_file, result, '0D optimization')
    # write to log file for debugging
    write_to_log(log_file, "Outlet resistances optimized! " + str(result.x))

    R_final = result.x # get the array of optimized resistances
    write_resistances(config_handler.config, R_final)

    

    return config_handler, result_handler


def optimize_pa_bcs(input_file,
                    mesh_surfaces_path,
                    clinical_targets: csv,
                    log_file=None,
                    steady=True,
                    show_optimization=True):
    '''
    optimize the outlet boundary conditions of a pulmonary arterial model by splitting the LPA and RPA
    into two RCR blocks. Using Nelder-Mead optimization method.

    :param input_file: 0d solver json input file name string
    :param clinical_targets: clinical targets input csv
    :param log_file: str with path to log_file
    :param make_steady: if the input file has an unsteady inflow, make the inflow steady
    :param unsteady: True if the input file has unsteady inflow
    :param change_to_R: True if you want to change the input config from RCR to R boundary conditions
    :param show_optimization: True if you want to display a track of the optimization results

    :return optimized_pa_config: 0D pa config with optimized BCs
    :return pa_flow: flow result with optimized preop BCs
    '''

    # get the clinical target values
    clinical_targets = ClinicalTargets.from_csv(clinical_targets, steady=steady)

    clinical_targets.log_clinical_targets(log_file)

    # initialize the data handlers
    config_handler = ConfigHandler.from_json(input_file)
    result_handler = ResultHandler.from_config(config_handler.config)

    pa_config = PAConfig.from_config_handler(config_handler, clinical_targets)

    pa_config.optimize()

    write_to_log(log_file, "*** optimized values ****")
    write_to_log(log_file, "MPA pressure: " + str(pa_config.P_mpa))
    write_to_log(log_file, "RPA pressure: " + str(pa_config.P_rpa))
    write_to_log(log_file, "LPA pressure: " + str(pa_config.P_lpa))
    write_to_log(log_file, "RPA flow split: " + str(pa_config.Q_rpa / clinical_targets.q))

    # get outlet areas
    rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path)
    
    # calculate proportional outlet resistances
    print('total number of outlets: ' + str(len(rpa_info) + len(lpa_info)))
    print('RPA total area: ' + str(sum(rpa_info.values())) + '\n')
    print('LPA total area: ' + str(sum(lpa_info.values())) + '\n')

    # distribute amongst all resistance conditions in the config
    assign_pa_bcs(config_handler, clinical_targets.q, rpa_info, lpa_info, pa_config.bcs["LPA_BC"].R, pa_config.bcs["RPA_BC"].R, clinical_targets.wedge_p)

    return config_handler, result_handler, pa_config


def pa_opt_loss_fcn(R, pa_config, targets):
    '''
    loss function for the PA optimization

    :param R: list of resistances from optimizer
    :param pa_config: pa config dict
    :param targets: optimization targets [Q_RPA, P_MPA, P_RPA, P_LPA]

    :return loss: sum of squares of differences between targets and model evaluation
    '''

    # write the optimization iteration resistances to the config
    write_pa_config_resistances(pa_config, R)

    # run the 0D solver
    pa_result = run_svzerodplus(pa_config)

    # get the result values to compare against targets [Q_rpa, P_mpa, P_rpa, P_lpa]
    pa_eval = get_pa_optimization_values(pa_result)

    loss = np.sum((pa_eval - targets) ** 2)

    # print for debugging
    # print(loss, pa_eval, targets)
    return loss


def assign_pa_bcs(config_handler, q_in, rpa_info, lpa_info, R_lpa, R_rpa, wedge_p=13332.2):
    '''
    assign resistances proportional to outlet area to the RPA and LPA outlet bcs.
    this assumes that the rpa and lpa cap info has not changed info since export from simvascular.
    In the case of AS1, this is LPA outlets first, and then RPA (alphabetical). This will also convert all outlet BCs to resistance BCs.

    :param config: svzerodplus config dict
    :param rpa_info: dict with rpa outlet info from vtk
    :param lpa_info: dict with lpa outlet info from vtk
    :param R_rpa: RPA outlet resistance value
    :param R_lpa: LPA outlet resistance value
    '''

    def Ri(Ai, A, R):
        return R * (A / Ai)
    
    # get RPA and LPA total area
    a_RPA = sum(rpa_info.values())
    a_LPA = sum(lpa_info.values())

    # initialize list of resistances
    all_R = {}

    for name, val in lpa_info.items():
        all_R[name] = Ri(val, a_LPA, R_lpa)
    
    for name, val in rpa_info.items():
        all_R[name] = Ri(val, a_RPA, R_rpa)
    
    # write the resistances to the config
    bc_idx = 0

    # get all resistance values
    R_list = list(all_R.values())

    # set the inflow
    config_handler.set_inflow(q_in)

    # loop through boundary conditions to assign resistance values
    for bc in config_handler.bcs.values():
        if bc.type == 'RESISTANCE':
            bc.values['R'] = R_list[bc_idx]
            bc.values['Pd'] = wedge_p

            bc_idx += 1


def construct_cwss_trees(config_handler, result_handler: ResultHandler, log_file=None, d_min=0.0049, vis_trees=False, fig_dir=None):
    '''
    construct structured trees at every outlet of the 0d model optimized against the outflow BC resistance,
    for the constant wall shear stress assumption.

    :param config: 0D solver config
    :param result: 0D solver result corresponding to config
    :param log_file: optional path to a log file
    :param d_min: minimum vessel diameter for tree optimization
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.

    :return roots: return the root TreeVessel objects of the outlet trees

    '''
    num_outlets = len(config_handler.config["boundary_conditions"]) - 2
    outlet_idx = 0

    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                print("** building tree for outlet " + str(outlet_idx) + " of " + str(num_outlets) + " **, d_min = " + str(d_min) + " **")
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        print(bc_config["bc_name"])
                        outlet_tree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                            config_handler.config["simulation_parameters"],
                                                                            bc_config)

                        R = bc_config["bc_values"]["R"]

                        # write to log file for debugging
                        write_to_log(log_file, "** building tree for resistance: " + str(R) + " **")

                        outlet_tree.optimize_tree_diameter(R, log_file, d_min=d_min)

                        # replace the bc resistance with the optimized value as it may be different than the initial value
                        bc_config["bc_values"]["R"] = outlet_tree.root.R_eq
                        print(' the equivalent resistance being added as the outlet boundary condition is ' + str(outlet_tree.root.R_eq))

                # write to log file for debugging
                write_to_log(log_file, "     the number of vessels is " + str(outlet_tree.count_vessels()))

                config_handler.trees.append(outlet_tree)
                outlet_idx += 1

    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTreeOutlet instances based on the preop result
    # config_handler.update_stree_hemodynamics(preop_result)

    result_handler.add_unformatted_result(preop_result, 'preop')


def construct_pries_trees(config_handler, result_handler, log_file=None, d_min=0.0049, tol=0.01, vis_trees=False, fig_dir=None):
    '''
    construct trees for pries and secomb adaptation and perform initial integration
    :param config: 0D solver preop config
    :param result: 0D solver result corresponding to config
    :param ps_params: Pries and Secomb parameters in the form [k_p, k_m, k_c, k_s, L (cm), S_0, tau_ref, Q_ref], default are from Pries et al. 2001
        units:
            k_p, k_m, k_c, k_s [=] dimensionless
            L [=] cm
            S_0 [=] dimensionless
            tau_ref [=] dyn/cm2
            Q_ref [=] cm3/s
    :param log_file: optional path to a log file
    :param d_min: minimum vessel diameter for tree optimization
    :param tol: tolerance for the pries and secomb integration
    :param vis_trees: boolean for visualizing trees
    :param fig_dir: [optional path to directory to save figures. Required if vis_trees = True.
    '''
    simparams = config_handler.config["simulation_parameters"]
    num_outlets = len(config_handler.config["boundary_conditions"])

    # compute a pretree result to use to optimize the trees
    pretree_result = run_svzerodplus(config_handler.config)

    # get the outlet flowrate
    q_outs = get_outlet_data(config_handler.config, pretree_result, "flow_out", steady=True)
    p_outs = get_outlet_data(config_handler.config, pretree_result, "pressure_out", steady=True)
    outlet_idx = 0 # need this when iterating through outlets 
    # get the outlet vessel
    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                               simparams,
                                                                               bc_config, 
                                                                               Q_outlet=[np.mean(q_outs[outlet_idx])],
                                                                               P_outlet=[np.mean(p_outs[outlet_idx])])
                        R = bc_config["bc_values"]["R"]

                        write_to_log(log_file, "** building tree " + str(outlet_idx) + " for R = " + str(R) + " **")

                        outlet_stree.optimize_tree_diameter(R, log_file, d_min=d_min, pries_secomb=True)

                        bc_config["bc_values"]["R"] = outlet_stree.root.R_eq

                        write_to_log(log_file, "     the number of vessels is " + str(outlet_stree.count_vessels()))

                config_handler.trees.append(outlet_stree)
                outlet_idx += 1

    # compute the preop result
    preop_result = run_svzerodplus(config_handler.config)

    # leaving vessel radius fixed, update the hemodynamics of the StructuredTreeOutlet instances based on the preop result
    config_handler.update_stree_hemodynamics(preop_result)

    # add the preop result to the result handler
    result_handler.add_unformatted_result(preop_result, 'preop')

    
def optimize_ps_params():
    '''
    method to optimize the pries and secomb parameters to compare with Ingrid's. To be implemented
    '''
    pass



class ClinicalTargets():
    '''
    class to handle clinical target values
    '''

    def __init__(self, mpa_p, lpa_p, rpa_p, q, rpa_split, wedge_p, steady):
        '''
        initialize the clinical targets object
        '''
        
        self.mpa_p = mpa_p
        self.lpa_p = lpa_p
        self.rpa_p = rpa_p
        self.q = q
        self.rpa_split = rpa_split
        self.q_rpa = q * rpa_split
        self.wedge_p = wedge_p
        self.steady = steady


    @classmethod
    def from_csv(cls, clinical_targets: csv, steady=True):
        '''
        initialize from a csv file
        '''
        # get the flowrate
        bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
        cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
        q = bsa * cardiac_index * 16.667 # cardiac output in L/min. convert to cm3/s

        # get the mpa pressures
        mpa_pressures = get_value_from_csv(clinical_targets, 'mpa pressures') # mmHg
        mpa_sys_p, mpa_dia_p = mpa_pressures.split("/")
        mpa_mean_p = int(get_value_from_csv(clinical_targets, 'mpa mean pressure'))

        # get the lpa pressures
        lpa_pressures = get_value_from_csv(clinical_targets, 'lpa pressures') # mmHg
        lpa_sys_p, lpa_dia_p = lpa_pressures.split("/")
        lpa_mean_p = int(get_value_from_csv(clinical_targets, 'lpa mean pressure'))

        # get the rpa pressures
        rpa_pressures = get_value_from_csv(clinical_targets, 'rpa pressures') # mmHg
        rpa_sys_p, rpa_dia_p = rpa_pressures.split("/")
        rpa_mean_p = int(get_value_from_csv(clinical_targets, 'rpa mean pressure'))

        # if steady, just take the mean
        if steady:
            mpa_p = mpa_mean_p
            lpa_p = lpa_mean_p
            rpa_p = rpa_mean_p
        else:
            mpa_p = [mpa_sys_p, mpa_dia_p, mpa_mean_p]
            lpa_p = [lpa_sys_p, lpa_dia_p, lpa_mean_p]
            rpa_p = [rpa_sys_p, rpa_dia_p, rpa_mean_p]

        # get wedge pressure
        wedge_p = int(get_value_from_csv(clinical_targets, 'wedge pressure'))

        # get RPA flow split
        rpa_split = float(get_value_from_csv(clinical_targets, 'pa flow split')[0:2]) / 100

        return cls(mpa_p, lpa_p, rpa_p, q, rpa_split, wedge_p, steady=steady)

        
    def log_clinical_targets(self, log_file):

        write_to_log(log_file, "*** clinical targets ****")
        write_to_log(log_file, "Q: " + str(self.q))
        write_to_log(log_file, "MPA pressures: " + str(self.mpa_p))
        write_to_log(log_file, "RPA pressures: " + str(self.rpa_p))
        write_to_log(log_file, "LPA pressures: " + str(self.lpa_p))
        write_to_log(log_file, "wedge pressure: " + str(self.wedge_p))
        write_to_log(log_file, "RPA flow split: " + str(self.rpa_split))


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
                 clinical_targets: ClinicalTargets):
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
        self.lpa_prox = lpa_prox
        self.rpa_dist = rpa_dist
        self.lpa_dist = lpa_dist

        self.simparams = simparams

        self.clinical_targets = clinical_targets

        self._config = {}
        self.junctions = {}
        self.vessel_map = {}
        self.bcs = {}
        self.initialize_config_maps()


        self.initalize_bcs(inflow, wedge_p)


    @classmethod
    def from_config_handler(cls, config_handler, clinical_targets: ClinicalTargets):
        '''
        initialize from a general config handler
        '''
        mpa = config_handler.get_segments('mpa')
        rpa_prox = config_handler.get_segments('rpa')
        lpa_prox = config_handler.get_segments('lpa')
        rpa_dist = Vessel.from_config({
            "boundary_conditions":{
                "outlet": "RPA_BC"
            },
            "vessel_id": 3, # needs to be changed later
            "vessel_length": 10.0,
            "vessel_name": "branch3_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": 1 / (config_handler.rpa.C_eq ** -1 - config_handler.rpa.C ** -1), # C_RPA_distal
                "L": config_handler.rpa.L_eq - config_handler.rpa.L, # L_RPA_distal
                "R_poiseuille": config_handler.rpa.R_eq - config_handler.rpa.zero_d_element_values.get("R_poiseuille"), # R_RPA_distal
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
                "C": 1 / (config_handler.lpa.C_eq ** -1 - config_handler.lpa.C ** -1), # C_LPA_distal
                "L": config_handler.lpa.L_eq - config_handler.lpa.L, # L_LPA_distal
                "R_poiseuille": config_handler.lpa.R_eq - config_handler.lpa.zero_d_element_values.get("R_poiseuille"), # R_LPA_distal
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
                   config_handler.bcs['RCR_0'].values["Pd"],
                   clinical_targets)
    

    def to_json(self, output_file):
        '''
        write the config to a json file
        '''

        with open(output_file, 'w') as ff:
            json.dump(self.config, ff)


    def simulate(self):
        '''
        run the simulation with the current config
        '''

        return run_svzerodplus(self.config)
    

    def initalize_bcs(self, inflow: BoundaryCondition, wedge_p: float):
        '''initialize the boundary conditions for the pa config
        '''

        # initialize the inflow
        self.bcs = {
            "INFLOW": inflow,

            "RPA_BC": BoundaryCondition.from_config({
                "bc_name": "RPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {
                    "R": 100.0,
                    "Pd": wedge_p
                }
            }),

            "LPA_BC": BoundaryCondition.from_config({
                "bc_name": "LPA_BC",
                "bc_type": "RESISTANCE",
                "bc_values": {
                    "R": 100.0,
                    "Pd": wedge_p
                }
            })
        }


    def initialize_config_maps(self):
        '''
        initialize the junctions for the pa config
        '''
        
        # change the vessel ids of the proximal vessels
        for vessel in self.lpa_prox:
            vessel.id = len(self.mpa) + self.lpa_prox.index(vessel)
            vessel.name = 'branch1' + '_seg' + str(self.lpa_prox.index(vessel))
        
        self.lpa_dist.id = self.lpa_prox[-1].id + 1
        self.lpa_dist.name = 'branch2' + '_seg0'
        
        for vessel in self.rpa_prox:
            vessel.id = self.lpa_dist.id + self.rpa_prox.index(vessel) + 1
            vessel.name = 'branch3' + '_seg' + str(self.rpa_prox.index(vessel))
        
        self.rpa_dist.id = self.rpa_prox[-1].id + 1
        self.rpa_dist.name = 'branch4' + '_seg0'

        # connect the vessels together
        self.mpa[-1].children = [self.lpa_prox[0], self.rpa_prox[0]]
        self.lpa_prox[-1].children = [self.lpa_dist]
        self.rpa_prox[-1].children = [self.rpa_dist]

        for vessel in self.mpa:
            self.vessel_map[vessel.id] = vessel
        
        for vessel in self.lpa_prox:
            self.vessel_map[vessel.id] = vessel
        
        for vessel in self.rpa_prox:
            self.vessel_map[vessel.id] = vessel
        
        self.vessel_map[self.lpa_dist.id] = self.lpa_dist
        self.vessel_map[self.rpa_dist.id] = self.rpa_dist

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
        

    def compute_steady_loss(self, R_guess, fun='SSE'):
        '''
        compute loss compared to the steady inflow optimization targets
        :param R_f: list of resistances to put into the config
        '''
        for obj, R_g in zip(self.lpa_prox + self.rpa_prox + [self.bcs['LPA_BC']] + [self.bcs['RPA_BC']], R_guess):
            obj.R = R_g
        # run the simulation
        self.result = self.simulate()

        # get the pressures
        # rpa flow, for flow split optimization
        self.Q_rpa = get_branch_result(self.result, 'flow_in', 1, steady=True)

        # mpa pressure
        self.P_mpa = get_branch_result(self.result, 'pressure_in', 0, steady=True) /  1333.2 

        # rpa pressure
        self.P_rpa = get_branch_result(self.result, 'pressure_out', 1, steady=True) / 1333.2

        # lpa pressure
        self.P_lpa = get_branch_result(self.result, 'pressure_out', 3, steady=True) / 1333.2


        if fun == 'SSE':
            loss = np.sum((self.P_mpa - self.clinical_targets.mpa_p) ** 2) + \
                np.sum((self.P_rpa - self.clinical_targets.rpa_p) ** 2) + \
                np.sum((self.P_lpa - self.clinical_targets.lpa_p) ** 2) + \
                np.sum((self.Q_rpa - self.clinical_targets.q_rpa) ** 2)

        print('R_guess: ' + str(R_guess)) 
        print('loss: ' + str(loss))

        return loss
    

    def compute_unsteady_loss(self, fun='SSE'):

        pass


    def optimize(self):
        '''
        optimize the resistances in the pa config
        '''

        self.to_json('pa_config_pre_opt.json')
        # define optimization bounds [0, inf)
        bounds = Bounds(lb=0, ub=math.inf)

        print([obj.R for obj in self.lpa_prox + self.rpa_prox + [self.bcs['LPA_BC']] + [self.bcs['RPA_BC']]])

        result = minimize(self.compute_steady_loss, 
                          [obj.R for obj in self.lpa_prox + self.rpa_prox + [self.bcs['LPA_BC']] + [self.bcs['RPA_BC']]], 
                          method="Nelder-Mead", bounds=bounds)

        print([self.Q_rpa / self.clinical_targets.q, self.P_mpa, self.P_rpa, self.P_lpa])

    @property
    def config(self):
        self.assemble_config()
        return self._config

        

