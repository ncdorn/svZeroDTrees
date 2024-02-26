from svzerodtrees.utils import *
import copy
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler


def adapt_pries_secomb(config_handler: ConfigHandler, result_handler: ResultHandler, log_file: str = None, tol: float = .01):
    '''
    adapt structured tree microvasculature model based on Pries et al. 1998

    :param postop_config: config dict from the stenosis repair
    :param trees: list of StructuredTreeOutlet instances, corresponding to the outlets of the 0D model
    :param preop_result: preoperative result array
    :param postop_result: postoperative result array, calculated after the stenosis repair
    :param log_file: path to log file, for writing important messages for debugging purposes

    :return: config dict post-adaptation, flow result post-adaptation, list of StructuredTreeOutlet instances
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

    # loop through the vessels and create StructuredTreeOutlet instances at the outlets, from the pre-adaptation tree instances
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

    :param postop_config: config dict from the stenosis repair, with StructuredTreeOutlet instances at the outlets
    :param trees: list of StructuredTreeOutlet instances, corresponding to the outlets of the 0D model
    :param preop_result: preoperative result array
    :param postop_result: postoperative result array, calculated after the stenosis repair
    :param log_file: path to log file, for writing important messages for debugging purposes

    :return: config dict post-adaptation, flow result post-adaptation, list of StructuredTreeOutlet instances
    '''
    print('adapting the vessels...')
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

    adapted_result = run_svzerodplus(config_handler.config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')


def adapt_constant_wss_threed(config_handler: ConfigHandler, preop_q, postop_q, log_file: str = None):
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

            print(R_old, R_new)
            
            # add the updated resistance to the boundary condition
            if bc.type == 'RESISTANCE':
                bc.R = R_new
            if bc.type == 'RCR':
                bc.Rp = 0.1 * R_new
                bc.Rd = 0.9 * R_new

            outlet_idx += 1
