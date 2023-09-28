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
    R_new = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on Pries and Secomb model **")

    # loop through the vessels and create StructuredTreeOutlet instances at the outlets, from the pre-adaptation tree instances
    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        # generate the postoperative tree with the postop outlet flowrate and pressure
                        outlet_stree = config_handler.trees[outlet_idx]
                        outlet_stree.block_dict["P_in"] = np.mean(postop_p[outlet_idx])
                        outlet_stree.block_dict["Q_out"] = np.mean(postop_q[outlet_idx])

                        # integrate pries and secomb until dD tolerance is reached
                        outlet_stree.integrate_pries_secomb(tol=tol)

                        write_to_log(log_file, 'pries and secomb integration completed for ' + outlet_stree.name)
                        write_to_log(log_file, "    R_new = " + str(outlet_stree.root.R_eq) + ", R_old = " + str(R_old[outlet_idx]))
                        write_to_log(log_file, "    The change in resistance is " + str(outlet_stree.root.R_eq - R_old[outlet_idx]))

                        # add the tree to the vessel config
                        config_handler.trees[outlet_idx] = outlet_stree

                        R_new.append(outlet_stree.root.R_eq)

                        # count up for outlets
                        outlet_idx += 1

    # write adapted tree R_eq to the adapted_config
    write_resistances(config_handler.config, R_new)

    # get the adapted flow and pressure result
    adapted_result = run_svzerodplus(config_handler.config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')

    write_to_log(log_file, 'pries and secomb adaptation completed for all trees. R_old = ' + str(R_old) + ' R_new = ' + str(R_new))



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
    # adapt the tree vessels based on the constant wall shear stress assumption
    preop_q = get_outlet_data(config_handler.config, result_handler.results['preop'], 'flow_out', steady=True)
    postop_q = get_outlet_data(config_handler.config, result_handler.results['postop'], 'flow_out', steady=True)
    # q_diff = [postop_q[i] - q_old for i, q_old in enumerate(preop_q)]

    R_adapt = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on constant wall shear stress assumption **")

    for vessel_config in config_handler.config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in config_handler.config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        # adapt the cwss tree
                        outlet_stree = config_handler.trees[outlet_idx]
                        R_old, R_new = outlet_stree.adapt_constant_wss(Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx])

                        # append the adapted equivalent resistance to the list of adapted resistances
                        R_adapt.append(R_new)

                        # write results to log file for debugging
                        write_to_log(log_file, "** adaptation results for " + str(outlet_stree.name) + " **")
                        write_to_log(log_file, "    R_new = " + str(outlet_stree.root.R_eq) + ", R_old = " + str(R_old))
                        write_to_log(log_file, "    The change in resistance is " + str(outlet_stree.root.R_eq - R_old))

                        outlet_idx += 1

    
    # write the adapted resistances to the config resistance boundary conditions
    write_resistances(config_handler.config, R_adapt)

    adapted_result = run_svzerodplus(config_handler.config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')

