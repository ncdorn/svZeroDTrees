from svzerodtrees.utils import *
import copy
from svzerodtrees.structuredtreebc import StructuredTreeOutlet
from svzerodtrees.results_handler import ResultHandler


def adapt_pries_secomb(postop_config: dict, trees: list, result_handler: ResultHandler, log_file: str = None, tol: float = .01):
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
    preop_q = get_outlet_data(postop_config, result_handler.results['preop'], 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_config, result_handler.results['postop'], 'flow_out', steady=True)
    postop_p = get_outlet_data(postop_config, result_handler.results['postop'], 'pressure_out', steady=True)

    # point to the postop config
    adapted_config = postop_config

    # initialize R_old and R_new for pre- and post-adaptation comparison
    R_old = [tree.root.R_eq for tree in trees]
    R_new = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on Pries and Secomb model **")

    # loop through the vessels and create StructuredTreeOutlet instances at the outlets, from the pre-adaptation tree instances
    for vessel_config in adapted_config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in adapted_config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        # generate the postoperative tree with the postop outlet flowrate and pressure
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                               adapted_config["simulation_parameters"],
                                                                               bc_config, 
                                                                               tree_exists=True,
                                                                               root=trees[outlet_idx].root,
                                                                               Q_outlet=[np.mean(postop_q[outlet_idx])],
                                                                               P_outlet=[np.mean(postop_p[outlet_idx])])
                        # integrate pries and secomb until dD tolerance is reached
                        outlet_stree.integrate_pries_secomb(tol=tol)

                        write_to_log(log_file, 'pries and secomb integration completed for ' + outlet_stree.name)
                        write_to_log(log_file, "    R_new = " + str(outlet_stree.root.R_eq) + ", R_old = " + str(R_old[outlet_idx]))
                        write_to_log(log_file, "    The change in resistance is " + str(outlet_stree.root.R_eq - R_old[outlet_idx]))

                        # add the tree to the vessel config
                        vessel_config["tree"] = outlet_stree.block_dict

                        R_new.append(outlet_stree.root.R_eq)

                        # count up for outlets
                        outlet_idx += 1

    # write adapted tree R_eq to the adapted_config
    write_resistances(adapted_config, R_new)

    # get the adapted flow and pressure result
    adapted_result = run_svzerodplus(adapted_config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')

    write_to_log(log_file, 'pries and secomb adaptation completed for all trees. R_old = ' + str(R_old) + ' R_new = ' + str(R_new))

    return adapted_config, result_handler, trees


def adapt_constant_wss(postop_config: dict, trees: list, result_handler: ResultHandler, log_file: str = None):
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
    preop_q = get_outlet_data(postop_config, result_handler.results['preop'], 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_config, result_handler.results['postop'], 'flow_out', steady=True)
    # q_diff = [postop_q[i] - q_old for i, q_old in enumerate(preop_q)]

    # point to the postop config
    adapted_config = postop_config

    R_adapt = []
    outlet_idx = 0 # index through outlets

    write_to_log(log_file, "** adapting trees based on constant wall shear stress assumption **")

    for vessel_config in adapted_config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in adapted_config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                        # adapt the cwss tree
                        outlet_stree = vessel_config["tree"]
                        R_old, R_new = outlet_stree.adapt_constant_wss(Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx])

                        # append the adapted equivalent resistance to the list of adapted resistances
                        R_adapt.append(R_new)

                        # write results to log file for debugging
                        write_to_log(log_file, "** adaptation results for " + str(outlet_stree.name) + " **")
                        write_to_log(log_file, "    R_new = " + str(outlet_stree.root.R_eq) + ", R_old = " + str(R_old))
                        write_to_log(log_file, "    The change in resistance is " + str(outlet_stree.root.R_eq - R_old))

                        outlet_idx += 1

    
    # write the adapted resistances to the config resistance boundary conditions
    write_resistances(adapted_config, R_adapt)

    adapted_result = run_svzerodplus(adapted_config)

    # add adapted result to the result handler
    result_handler.add_unformatted_result(adapted_result, 'adapted')

    return adapted_config, result_handler

