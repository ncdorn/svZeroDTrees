from svzerodtrees.utils import *
import copy
from svzerodtrees.structuredtreebc import StructuredTreeOutlet


def adapt_pries_secomb(postop_config, trees, preop_result, postop_result, log_file=None):
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
    preop_q = get_outlet_data(postop_config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_config, postop_result, 'flow_out', steady=True)
    postop_p = get_outlet_data(postop_config, postop_result, 'pressure_out', steady=True)

    # copy the postop_config to ensure we are not editing it
    adapted_config = copy.deepcopy(postop_config)

    # initialize R_old and R_new for pre- and post-adaptation comparison
    R_old = [tree.root.R_eq for tree in trees]
    R_new = []
    outlet_idx = 0 # index through outlets

    # loop through the vessels and create StructuredTreeOutlet instances at the outlets, from the pre-adaptation tree instances
    for vessel_config in postop_config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for bc_config in postop_config["boundary_conditions"]:
                    if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                        # generate the postoperative tree with the postop outlet flowrate and pressure
                        outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, 
                                                                               postop_config["simulation_parameters"],
                                                                               bc_config, 
                                                                               tree_exists=True,
                                                                               root=trees[outlet_idx].root,
                                                                               Q_outlet=[np.mean(postop_q[outlet_idx])],
                                                                               P_outlet=[np.mean(postop_p[outlet_idx])])
                        # integrate pries and secomb until dD tolerance is reached
                        outlet_stree.integrate_pries_secomb()

                        write_to_log(log_file, 'pries and secomb integration completed for ' + outlet_stree.name)

                        R_new.append(outlet_stree.root.R_eq)

    # write adapted tree R_eq to the adapted_config
    write_resistances(adapted_config, R_new)

    # get the adapted flow and pressure result
    adapted_result = run_svzerodplus(adapted_config)

    write_to_log(log_file, 'pries and secomb adaptation completed for all trees. R_old = ' + str(R_old) + ' R_new = ' + str(R_new))

    return adapted_config, adapted_result, trees


def adapt_constant_wss(postop_config, trees, preop_result, postop_result, log_file=None):
    '''
    adapt structured trees based on the constant wall shear stress assumption

    :param postop_config: config dict from the stenosis repair
    :param trees: list of StructuredTreeOutlet instances, corresponding to the outlets of the 0D model
    :param preop_result: preoperative result array
    :param postop_result: postoperative result array, calculated after the stenosis repair
    :param log_file: path to log file, for writing important messages for debugging purposes

    :return: config dict post-adaptation, flow result post-adaptation, list of StructuredTreeOutlet instances
    '''
    # adapt the tree vessels based on the constant wall shear stress assumption
    preop_q = get_outlet_data(postop_config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_config, postop_result, 'flow_out', steady=True)
    # q_diff = [postop_q[i] - q_old for i, q_old in enumerate(preop_q)]
    adapted_config = copy.deepcopy(postop_config)

    R_adapt = []
    outlet_idx = 0 # index through outlets

    for tree in trees: # loop through a list of outlet trees
        # adapt the diameter of each vessel in the tree
        R_old, R_new = tree.adapt_constant_wss(Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx])

        # append the adapted equivalent resistance to the list of adapted resistances
        R_adapt.append(R_new)

        # write results to log file for debugging
        write_to_log(log_file, "** adaptation results for " + str(tree.name) + " **")
        write_to_log(log_file, "    R_new = " + str(tree.root.R_eq) + ", R_old = " + str(R_old))
        write_to_log(log_file, "    The change in resistance is " + str(tree.root.R_eq - R_old))

        outlet_idx += 1
    
    # write the adapted resistances to the config resistance boundary conditions
    write_resistances(adapted_config, R_adapt)

    adapted_result = run_svzerodplus(adapted_config)

    return adapted_config, adapted_result, trees

