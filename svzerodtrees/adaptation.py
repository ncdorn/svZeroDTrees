from svzerodtrees.utils import *
import copy
from svzerodtrees.structuredtreebc import StructuredTreeOutlet


def adapt_pries_secomb(postop_config, trees, preop_result, postop_result, log_file=None):
    preop_q = get_outlet_data(postop_config, preop_result, 'flow_out', steady=True)
    # get the postop outlet flowrate and pressure
    postop_q = get_outlet_data(postop_config, postop_result, 'flow_out', steady=True)
    print(preop_q, postop_q)
    postop_p = get_outlet_data(postop_config, postop_result, 'pressure_out', steady=True)

    adapted_config = copy.deepcopy(postop_config)

    R_old = [tree.root.R_eq for tree in trees]
    R_new = []
    outlet_idx = 0 # index through outlets

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
                        outlet_stree.integrate_pries_secomb()
                        R_new.append(outlet_stree.root.R_eq)

    write_resistances(adapted_config, R_new)
    adapted_result = run_svzerodplus(adapted_config)

    return adapted_config, adapted_result, trees


def adapt_constant_wss(postop_config, trees, preop_result, postop_result, log_file=None):
    # adapt the tree vessels based on the constant wall shear stress assumption
    preop_q = get_outlet_data(postop_config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(postop_config, postop_result, 'flow_out', steady=True)
    # q_diff = [postop_q[i] - q_old for i, q_old in enumerate(preop_q)]
    adapted_config = copy.deepcopy(postop_config)

    R_new = []
    outlet_idx = 0 # index through outlets
    for tree in trees: # loop through a list of outlet trees
        R_old = tree.root.R_eq  # calculate pre-adaptation resistance

        def constant_wss(d, Q=preop_q[outlet_idx], Q_new=postop_q[outlet_idx]): # function for recursion
            # adapt the radius of the vessel based on the constant shear stress assumption
            return (Q_new / Q) ** (1 / 3) * d

        def update_diameter(vessel, update_func): # function for recursion
            # preorder traversal to update the diameters of all the vessels in the tree  
            if vessel:
                vessel.d = update_func(vessel.d)
                vessel.update_vessel_info()
                update_diameter(vessel.left, update_func)
                update_diameter(vessel.right, update_func)

        update_diameter(tree.root, constant_wss) # recursively update the diameter of the vessels

        tree.create_block_dict()

        R_new.append(tree.root.R_eq)

        write_to_log(log_file, "** adaptation results for " + str(tree.name) + " **")
        write_to_log(log_file, "    R_new = " + str(tree.root.R_eq) + ", R_old = " + str(R_old))
        write_to_log(log_file, "    The change in resistance is " + str(tree.root.R_eq - R_old))

        outlet_idx += 1
    
    write_resistances(adapted_config, R_new)

    adapted_result = run_svzerodplus(adapted_config)

    return adapted_config, adapted_result, trees

