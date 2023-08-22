from svzerodtrees.utils import *
# module for Pries and Secomb adaptation

def optimize(ps_params, trees, q_outs, dt=0.01, P_d=1333.2, steady=True):
    
    # initialize and calculate the Pries and Secomb parameters in the TreeVessel objects via a postorder traversal
    dD_list = [] # initialize the list of dDs for the outlet calculation
    for i, tree in enumerate(trees):
        SS_dD = 0.0 # sum of squared dDs initial guess
        converged = False
        threshold = 10 ** -5
        while not converged:
            tree_config = tree.create_solver_config([q_outs[i]], P_d)
            tree_result = run_svzerodplus(tree_config)
            print('what the fuck')
            assign_flow_to_root(tree_result, tree.root, steady=steady)
            next_SS_dD = 0.0 # initializing sum of squared dDs, value to minimize
            def stimulate(vessel):
                if vessel:
                    stimulate(vessel.left)
                    stimulate(vessel.right)
                    vessel_dD = vessel.adapt_pries_secomb(ps_params, dt)
                    nonlocal next_SS_dD
                    next_SS_dD += vessel_dD ** 2
            stimulate(tree.root)
            dD_diff = abs(next_SS_dD ** 2 - SS_dD ** 2)
            if dD_diff < threshold:
                converged = True
            
            SS_dD = next_SS_dD
            print(dD_diff)
        print('Pries and Secomb integration completed!')
        dD_list.append(next_SS_dD)

    SSE = sum(dD ** 2 for dD in dD_list)

    return SSE