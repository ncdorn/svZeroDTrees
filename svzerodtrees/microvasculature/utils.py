from ..io.utils import *

def assign_flow_to_root(result_array, root, steady=False):
    '''
    assign flow values to each TreeVessel instance in a StructuredTreOutlet tree

    :param result_array: svzerodplus result array of the structured tree
    :param root: root TreeVessel instance
    :param steady: True if the model has steady inflow
    '''
    def assign_flow(vessel):
        if vessel:
            # assign flow values to the vessel
            vessel_name = f"branch{vessel.id}_seg0"
            vessel.Q = get_branch_result(result_array, 'flow_in', vessel_name, steady=steady)
            vessel.P_in = get_branch_result(result_array, 'pressure_in', vessel_name, steady=steady)

            # print(f"vessel {vessel.id} Q: {vessel.Q}, P_in: {vessel.P_in}, t_w: {vessel.wall_shear_stress()}, sigma_theta: {vessel.intramural_stress()}")
            # recursive step
            assign_flow(vessel.left)
            assign_flow(vessel.right)
    
    assign_flow(root)
