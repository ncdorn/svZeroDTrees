import numpy as np


'''
io utils
'''
def get_branch_id(vessel_config):
    '''
    get the integer id of a branch for a given vessel

    :param vessel_config: config dict of a vessel

    :return: integer branch id
    '''

    parts = vessel_config["vessel_name"].split("_")
    br_part = next((part for part in parts if part.startswith("branch")), None)
    seg_part = next((part for part in parts if part.startswith("seg")), None)
    if br_part is None or seg_part is None:
        raise ValueError(
            f"vessel_name must contain branch* and seg* tokens: {vessel_config['vessel_name']}"
        )

    br = int(br_part[6:])
    seg = int(seg_part[3:])

    return br, seg


def get_branch_result(result_array, data_name: str, branch: int, steady: bool=False):
    '''
    get the flow, pressure or wss result for a model branch form an unformatted result

    :param result_array: svzerodplus DataFrame
    :param data_name: q, p or wss
    :param branch: branch id to get result for
    :param steady: True if the model inflow is steady or youw want to get the average value

    :return: result array for branch and QoI
    '''

    if steady:
        return np.mean(result_array[data_name][branch])
    else:
        # return result_array[data_name][branch] # old code
        return result_array[result_array.name == branch][data_name].to_numpy()
