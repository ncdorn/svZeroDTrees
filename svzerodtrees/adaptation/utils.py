
'''
adaptation utils
'''

def write_resistances(config, resistances):
    '''
    write a list of resistances to the outlet bcs of a config dict

    :param config: svzerodplus config dict
    :param resistances: list of resistances, ordered by outlet in the config
    '''
    idx = 0
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':

            bc_config['bc_values']['R'] = resistances[idx]

            idx += 1

