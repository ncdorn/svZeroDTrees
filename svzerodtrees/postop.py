from svzerodtrees.utils import *

def summarize_results(config, preop_result, postop_result, final_result, condition='repair'):
    '''
    :param config: 0d config dict
    :param preop_result: preop result array
    :param postop_result: postop result array
    :param final_result: post adaptation result array
    :param condition: name of the experimental condition
    :return summ_results: dict with summarized results
    '''

    summ_results = {condition: {}}
    # get the flowrates preop, postop, and post adaptation
    preop_q = get_outlet_data(config, preop_result, 'flow_out', steady=True)
    postop_q = get_outlet_data(config, postop_result, 'flow_out', steady=True)
    final_q = get_outlet_data(config, final_result, 'flow_out', steady=True)
    summ_results[condition]['q'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}
    # get the pressures preop, postop and post adaptation
    preop_p = get_outlet_data(config, preop_result, 'pressure_out', steady=True)
    postop_p = get_outlet_data(config, postop_result, 'pressure_out', steady=True)
    final_p = get_outlet_data(config, final_result, 'pressure_out', steady=True)
    summ_results[condition]['p'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}
    # get the wall shear stress at the outlet
    preop_wss = get_outlet_data(config, preop_result, 'wss', steady=True)
    postop_wss = get_outlet_data(config, postop_result, 'wss', steady=True)
    final_wss = get_outlet_data(config, final_result, 'wss', steady=True)
    summ_results[condition]['wss'] = {'preop': preop_wss, 'postop': postop_wss, 'final': final_wss}

    return summ_results