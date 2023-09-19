from svzerodtrees.utils import *

def summarize_results(config, preop_result, postop_result, final_result, condition='repair'):
    '''
    summarize the adaptation results into preop, postop, post-adaptation flow, pressure and wss.

    :param config: 0d config dict
    :param preop_result: preop result array
    :param postop_result: postop result array
    :param final_result: post adaptation result array
    :param condition: name of the experimental condition

    :return: summ_results dict with summarized results
    '''

    # find the lpa and rpa branch
    rpa_lpa_branch = find_rpa_lpa_branches(config)

    # initialize summary results dict
    summ_results = {condition: {}}

    # get summary results for the MPA
    summ_results[condition]['mpa'] = branch_summary_result(config, preop_result, postop_result, final_result, 0)

    # get summary results for the RPA
    summ_results[condition]['rpa'] = branch_summary_result(config, preop_result, postop_result, final_result, rpa_lpa_branch[0])

    # get summary results for the LPA
    summ_results[condition]['lpa'] = branch_summary_result(config, preop_result, postop_result, final_result, rpa_lpa_branch[1])



    return summ_results


def branch_summary_result(config, preop_result, postop_result, final_result, branch: int):
    '''
    get a dict containing the preop, postop and final q, p, wss for a specified branch

    :param preop_result: preop result array
    :param postop_result: postop result array
    :param final_result: final result array
    :param branch: branch id
    :param name: name of the branch

    :return branch_summary: dict with preop, postop and final outlet q, p, wss
    '''
    
    # initialize branch summary dict
    branch_summary = {}

    # get the inlet flowrates preop, postop, and post adaptation
    preop_q = get_branch_result(preop_result, 'flow_in', branch, steady=True)
    postop_q = get_branch_result(postop_result, 'flow_in', branch, steady=True)
    final_q = get_branch_result(final_result, 'flow_in', branch, steady=True)
    branch_summary['q_in'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}

    # get the outlet flowrates preop, postop, and post adaptation
    preop_q = get_branch_result(preop_result, 'flow_out', branch, steady=True)
    postop_q = get_branch_result(postop_result, 'flow_out', branch, steady=True)
    final_q = get_branch_result(final_result, 'flow_out', branch, steady=True)
    branch_summary['q_out'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}

    # get the inlet pressures preop, postop and post adaptation, in mmHg
    preop_p = get_branch_result(preop_result, 'pressure_in', branch, steady=True) / 1333.22
    postop_p = get_branch_result(postop_result, 'pressure_in', branch, steady=True) / 1333.22
    final_p = get_branch_result(final_result, 'pressure_in', branch, steady=True) / 1333.22
    branch_summary['p_in'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}

    # get the outlet pressures preop, postop and post adaptation, in mm Hg
    preop_p = get_branch_result(preop_result, 'pressure_out', branch, steady=True) / 1333.22
    postop_p = get_branch_result(postop_result, 'pressure_out', branch, steady=True) / 1333.22
    final_p = get_branch_result(final_result, 'pressure_out', branch, steady=True) / 1333.22
    branch_summary['p_out'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}

    # get the wall shear stress at the outlet
    preop_wss = get_wss(config, preop_result, branch, steady=True)
    postop_wss = get_wss(config, postop_result, branch, steady=True)
    final_wss = get_wss(config, final_result, branch, steady=True)
    branch_summary['wss'] = {'preop': preop_wss, 'postop': postop_wss, 'final': final_wss}



    return branch_summary
