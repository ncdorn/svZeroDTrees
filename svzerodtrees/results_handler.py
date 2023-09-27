from svzerodtrees.utils import *

class ResultHandler:
    '''
    class to handle preop, postop and post adaptation results from the structured tree simulation
    '''

    def __init__(self):
        self.rpa_lpa_branch = None
        self.vessels = []
        self.results = {}
    
    def get_branches(self, config):

        if self.rpa_lpa_branch is None:
            self.rpalpa_branch = find_rpa_lpa_branches(config)

        for vessel_config in config['vessels']:
            id = get_branch_id(vessel_config)
            if id not in [0, self.lpa_rpa_branch[0], self.rpa_lpa_branch[1]]:
                if id not in self.vessels:
                    self.vessels.append(id)
    

    def add_result(self, result, name):
        '''
        add a result array to the result handler

        :param result: the result to add
        :param name: the name of the result
        '''

        # add mpa results
        self.results['mpa']['q_in'][name] = get_branch_result(result, 'flow_in', 0, steady=True)
        self.results['mpa']['q_out'][name] = get_branch_result(result, 'flow_out', 0, steady=True)
        self.results['mpa']['p_in'][name] = get_branch_result(result, 'pressure_in', 0, steady=True)
        self.results['mpa']['p_out'][name] = get_branch_result(result, 'pressure_out', 0, steady=True)

        # add rpa results
        self.results['rpa']['q_in'][name] = get_branch_result(result, 'flow_in', self.rpa_lpa_branch[0], steady=True)
        self.results['rpa']['q_out'][name] = get_branch_result(result, 'flow_out', self.rpa_lpa_branch[0], steady=True)
        self.results['rpa']['p_in'][name] = get_branch_result(result, 'pressure_in', self.rpa_lpa_branch[0], steady=True)
        self.results['rpa']['p_out'][name] = get_branch_result(result, 'pressure_out', self.rpa_lpa_branch[0], steady=True)

        # add lpa results
        self.results['lpa']['q_in'][name] = get_branch_result(result, 'flow_in', self.rpa_lpa_branch[1], steady=True)
        self.results['lpa']['q_out'][name] = get_branch_result(result, 'flow_out', self.rpa_lpa_branch[1], steady=True)
        self.results['lpa']['p_in'][name] = get_branch_result(result, 'pressure_in', self.rpa_lpa_branch[1], steady=True)
        self.results['lpa']['p_out'][name] = get_branch_result(result, 'pressure_out', self.rpa_lpa_branch[1], steady=True)

        # get the result of all the other branches
        for id in self.vessels:
            if id not in [0, self.rpa_lpa_branch[0], self.rpa_lpa_branch[1]]:
                self.results[id]['q_in'][name] = get_branch_result(result, 'flow_in', id, steady=True)
                self.results[id]['q_out'][name] = get_branch_result(result, 'flow_out', id, steady=True)
                self.results[id]['p_in'][name] = get_branch_result(result, 'pressure_in', id, steady=True)
                self.results[id]['p_out'][name] = get_branch_result(result, 'pressure_out', id, steady=True)

