from svzerodtrees.utils import *
import pickle
import json

class ResultHandler:
    '''
    class to handle preop, postop and post adaptation results from the structured tree simulation
    '''

    def __init__(self, vessels, lpa_branch, rpa_branch, viscosity):

        self.lpa_branch = lpa_branch
        self.rpa_branch = rpa_branch
        # vessel list in a dict to be compatible with utils
        self.vessels = {'vessels': vessels}
        self.viscosity = viscosity # for wss calculations
        self.results = {}
        self.clean_results = {}
    
    @classmethod
    def from_config(cls, config):
        '''
        class method to generate the results handler with vessel and config information

        :param config: 0d config dict

        :return: ResultHandler instance
        '''

        # get rpa_lpa_branch ids
        lpa_branch, rpa_branch = find_lpa_rpa_branches(config)

        # get vessel info and vessel ids (vessel info necessary for wss calculations)
        vessels = []
        for vessel_config in config['vessels']:
            # add to the vessel list
            vessels.append(vessel_config)

        # get the viscosity
        viscosity = config['simulation_parameters']['viscosity']

        return ResultHandler(vessels, lpa_branch, rpa_branch, viscosity)

    def get_branches(self, config):

        if self.rpa_branch is None and self.lpa_branch is None:
            self.lpa_branch, self.rpa_branch = find_lpa_rpa_branches(config)

        for vessel_config in config['vessels']:
            id = get_branch_id(vessel_config)
            if id not in [0, self.lpa_branch, self.rpa_branch]:
                if id not in self.vessels:
                    self.vessels.append(id)
    

    def format_results(self):
        '''
        format the results into preop, postop and adapted for each branch, for use in visualization
        '''

        # get summary results for the MPA
        self.clean_results['mpa'] = self.format_branch_result(0)

        # get summary results for the RPA
        self.clean_results['rpa'] = self.format_branch_result(self.rpa_branch)

        # get summary results for the LPA
        self.clean_results['lpa'] = self.format_branch_result(self.lpa_branch)

        # get summary results for all other vessels
        for vessel_config in self.vessels['vessels']:
            id = get_branch_id(vessel_config)
            if id not in [0, self.lpa_branch, self.rpa_branch]:
                self.clean_results[id] = self.format_branch_result(id)

    
    def format_branch_result(self, branch: int):
        '''
        get a dict containing the preop, postop and final q, p, wss for a specified branch

        :param branch: branch id

        :return branch_summary: dict with preop, postop and final outlet q, p, wss
        '''
        
        # initialize branch summary dict
        branch_result = {}

        # get the inlet flowrates preop, postop, and post adaptation
        preop_q = get_branch_result(self.results['preop'], 'flow_in', branch, steady=True)
        postop_q = get_branch_result(self.results['postop'], 'flow_in', branch, steady=True)
        final_q = get_branch_result(self.results['adapted'], 'flow_in', branch, steady=True)
        branch_result['q_in'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}

        # get the outlet flowrates preop, postop, and post adaptation
        preop_q = get_branch_result(self.results['preop'], 'flow_out', branch, steady=True)
        postop_q = get_branch_result(self.results['postop'], 'flow_out', branch, steady=True)
        final_q = get_branch_result(self.results['adapted'], 'flow_out', branch, steady=True)
        branch_result['q_out'] = {'preop': preop_q, 'postop': postop_q, 'final': final_q}

        # get the inlet pressures preop, postop and post adaptation, in mmHg
        preop_p = get_branch_result(self.results['preop'], 'pressure_in', branch, steady=True) / 1333.22
        postop_p = get_branch_result(self.results['postop'], 'pressure_in', branch, steady=True) / 1333.22
        final_p = get_branch_result(self.results['adapted'], 'pressure_in', branch, steady=True) / 1333.22
        branch_result['p_in'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}

        # get the outlet pressures preop, postop and post adaptation, in mm Hg
        preop_p = get_branch_result(self.results['preop'], 'pressure_out', branch, steady=True) / 1333.22
        postop_p = get_branch_result(self.results['postop'], 'pressure_out', branch, steady=True) / 1333.22
        final_p = get_branch_result(self.results['adapted'], 'pressure_out', branch, steady=True) / 1333.22
        branch_result['p_out'] = {'preop': preop_p, 'postop': postop_p, 'final': final_p}

        # get the wall shear stress at the outlet
        preop_wss = get_wss(self.vessels, self.viscosity, self.results['preop'], branch, steady=True)
        postop_wss = get_wss(self.vessels, self.viscosity, self.results['postop'], branch, steady=True)
        final_wss = get_wss(self.vessels, self.viscosity, self.results['adapted'], branch, steady=True)
        branch_result['wss'] = {'preop': preop_wss, 'postop': postop_wss, 'final': final_wss}


        return branch_result

    def add_unformatted_result(self, result, name):
        '''
        add an unformatted svzerodplus result to the result handler

        :param result: the result to add
        :param name: the name of the result (preop, postop, final)
        '''

        self.results[name] = result

    def to_file(self, file_name: str):
        '''
        write the result handler to a pickle file

        :param file_name: name of the file to write to
        '''

        with open(file_name, 'wb') as ff:
            pickle.dump(self, ff)
    
    def to_json(self, file_name: str):
        '''
        write the result handler to a json file

        :param file_name: name of the file to write to
        '''

        self.format_results()

        with open(file_name, 'w') as ff:
            json.dump(self.clean_results, ff)


    def format_result_for_cl_projection(self, timestep):
        '''format a pressure or flow result for the centerline projection

        Args:
            timestep (str): timestep to format, ['preop', 'postop', 'adapted']
        
        
        '''

        cl_mappable_result = {"flow": {}, "pressure": {}, "distance": {},"time": {}}

        branches = list(self.clean_results.keys())
        for branch in branches:
            if branch == 'mpa':
                branches[branches.index(branch)] = 0
            elif branch == 'lpa':
                branches[branches.index(branch)] = self.lpa_branch
            elif branch == 'rpa':
                branches[branches.index(branch)] = self.rpa_branch

        fields = list(self.results['preop'].keys())

        fields.sort() # should be ['flow_in', 'flow_out', 'pressure_in', 'pressure_out']

        if timestep == 'adaptation':
            for field in ['q', 'p']:
                    cl_mappable_result[field] = {
                        branch: [(self.clean_results[branch][field + "_in"]['postop'] - 
                                 self.clean_results[branch][field + "_in"]['final']) / 
                                 self.clean_results[branch][field + "_in"]['postop'], 
                                (self.clean_results[branch][field + "_out"]['postop'] - 
                                 self.clean_results[branch][field + "_out"]['final']) / 
                                 self.clean_results[branch][field + "_out"]['postop'], ]
                        for branch in branches}
        else:
            for field in ['flow', 'pressure']:
                    cl_mappable_result[field] = {
                        branch: [self.results[timestep][field + "_in"][branch], 
                                self.results[timestep][field + "_out"][branch]]
                        for branch in branches}
                    # now, change the branch id for mpa, lpa, rpa
                    for branch in cl_mappable_result[field].keys():
                        if branch == 'mpa':
                            cl_mappable_result[field][0] = cl_mappable_result[field].pop(branch)
                        elif branch == 'lpa':
                            cl_mappable_result[field][self.lpa_branch] = cl_mappable_result[field].pop(branch)
                        elif branch == 'rpa':
                            cl_mappable_result[field][self.rpa_branch] = cl_mappable_result[field].pop(branch)
        
        # construct the distance dict

        return cl_mappable_result


