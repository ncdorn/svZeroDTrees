from ...io import ConfigHandler
from .simulation_file import SimulationFile

class SvZeroDInterface(SimulationFile):
    '''
    class to handle the svZeroD_interface.dat file in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the svZeroD_interface object'''
        super().__init__(path)


    def initialize(self):
        '''
        initialize from a pre-existing svZeroD_interface.dat file'''

        with open(self.path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]


        self.interface_library_path = lines[lines.index('interface library path:')+1]

        self.svZeroD_input_file = lines[lines.index('svZeroD input file:')+1]

        self.coupling_block_to_surf_id = {} 
        for line in lines[lines.index('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file):')+1:]:
            if line == '':
                break
            block, surf_id = line.split(' ')
            self.coupling_block_to_surf_id[block] = surf_id
        
        self.initialize_flows = lines[lines.index('Initialize external coupling block flows:')+1]

        self.initial_flow = lines[lines.index('External coupling block initial flows (one number is provided, it is applied to all coupling blocks):')+1]

        self.initialize_pressures = lines[lines.index('Initialize external coupling block pressures:')+1]

        self.initial_pressure = lines[lines.index('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks):')+1]
        

    def write(self,
              threed_coupler_path,
              interface_path='/home/users/ndorn/svZeroDSolver/Release/src/interface/libsvzero_interface.so',
              initialize_flows=0,
              initial_flow=0.0,
              initialize_pressures=1,
              initial_pressure=60.0):
        '''
        write the svZeroD_interface.dat file'''
        
        print('writing svZeroD interface file...')

        threed_coupler = ConfigHandler.from_json(threed_coupler_path, is_pulmonary=False, is_threed_interface=True)

        outlet_blocks = [block.name for block in list(threed_coupler.coupling_blocks.values())]

        with open(self.path, 'w') as ff:
            ff.write('interface library path: \n')
            ff.write(interface_path + '\n\n')

            ff.write('svZeroD input file: \n')
            ff.write(threed_coupler_path + '\n\n')
            
            ff.write('svZeroD external coupling block names to surface IDs (where surface IDs are from *.svpre file): \n')
            for idx, bc in enumerate(outlet_blocks):
                ff.write(f'{bc} {idx}\n')

            ff.write('\n')
            ff.write('Initialize external coupling block flows: \n')
            ff.write(f'{initialize_flows}\n\n')

            ff.write('External coupling block initial flows (one number is provided, it is applied to all coupling blocks): \n')
            ff.write(f'{initial_flow}\n\n')

            ff.write('Initialize external coupling block pressures: \n')
            ff.write(f'{initialize_pressures}\n\n')

            ff.write('External coupling block initial pressures (one number is provided, it is applied to all coupling blocks): \n')
            ff.write(f'{initial_pressure}\n\n')

        self.initialize()

        self.is_written = True