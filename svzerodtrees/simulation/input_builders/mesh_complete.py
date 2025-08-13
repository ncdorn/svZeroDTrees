from .simulation_file import SimulationFile
import os
from .vtp_file import VTPFile
import glob
import vtk

class MeshComplete(SimulationFile):
    '''
    class to handle the mesh complete directory
    '''

    def __init__(self, path, 
                 mesh_surfaces_dir='mesh-surfaces', 
                 volume_mesh='mesh-complete.mesh.vtu', 
                 exterior_mesh='mesh-complete.exterior.vtp', 
                 walls_combined='walls_combined.vtp'):
        '''
        initialize the mesh complete directory'''

        super().__init__(path)

        self.mesh_surfaces_dir = os.path.join(path, mesh_surfaces_dir)

        self.volume_mesh = os.path.join(path, volume_mesh)

        self.walls_combined = VTPFile(os.path.join(path, walls_combined))

        self.exterior_mesh = VTPFile(os.path.join(path, exterior_mesh))


    def initialize(self):
        '''
        get the mesh surfaces in the mesh complete directory as VTP objects'''

        filelist_raw = glob.glob(os.path.join(self.path, 'mesh-surfaces', '*.vtp'))

        filelist = [file for file in filelist_raw if 'wall' not in file]

        filelist.sort()

        # find the inflow vtp
        if 'inflow' in filelist[-1].lower():
            # inflow must be the last element, move to the front
            inflow = filelist.pop(-1)
            filelist.insert(0, inflow)
        elif 'inflow' in filelist[0].lower():
            # inflow is the first element
            pass
        else:
            # find inflow.vtp in the list of files, pop it and move it to the front
            inflow = [file for file in filelist if 'inflow' in file.lower()][0]
            filelist.remove(inflow)
            filelist.insert(0, inflow)
        
        self.mesh_surfaces = {}
        for file in filelist:
            self.mesh_surfaces[os.path.basename(file)] = VTPFile(file)

        self.assign_lobe()
    
    def write(self):
        '''
        write the mesh complete directory'''

        pass

    def scale(self, scale_factor=0.1):
        '''
        scale the mesh complete directory by a scale factor
        '''

        print(f'scaling mesh complete by factor {scale_factor}...')

        # scale the volume mesh
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.volume_mesh)
        reader.Update()

        transform = vtk.vtkTransform()
        transform.Scale(scale_factor, scale_factor, scale_factor)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(transform_filter.GetOutput())
        writer.SetFileName(self.volume_mesh)
        writer.Write()

        # scale the walls combined
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.walls_combined)
        reader.Update()

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(transform_filter.GetOutput())
        writer.SetFileName(self.walls_combined)
        writer.Write()

        # scale the mesh surfaces
        for surface in self.mesh_surfaces:
            surface.scale(scale_factor=scale_factor)

    def assign_lobe_old(self):
        '''
        assign upper, middle or lower lobe location to left and right outlets, except the inlet, based on the center of mass y coourdinate'''

        # get the y coord of lpa and rpa outlets
        lpa_locs = [vtp.get_location()[1] for vtp in self.mesh_surfaces.values() if vtp.lpa]
        rpa_locs = [vtp.get_location()[1] for vtp in self.mesh_surfaces.values() if vtp.rpa]
        
        # get the lobe size (1/3 of the y range)
        lpa_lobe_size = (max(lpa_locs) - min(lpa_locs)) / 3
        rpa_lobe_size = (max(rpa_locs) - min(rpa_locs)) / 3

        # assign outlet lobe location
        for vtp in self.mesh_surfaces.values():
            if vtp.lpa:
                if vtp.get_location()[1] < min(lpa_locs) + lpa_lobe_size:
                    vtp.lobe = 'lower'
                elif vtp.get_location()[1] > max(lpa_locs) - lpa_lobe_size:
                    vtp.lobe = 'upper'
                else:
                    vtp.lobe = 'middle'
            elif vtp.rpa:
                if vtp.get_location()[1] < min(rpa_locs) + rpa_lobe_size:
                    vtp.lobe = 'lower'
                elif vtp.get_location()[1] > max(rpa_locs) - rpa_lobe_size:
                    vtp.lobe = 'upper'
                else:
                    vtp.lobe = 'middle'
        
        # count the number of outlets in each lobe
        lpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'upper'])
        lpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'middle'])
        lpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'lower'])

        rpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'upper'])
        rpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'middle'])
        rpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'lower'])

        print(f'outlets by lobe: LPA upper: {lpa_upper}, middle: {lpa_middle}, lower: {lpa_lower}')
        print(f'outlets by lobe: RPA upper: {rpa_upper}, middle: {rpa_middle}, lower: {rpa_lower}\n')

    def swap_lpa_rpa(self):
        '''
        swap the lpa and rpa outlets
        '''

        for filename in os.listdir(self.mesh_surfaces_dir):
            new_filename = None
            if "LPA" in filename:
                new_filename = filename.replace("LPA", "RPA_")
            elif "RPA" in filename:
                new_filename = filename.replace("RPA", "LPA_")
            
            if new_filename:
                os.rename(os.path.join(self.mesh_surfaces_dir, filename), os.path.join(self.mesh_surfaces_dir, new_filename))
                print(f"Renamed: {filename} to {new_filename}")

        self.initialize()

    def rename_vtps(self):
        '''
        convert surface names from cap_l_pa_n_x.vtp (Derrick's naming convention) to cap_lpa_n.vtp
        '''
 
        for filename in os.listdir(self.mesh_surfaces_dir):
            if '_pa_' in filename:
                tag, vtp = filename.split('.')
                tag = tag.replace('_pa_', 'pa_')
                if 'x' in tag:
                    tag = tag.split('_x')[0]
                new_filename = f'{tag}.{vtp}'
                os.rename(os.path.join(self.mesh_surfaces_dir, filename), os.path.join(self.mesh_surfaces_dir, new_filename))
                print(f"Renamed: {filename} to {new_filename}")


    def assign_lobe(self):
        '''
        assign lobes by sorting the outlets and taking top 1/3 as upper, middle 1/3 as middle and bottom 1/3 as lower
        '''
        # sort lpa and rpa outlets by y coordinates
        sorted_lpa = sorted([outlet for outlet in self.mesh_surfaces.values() if outlet.lpa], key=lambda x: x.get_location()[1])
        sorted_rpa = sorted([outlet for outlet in self.mesh_surfaces.values() if outlet.rpa], key=lambda x: x.get_location()[1])
        
        # get lobe size (1/3 of the y range)
        lpa_lobe_quarter = len(sorted_lpa) // 4
        rpa_lobe_quarter = len(sorted_rpa) // 4

        # assign outlet lobe location
        for i, vtp in enumerate(sorted_lpa):
            if i < lpa_lobe_quarter:
                vtp.lobe = 'lower'
            elif i < lpa_lobe_quarter * 3:
                vtp.lobe = 'middle'
            else:
                vtp.lobe = 'upper'
        for i, vtp in enumerate(sorted_rpa):
            if i < rpa_lobe_quarter:
                vtp.lobe = 'lower'
            elif i < rpa_lobe_quarter * 3:
                vtp.lobe = 'middle'
            else:
                vtp.lobe = 'upper'

        # count the number of outlets in each lobe
        lpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'upper'])
        lpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'middle'])
        lpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.lpa and vtp.lobe == 'lower'])
        rpa_upper = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'upper'])
        rpa_middle = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'middle'])
        rpa_lower = len([vtp for vtp in self.mesh_surfaces.values() if vtp.rpa and vtp.lobe == 'lower'])

        print(f'outlets by lobe: LPA upper: {lpa_upper}, middle: {lpa_middle}, lower: {lpa_lower}')
        print(f'outlets by lobe: RPA upper: {rpa_upper}, middle: {rpa_middle}, lower: {rpa_lower}\n')

        self.n_outlets = lpa_upper + lpa_middle + lpa_lower + rpa_upper + rpa_middle + rpa_lower