import vtk
from .simulation_file import SimulationFile

class VTPFile(SimulationFile):
    '''
    class to handle vtp files in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the vtp object'''
        super().__init__(path)

        self.lobe = None # to be assigned later

        if 'lpa' in self.filename.lower():
            self.lpa = True
            self.rpa = False
            self.inflow = False
        elif 'rpa' in self.filename.lower():
            self.rpa = True
            self.lpa = False
            self.inflow = False
        elif 'inflow' or 'mpa' in self.filename.lower():
            self.inflow = True
            self.lpa = False
            self.rpa = False
    
    def initialize(self):
        '''
        initialize the vtp file'''
        self.get_area()

    def write(self):
        '''
        write the vtp file'''

        pass
    
    def get_area(self):
        # with open(infile):
        # print('file able to be opened!')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()
        poly = reader.GetOutputPort()
        masser = vtk.vtkMassProperties()
        masser.SetInputConnection(poly)
        masser.Update()

        self.area = masser.GetSurfaceArea()

    def get_location(self):
        '''
        get the center of mass of the outlet'''
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()
        poly = reader.GetOutputPort()
        com = vtk.vtkCenterOfMass()
        com.SetInputConnection(poly)
        com.Update()

        self.center = com.GetCenter()

        return self.center

    def scale(self, scale_factor=0.1):
        '''
        scale a vtp file from mm to cm (multiply by 0.1) using vtkTransform
        '''

        print(f'scaling {self.filename} by factor {scale_factor}...')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.filename)
        reader.Update()

        # get the area before scaling
        print(f'area before scaling: {self.area}')

        transform = vtk.vtkTransform()
        transform.Scale(scale_factor, scale_factor, scale_factor)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(transform_filter.GetOutput())   
        writer.SetFileName(self.filename)
        writer.Write()

        # get the area after scaling
        self.get_area()
        print(f'area after scaling: {self.area}')
