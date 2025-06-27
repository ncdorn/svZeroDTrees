from .simulation_file import SimulationFile
import xml.etree.ElementTree as ET

class SvMPxml(SimulationFile):
    '''
    class to handle the svFSI.xml file in the simulation directory'''

    def __init__(self, path):
        '''
        initialize the svFSIxml object'''
        super().__init__(path)

    def initialize(self):
        '''
        parse the pre-existing svFSI.xml file'''

        self.xml_tree = ET.parse(self.path)
        self.xml_root = self.xml_tree.getroot()

    def write(self, mesh_complete, scale_factor=1.0, n_tsteps=1000, dt=0.01):
        '''
        write the svFSI.xml file
        
        :param mesh_complete: MeshComplete object'''

        self.n_tsteps = n_tsteps
        self.dt = dt

        print('writing svFSIplus.xml...')

        # generate XML tree
        svfsifile = ET.Element("svMultiPhysicsFile")
        svfsifile.set("version", "0.1")

        # General Simulation Parameters
        gensimparams = ET.SubElement(svfsifile, "GeneralSimulationParameters")

        cont_prev_sim = ET.SubElement(gensimparams, "Continue_previous_simulation")
        cont_prev_sim.text = "false"

        num_spatial_dims = ET.SubElement(gensimparams, "Number_of_spatial_dimensions")
        num_spatial_dims.text = "3"

        num_time_steps = ET.SubElement(gensimparams, "Number_of_time_steps")
        num_time_steps.text = str(n_tsteps)

        time_step_size = ET.SubElement(gensimparams, "Time_step_size")
        time_step_size.text = str(dt)

        spec_radius = ET.SubElement(gensimparams, "Spectral_radius_of_infinite_time_step")
        spec_radius.text = "0.5"

        stop_trigger = ET.SubElement(gensimparams, "Searched_file_name_to_trigger_stop")
        stop_trigger.text = "STOP_SIM"

        save_results_to_vtk = ET.SubElement(gensimparams, "Save_results_to_VTK_format")
        save_results_to_vtk.text = "1"

        name_prefix = ET.SubElement(gensimparams, "Name_prefix_of_saved_VTK_files")
        name_prefix.text = "result"

        increment_vtk = ET.SubElement(gensimparams, "Increment_in_saving_VTK_files")
        increment_vtk.text = "20"

        start_saving_tstep = ET.SubElement(gensimparams, "Start_saving_after_time_step")
        start_saving_tstep.text = "1"

        incrememnt_restart = ET.SubElement(gensimparams, "Increment_in_saving_restart_files")
        incrememnt_restart.text = "10"

        convert_bin_vtk = ET.SubElement(gensimparams, "Convert_BIN_to_VTK_format")
        convert_bin_vtk.text = "0"

        verbose = ET.SubElement(gensimparams, "Verbose")
        verbose.text = "1"

        warning = ET.SubElement(gensimparams, "Warning")
        warning.text = "0"

        debug = ET.SubElement(gensimparams, "Debug")
        debug.text = "0"

        # add mesh
        add_mesh = ET.SubElement(svfsifile, "Add_mesh")
        add_mesh.set("name", "msh")

        msh_file_path = ET.SubElement(add_mesh, "Mesh_file_path")
        msh_file_path.text = mesh_complete.volume_mesh


        for vtp in mesh_complete.mesh_surfaces.values():
            add_face = ET.SubElement(add_mesh, "Add_face")
            add_face.set("name", vtp.filename.split('.')[0])

            face_file_path = ET.SubElement(add_face, "Face_file_path")
            face_file_path.text = vtp.path
        
        add_wall = ET.SubElement(add_mesh, "Add_face")
        add_wall.set("name", "wall")

        mesh_scale_Factor = ET.SubElement(add_mesh, "Mesh_scale_factor")
        mesh_scale_Factor.text = str(scale_factor)

        wall_file_path = ET.SubElement(add_wall, "Face_file_path")
        wall_file_path.text = mesh_complete.walls_combined.path

        # add equation
        add_eqn = ET.SubElement(svfsifile, "Add_equation")
        add_eqn.set("type", "fluid")

        coupled = ET.SubElement(add_eqn, "Coupled")
        coupled.text = "1"

        min_iterations = ET.SubElement(add_eqn, "Min_iterations")
        min_iterations.text = "3"

        max_iterations = ET.SubElement(add_eqn, "Max_iterations")
        max_iterations.text = "10"

        tolerance = ET.SubElement(add_eqn, "Tolerance")
        tolerance.text = "1e-3"

        backflow_stab = ET.SubElement(add_eqn, "Backflow_stabilization_coefficient")
        backflow_stab.text = "0.2"

        density = ET.SubElement(add_eqn, "Density")
        density.text = "1.06"

        viscosity = ET.SubElement(add_eqn, "Viscosity", {"model": "Constant"})
        value = ET.SubElement(viscosity, "Value")
        value.text = "0.04"

        output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
        velocity = ET.SubElement(output, "Velocity")
        velocity.text = "true"

        pressure = ET.SubElement(output, "Pressure")
        pressure.text = "true"

        traction = ET.SubElement(output, "Traction")
        traction.text = "true"

        wss = ET.SubElement(output, "WSS")
        wss.text = "true"

        vorticity = ET.SubElement(output, "Vorticity")
        vorticity.text = "true"

        divergence = ET.SubElement(output, "Divergence")
        divergence.text = "true"

        ls = ET.SubElement(add_eqn, "LS", {"type": "NS"})

        linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
        preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
        preconditioner.text = "fsils"

        ls_max_iterations = ET.SubElement(ls, "Max_iterations")
        ls_max_iterations.text = "10"

        ns_gm_max_iterations = ET.SubElement(ls, "NS_GM_max_iterations")
        ns_gm_max_iterations.text = "3"

        ns_cg_max_iterations = ET.SubElement(ls, "NS_CG_max_iterations")
        ns_cg_max_iterations.text = "500"

        ls_tolerance = ET.SubElement(ls, "Tolerance")
        ls_tolerance.text = "1e-3"

        ns_gm_tolerance = ET.SubElement(ls, "NS_GM_tolerance")
        ns_gm_tolerance.text = "1e-3"

        ns_cg_tolerance = ET.SubElement(ls, "NS_CG_tolerance")
        ns_cg_tolerance.text = "1e-3"

        krylov_space_dim = ET.SubElement(ls, "Krylov_space_dimension")
        krylov_space_dim.text = "50"



        couple_to_svzerod = ET.SubElement(add_eqn, "Couple_to_svZeroD")
        couple_to_svzerod.set("type", "SI")

        # add boundary conditions
        for vtp in mesh_complete.mesh_surfaces.values():
            add_bc = ET.SubElement(add_eqn, "Add_BC")
            add_bc.set("name", vtp.filename.split('.')[0])
            
            typ = ET.SubElement(add_bc, "Type")
            typ.text = "Neu"
            time_dep = ET.SubElement(add_bc, "Time_dependence")
            time_dep.text = "Coupled"
        
        # add wall bc
        add_wall_bc = ET.SubElement(add_eqn, "Add_BC")
        add_wall_bc.set("name", "wall")
        typ = ET.SubElement(add_wall_bc, "Type")
        typ.text = "Dir"
        time_dep = ET.SubElement(add_wall_bc, "Time_dependence")
        time_dep.text = "Steady"
        value = ET.SubElement(add_wall_bc, "Value")
        value.text = "0.0"

        # Create the XML tree
        self.xml_tree = ET.ElementTree(svfsifile)

        ET.indent(self.xml_tree.getroot())

        # def prettify(elem):
        #     """Return a pretty-printed XML string for the Element."""
        #     rough_string = ET.tostring(elem, 'utf-8')
        #     reparsed = xml.dom.minidom.parseString(rough_string)
        #     return reparsed.toprettyxml(indent="  ")
        

        # pretty_xml_str = prettify(svfsifile)

        # print(pretty_xml_str)

        # Write the XML to a file
        with open(self.path, "wb") as file:
            self.xml_tree.write(file, encoding="utf-8", xml_declaration=True)

        self.is_written = True