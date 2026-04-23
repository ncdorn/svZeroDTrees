from .simulation_file import SimulationFile
from ...io import ConfigHandler
from dataclasses import asdict, is_dataclass
import xml.etree.ElementTree as ET
import os


def _normalize_tissue_support(tissue_support):
    if tissue_support is None:
        return None
    if is_dataclass(tissue_support):
        tissue_support = asdict(tissue_support)
    elif not isinstance(tissue_support, dict):
        tissue_support = vars(tissue_support)

    enabled = bool(tissue_support.get("enabled", True))
    support_type = str(tissue_support.get("type", "uniform")).lower()
    if support_type not in {"uniform", "spatial"}:
        raise ValueError("tissue_support.type must be one of uniform|spatial")

    normalized = {
        "enabled": enabled,
        "type": support_type,
        "apply_along_normal_direction": bool(
            tissue_support.get("apply_along_normal_direction", True)
        ),
        "stiffness": tissue_support.get("stiffness"),
        "damping": tissue_support.get("damping"),
        "spatial_values_file_path": tissue_support.get("spatial_values_file_path"),
    }
    if not enabled:
        return normalized

    if support_type == "uniform":
        if normalized["stiffness"] is None or normalized["damping"] is None:
            raise ValueError("uniform tissue_support requires stiffness and damping")
        normalized["stiffness"] = float(normalized["stiffness"])
        normalized["damping"] = float(normalized["damping"])
        if normalized["stiffness"] < 0.0 or normalized["damping"] < 0.0:
            raise ValueError("tissue_support stiffness and damping must be non-negative")
        if normalized["spatial_values_file_path"] is not None:
            raise ValueError("uniform tissue_support forbids spatial_values_file_path")
    else:
        if not normalized["spatial_values_file_path"]:
            raise ValueError("spatial tissue_support requires spatial_values_file_path")
        if normalized["stiffness"] is not None or normalized["damping"] is not None:
            raise ValueError("spatial tissue_support forbids stiffness and damping")

    return normalized


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

    def write(self,
              mesh_complete,
              scale_factor=1.0,
              n_tsteps=1000,
              dt=0.01,
              wall_model="rigid",
              elasticity_modulus=5062674.563165,
              poisson_ratio=0.5,
              shell_thickness=0.12,
              prestress_file_path=None,
              tissue_support=None,
              simulation_mode="flow",
              traction_file_path=None,
              threed_coupler=None,
              inflow_boundary_condition="neumann",
              inflow_file_path=None,
              coupling_type="semi-implicit",
              configuration_file="svzerod_3Dcoupling.json",
              shared_library="/home/users/ndorn/svZeroDSolver/Release/src/interface/libsvzero_interface.so",
              initial_flows=0.0,
              initial_pressures=0.0):
        '''
        write the svFSI.xml file
        
        :param mesh_complete: MeshComplete object'''

        self.n_tsteps = n_tsteps
        self.dt = dt
        inflow_boundary_condition = str(inflow_boundary_condition).lower()
        if inflow_boundary_condition not in {"neumann", "dirichlet"}:
            raise ValueError("inflow_boundary_condition must be one of neumann|dirichlet")

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

        simulation_mode = str(simulation_mode).lower()
        if simulation_mode not in {"flow", "prestress"}:
            raise ValueError("simulation_mode must be one of flow|prestress")

        wall_model = str(wall_model).lower()
        if wall_model not in {"rigid", "deformable"}:
            raise ValueError("wall_model must be one of rigid|deformable")
        if wall_model == "deformable":
            if float(elasticity_modulus) <= 0.0:
                raise ValueError("elasticity_modulus must be > 0 for deformable wall model")
            if float(shell_thickness) <= 0.0:
                raise ValueError("shell_thickness must be > 0 for deformable wall model")
            if not (-1.0 < float(poisson_ratio) <= 0.5):
                raise ValueError("poisson_ratio must satisfy -1.0 < v <= 0.5 for deformable wall model")
        tissue_support = _normalize_tissue_support(tissue_support)
        if tissue_support is not None and tissue_support["enabled"] and wall_model != "deformable":
            raise ValueError("tissue_support is only valid with wall_model='deformable'")

        if simulation_mode == "prestress":
            if not traction_file_path:
                raise ValueError("traction_file_path is required for simulation_mode='prestress'")

            add_mesh = ET.SubElement(svfsifile, "Add_mesh")
            add_mesh.set("name", "wall")
            set_shell = ET.SubElement(add_mesh, "Set_mesh_as_shell")
            set_shell.text = "true"
            msh_file_path = ET.SubElement(add_mesh, "Mesh_file_path")
            msh_file_path.text = mesh_complete.walls_combined.path

            add_eqn = ET.SubElement(svfsifile, "Add_equation")
            add_eqn.set("type", "CMM")
            coupled = ET.SubElement(add_eqn, "Coupled")
            coupled.text = "true"
            min_iterations = ET.SubElement(add_eqn, "Min_iterations")
            min_iterations.text = "3"
            max_iterations = ET.SubElement(add_eqn, "Max_iterations")
            max_iterations.text = "30"
            tolerance = ET.SubElement(add_eqn, "Tolerance")
            tolerance.text = "1e-12"
            prestress = ET.SubElement(add_eqn, "Prestress")
            prestress.text = "true"
            initialize = ET.SubElement(add_eqn, "Initialize")
            initialize.text = "prestress"

            poisson_ratio_node = ET.SubElement(add_eqn, "Poisson_ratio")
            poisson_ratio_node.text = str(poisson_ratio)
            shell_thickness_node = ET.SubElement(add_eqn, "Shell_thickness")
            shell_thickness_node.text = str(shell_thickness)
            elasticity_modulus_node = ET.SubElement(add_eqn, "Elasticity_modulus")
            elasticity_modulus_node.text = str(elasticity_modulus)

            output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
            displacement = ET.SubElement(output, "Displacement")
            displacement.text = "true"
            stress = ET.SubElement(output, "Stress")
            stress.text = "true"

            ls = ET.SubElement(add_eqn, "LS", {"type": "GMRES"})
            linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
            preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
            preconditioner.text = "fsils"
            ls_max_iterations = ET.SubElement(ls, "Max_iterations")
            ls_max_iterations.text = "500"
            ls_tolerance = ET.SubElement(ls, "Tolerance")
            ls_tolerance.text = "1e-12"

            add_bf = ET.SubElement(add_eqn, "Add_BF")
            add_bf.set("mesh", "wall")
            bf_type = ET.SubElement(add_bf, "Type")
            bf_type.text = "traction"
            time_dep = ET.SubElement(add_bf, "Time_dependence")
            time_dep.text = "spatial"
            spatial_values = ET.SubElement(add_bf, "Spatial_values_file_path")
            spatial_values.text = str(traction_file_path)

            self.xml_tree = ET.ElementTree(svfsifile)
            ET.indent(self.xml_tree.getroot())
            with open(self.path, "wb") as file:
                self.xml_tree.write(file, encoding="utf-8", xml_declaration=True)
            self.is_written = True
            return

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
        add_eqn.set("type", "fluid" if wall_model == "rigid" else "CMM")

        coupled = ET.SubElement(add_eqn, "Coupled")
        coupled.text = "true" if wall_model == "deformable" else "1"

        min_iterations = ET.SubElement(add_eqn, "Min_iterations")
        min_iterations.text = "3"

        max_iterations = ET.SubElement(add_eqn, "Max_iterations")
        max_iterations.text = "10"

        tolerance = ET.SubElement(add_eqn, "Tolerance")
        tolerance.text = "0.01" if wall_model == "deformable" else "1e-3"

        if wall_model == "rigid":
            backflow_stab = ET.SubElement(add_eqn, "Backflow_stabilization_coefficient")
            backflow_stab.text = "0.2"

        if wall_model == "rigid":
            density = ET.SubElement(add_eqn, "Density")
            density.text = "1.06"
        else:
            fluid_density = ET.SubElement(add_eqn, "Fluid_density")
            fluid_density.text = "1.06"
            solid_density = ET.SubElement(add_eqn, "Solid_density")
            solid_density.text = "1.0"
            poisson_ratio_node = ET.SubElement(add_eqn, "Poisson_ratio")
            poisson_ratio_node.text = str(poisson_ratio)
            shell_thickness_node = ET.SubElement(add_eqn, "Shell_thickness")
            shell_thickness_node.text = str(shell_thickness)
            elasticity_modulus_node = ET.SubElement(add_eqn, "Elasticity_modulus")
            elasticity_modulus_node.text = str(elasticity_modulus)

        viscosity = ET.SubElement(add_eqn, "Viscosity", {"model": "Constant"})
        value = ET.SubElement(viscosity, "Value")
        value.text = "0.04"

        output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
        if wall_model == "deformable":
            displacement = ET.SubElement(output, "Displacement")
            displacement.text = "true"
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

        ls = ET.SubElement(add_eqn, "LS", {"type": "NS" if wall_model == "rigid" else "GMRES"})

        linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
        preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
        preconditioner.text = "fsils"

        ls_max_iterations = ET.SubElement(ls, "Max_iterations")
        ls_max_iterations.text = "10" if wall_model == "rigid" else "100"

        ls_tolerance = ET.SubElement(ls, "Tolerance")
        ls_tolerance.text = "1e-3" if wall_model == "rigid" else "0.01"

        if wall_model == "rigid":
            ns_gm_max_iterations = ET.SubElement(ls, "NS_GM_max_iterations")
            ns_gm_max_iterations.text = "3"

            ns_cg_max_iterations = ET.SubElement(ls, "NS_CG_max_iterations")
            ns_cg_max_iterations.text = "500"

            ns_gm_tolerance = ET.SubElement(ls, "NS_GM_tolerance")
            ns_gm_tolerance.text = "1e-3"

            ns_cg_tolerance = ET.SubElement(ls, "NS_CG_tolerance")
            ns_cg_tolerance.text = "1e-3"

            krylov_space_dim = ET.SubElement(ls, "Krylov_space_dimension")
            krylov_space_dim.text = "50"



        coupling_blocks_by_surface = {}
        if threed_coupler is not None:
            if isinstance(threed_coupler, str):
                threed_coupler = ConfigHandler.from_json(threed_coupler, is_pulmonary=False, is_threed_interface=True)
            for block in threed_coupler.coupling_blocks.values():
                surface = getattr(block, "surface", None)
                if not surface:
                    continue
                surface_base = os.path.splitext(os.path.basename(surface))[0].lower()
                coupling_blocks_by_surface[surface_base] = block.name
            if coupling_blocks_by_surface:
                svzerod_interface = ET.SubElement(add_eqn, "svZeroDSolver_interface")
                coupling_type_node = ET.SubElement(svzerod_interface, "Coupling_type")
                coupling_type_node.text = str(coupling_type)
                config_node = ET.SubElement(svzerod_interface, "Configuration_file")
                config_node.text = str(configuration_file)
                shared_lib_node = ET.SubElement(svzerod_interface, "Shared_library")
                shared_lib_node.text = str(shared_library)
                init_flows_node = ET.SubElement(svzerod_interface, "Initial_flows")
                init_flows_node.text = str(initial_flows)
                init_pressures_node = ET.SubElement(svzerod_interface, "Initial_pressures")
                init_pressures_node.text = str(initial_pressures)

        # add boundary conditions
        dirichlet_inflow_written = False
        for vtp in mesh_complete.mesh_surfaces.values():
            add_bc = ET.SubElement(add_eqn, "Add_BC")
            surface_name = vtp.filename.split('.')[0]
            add_bc.set("name", surface_name)

            typ = ET.SubElement(add_bc, "Type")
            time_dep = ET.SubElement(add_bc, "Time_dependence")
            if inflow_boundary_condition == "dirichlet" and "inflow" in surface_name.lower():
                typ.text = "Dir"
                time_dep.text = "Unsteady"
                temporal_values = ET.SubElement(add_bc, "Temporal_values_file_path")
                temporal_values.text = os.path.basename(inflow_file_path or "inflow.flow")
                profile = ET.SubElement(add_bc, "Profile")
                profile.text = "Parabolic"
                impose_flux = ET.SubElement(add_bc, "Impose_flux")
                impose_flux.text = "true"
                dirichlet_inflow_written = True
                continue

            typ.text = "Neu"
            time_dep.text = "Coupled"

            block_name = coupling_blocks_by_surface.get(surface_name.lower())
            if block_name:
                svzerod_block = ET.SubElement(add_bc, "svZeroDSolver_block")
                svzerod_block.text = block_name
        if inflow_boundary_condition == "dirichlet" and not dirichlet_inflow_written:
            raise ValueError("dirichlet inflow_boundary_condition requires at least one inflow surface")
        
        # add wall bc
        add_wall_bc = ET.SubElement(add_eqn, "Add_BC")
        add_wall_bc.set("name", "wall")
        typ = ET.SubElement(add_wall_bc, "Type")
        if wall_model == "deformable":
            typ.text = "CMM"
            if prestress_file_path:
                prestress_path_node = ET.SubElement(add_wall_bc, "Prestress_file_path")
                prestress_path_node.text = str(prestress_file_path)
            if tissue_support is not None and tissue_support["enabled"]:
                support = ET.SubElement(add_wall_bc, "Tissue_support")
                if tissue_support["type"] == "uniform":
                    stiffness = ET.SubElement(support, "Stiffness")
                    stiffness.text = str(tissue_support["stiffness"])
                    damping = ET.SubElement(support, "Damping")
                    damping.text = str(tissue_support["damping"])
                else:
                    spatial_values = ET.SubElement(support, "Spatial_values_file_path")
                    spatial_values.text = str(tissue_support["spatial_values_file_path"])
                normal_only = ET.SubElement(support, "Apply_along_normal_direction")
                normal_only.text = (
                    "true" if tissue_support["apply_along_normal_direction"] else "false"
                )
        else:
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
