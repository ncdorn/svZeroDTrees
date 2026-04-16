
import numpy as np
from ..io import *
from ..io.blocks.boundary_condition import resolve_impedance_timepoint_contract
from ..utils import *
from ..microvasculature import StructuredTree, TreeParameters
from ..simulation.threedutils import vtp_info
from .utils import *

# this is where the logic for assigning boundary conditions to the 3D model is implemented


def _attach_tree_metadata(tree, params, *, generation_mode, side, bc_names, outlet_names):
    tree.inductance = float(params.inductance)
    tree.generation_mode = generation_mode
    tree.outlet_mapping = {
        "mode": generation_mode,
        "side": side,
        "bc_names": list(bc_names),
        "outlet_names": list(outlet_names),
    }
    return tree.to_dict()

def construct_impedance_trees(config_handler, 
                              mesh_surfaces_path, 
                              wedge_pressure, 
                              lpa_params: TreeParameters,
                              rpa_params: TreeParameters,
                              d_min, 
                              convert_to_cm=False, 
                              is_pulmonary=True, 
                              n_procs=24,
                              use_mean=False,
                              specify_diameter=False,
                              diameter_scale=0.0,
                              diameter_std_cap=None):
    '''
    construct impedance trees for outlet BCs
    
    :param k2: stiffness parameter 2
    :param k3: stiffness parameter 3
    :param use_mean: when True, build only two trees (LPA/RPA) and reuse for all outlets
    :param diameter_scale: for unique trees, shrink diameter spread toward the mean (0=all mean, 1=full spread)
    :param diameter_std_cap: optional cap in std deviations on diameter deviation before scaling'''

    def _cap_diameter(area):
        return (area / np.pi) ** 0.5 * 2

    def _scaled_diameter(cap_d, mean_d, std_d):
        if diameter_std_cap is not None and std_d > 0:
            max_dev = diameter_std_cap * std_d
            cap_d = mean_d + np.clip(cap_d - mean_d, -max_dev, max_dev)
        return mean_d + diameter_scale * (cap_d - mean_d)

    # get outlet areas
    if is_pulmonary:
        rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True)

        cap_info = lpa_info | rpa_info
    else:
        cap_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=False)
    
    # get the mean and standard deviation of the cap areas
    # lpa_areas = np.array(list(lpa_info.values()))
    # rpa_areas = np.array(list(rpa_info.values()))

    outlet_bc_names = [name for name, bc in config_handler.bcs.items() if 'inflow' not in bc.name.lower()]

    # assumed that cap and boundary condition orders match
    if len(outlet_bc_names) != len(cap_info):
        print('number of outlet boundary conditions does not match number of cap surfaces, automatically assigning bc names...')
        for i, name in enumerate(outlet_bc_names):
            # delete the unused bcs
            del config_handler.bcs[name]
        outlet_bc_names = [f'IMPEDANCE_{i}' for i in range(len(cap_info))]
    cap_to_bc = {list(cap_info.keys())[i]: outlet_bc_names[i] for i in range(len(outlet_bc_names))}
    if not hasattr(config_handler, "bc_inductance"):
        config_handler.bc_inductance = {}

    lpa_diameters = np.array([_cap_diameter(area) for area in lpa_info.values()]) if is_pulmonary else np.array([])
    rpa_diameters = np.array([_cap_diameter(area) for area in rpa_info.values()]) if is_pulmonary else np.array([])

    lpa_mean_dia = np.mean(lpa_diameters) if lpa_diameters.size > 0 else None
    rpa_mean_dia = np.mean(rpa_diameters) if rpa_diameters.size > 0 else None
    lpa_std_dia = np.std(lpa_diameters) if lpa_diameters.size > 0 else 0.0
    rpa_std_dia = np.std(rpa_diameters) if rpa_diameters.size > 0 else 0.0

    if use_mean:
        '''use the mean diameter of the cap surfaces to construct the lpa and rpa trees and use these trees for all outlets'''
        if specify_diameter:
            lpa_mean_dia = lpa_params.diameter
            rpa_mean_dia = rpa_params.diameter

        else:
            print(f'LPA mean diameter: {lpa_mean_dia}')
            print(f'RPA mean diameter: {rpa_mean_dia}')
            print(f'LPA std diameter: {lpa_std_dia}')
            print(f'RPA std diameter: {rpa_std_dia}')


        _, _, kernel_steps = resolve_impedance_timepoint_contract(
            config_handler.simparams.to_dict()
        )
        time_array = config_handler.inflows[next(iter(config_handler.inflows))].t

        lpa_tree = StructuredTree(name='LPA', time=time_array, simparams=config_handler.simparams, compliance_model=lpa_params.compliance_model)
        print(f'building LPA tree with lpa parameters: {lpa_params.summary()}')

        lpa_tree.build(
            initial_d=lpa_mean_dia,
            d_min=lpa_params.d_min,
            lrr=lpa_params.lrr,
            alpha=lpa_params.alpha,
            beta=lpa_params.beta,
            xi=lpa_params.xi,
            eta_sym=lpa_params.eta_sym,
        )
        lpa_tree.compute_olufsen_impedance(n_procs=n_procs, tsteps=kernel_steps)
        lpa_tree.plot_stiffness(path='lpa_stiffness_plot.png')

        rpa_tree = StructuredTree(name='RPA', time=time_array, simparams=config_handler.simparams, compliance_model=rpa_params.compliance_model)
        print(f'building RPA tree with rpa parameters: {rpa_params.summary()}')

        rpa_tree.build(
            initial_d=rpa_mean_dia,
            d_min=rpa_params.d_min,
            lrr=rpa_params.lrr,
            alpha=rpa_params.alpha,
            beta=rpa_params.beta,
            xi=rpa_params.xi,
            eta_sym=rpa_params.eta_sym,
        )
        rpa_tree.compute_olufsen_impedance(n_procs=n_procs, tsteps=kernel_steps)
        rpa_tree.plot_stiffness(path='rpa_stiffness_plot.png')

        lpa_bc_names = []
        lpa_outlet_names = []
        rpa_bc_names = []
        rpa_outlet_names = []

        # distribute the impedance to lpa and rpa specifically
        for idx, (cap_name, area) in enumerate(cap_info.items()):
            print(f'generating tree {idx + 1} of {len(cap_info)} for cap {cap_name}...')
            if 'lpa' in cap_name.lower():
                bc_name = cap_to_bc[cap_name]
                config_handler.bcs[cap_to_bc[cap_name]] = lpa_tree.create_impedance_bc(
                    bc_name,
                    0,
                    wedge_pressure * 1333.2,
                )
                config_handler.bc_inductance[bc_name] = lpa_params.inductance
                lpa_bc_names.append(bc_name)
                lpa_outlet_names.append(cap_name)
            elif 'rpa' in cap_name.lower():
                bc_name = cap_to_bc[cap_name]
                config_handler.bcs[cap_to_bc[cap_name]] = rpa_tree.create_impedance_bc(
                    bc_name,
                    1,
                    wedge_pressure * 1333.2,
                )
                config_handler.bc_inductance[bc_name] = rpa_params.inductance
                rpa_bc_names.append(bc_name)
                rpa_outlet_names.append(cap_name)
            else:
                raise ValueError('cap name not recognized')

        config_handler.tree_params[lpa_tree.name] = _attach_tree_metadata(
            lpa_tree,
            lpa_params,
            generation_mode="shared_by_side",
            side="lpa",
            bc_names=lpa_bc_names,
            outlet_names=lpa_outlet_names,
        )
        config_handler.tree_params[rpa_tree.name] = _attach_tree_metadata(
            rpa_tree,
            rpa_params,
            generation_mode="shared_by_side",
            side="rpa",
            bc_names=rpa_bc_names,
            outlet_names=rpa_outlet_names,
        )
            
    else:
        '''build a unique tree for each outlet'''
        _, _, kernel_steps = resolve_impedance_timepoint_contract(
            config_handler.simparams.to_dict()
        )
        for idx, (cap_name, area) in enumerate(cap_info.items()):

            print(f'generating tree {idx} of {len(cap_info)} for cap {cap_name}...')
            cap_d = _cap_diameter(area)
            if 'lpa' in cap_name.lower():
                print(f'building tree with lpa parameters: {lpa_params.summary()}')
                params = lpa_params
                mean_d = lpa_mean_dia
                std_d = lpa_std_dia
            elif 'rpa' in cap_name.lower():
                print(f'building tree with rpa parameters: {rpa_params.summary()}')
                params = rpa_params
                mean_d = rpa_mean_dia
                std_d = rpa_std_dia
            else:
                raise ValueError('cap name not recognized')
            
            scaled_d = _scaled_diameter(cap_d, mean_d, std_d)
            tree = StructuredTree(name=cap_name, time=config_handler.bcs['INFLOW'].t, simparams=config_handler.simparams, compliance_model=params.compliance_model)

            tree.build(
                initial_d=scaled_d,
                d_min=params.d_min,
                lrr=params.lrr,
                alpha=params.alpha,
                beta=params.beta,
                xi=params.xi,
                eta_sym=params.eta_sym,
            )

            # compute the impedance in frequency domain
            tree.compute_olufsen_impedance(n_procs=n_procs, tsteps=kernel_steps)

            bc_name = cap_to_bc[cap_name]

            config_handler.bcs[bc_name] = tree.create_impedance_bc(
                bc_name,
                idx,
                wedge_pressure * 1333.2,
            )
            config_handler.bc_inductance[bc_name] = params.inductance
            config_handler.tree_params[tree.name] = _attach_tree_metadata(
                tree,
                params,
                generation_mode="per_outlet",
                side="lpa" if 'lpa' in cap_name.lower() else "rpa",
                bc_names=[bc_name],
                outlet_names=[cap_name],
            )


def assign_rcr_bcs(config_handler, 
                   mesh_surfaces_path, 
                   wedge_pressure,
                   rcr_params,
                   convert_to_cm=False, 
                   is_pulmonary=True):
    """ Assign RCR boundary conditions to the model based on the provided parameters.

    Args:
        config_handler: The configuration handler instance.
        mesh_surfaces_path: The path to the mesh surfaces.
        wedge_pressure: The wedge pressure to apply.
        rcr_params: The RCR parameters to use, usually the output from the bc tuning algorithm structured as [R_LPA, C_LPA, R_RPA, C_RPA].
        convert_to_cm: Whether to convert units to cm.
        is_pulmonary: Whether the model is pulmonary.
        n_procs: The number of processes to use.
    """

    # get outlet areas
    if is_pulmonary:
        rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True)

        cap_info = lpa_info | rpa_info
    else:
        cap_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=False)
    
    # get the mean and standard deviation of the cap areas
    lpa_total_area = np.sum(np.array(list(lpa_info.values())))
    rpa_total_area = np.sum(np.array(list(rpa_info.values())))

    outlet_bc_names = [name for name, bc in config_handler.bcs.items() if 'inflow' not in bc.name.lower()]

    # assumed that cap and boundary condition orders match
    if len(outlet_bc_names) != len(cap_info):
        print('number of outlet boundary conditions does not match number of cap surfaces, automatically assigning bc names...')
        for i, name in enumerate(outlet_bc_names):
            # delete the unused bcs
            del config_handler.bcs[name]
        outlet_bc_names = [f'RCR_{i}' for i in range(len(cap_info))]

    cap_to_bc = {list(cap_info.keys())[i]: outlet_bc_names[i] for i in range(len(outlet_bc_names))}

    # build a unique tree for each outlet
    for idx, (cap_name, area) in enumerate(cap_info.items()):

        if 'lpa' in cap_name.lower():
            print(f'creating RCR BC for LPA with parameters: {rcr_params[:2]}')
            resistance, capacitance = rcr_params[:2]
            adjusted_resistance = resistance * (lpa_total_area / area)
            # create BC object
            rcr_bc = generate_outlet_rcr(adjusted_resistance, capacitance, wedge_pressure * 1333.2, f'RCR_{idx}')
        elif 'rpa' in cap_name.lower():
            print(f'creating RCR BC for RPA with parameters: {rcr_params[2:]}')
            resistance, capacitance = rcr_params[2:]
            adjusted_resistance = resistance * (rpa_total_area / area)
            # create BC object
            rcr_bc = generate_outlet_rcr(adjusted_resistance, capacitance, wedge_pressure * 1333.2, f'RCR_{idx}')
        else:
            raise ValueError('cap name not recognized')

        bc_name = cap_to_bc[cap_name]

        config_handler.bcs[bc_name] = rcr_bc
