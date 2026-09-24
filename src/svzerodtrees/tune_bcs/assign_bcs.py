
import numpy as np
from ..io import *
from ..io.blocks.boundary_condition import resolve_impedance_timepoint_contract
from ..utils import *
from ..microvasculature.structured_tree.structuredtree import StructuredTree
from ..microvasculature.structured_tree.dc_resistance import conductance_matched_diameter
from ..microvasculature.treeparams import TreeParameters
from ..simulation.threedutils import vtp_info
from .utils import *
from .outlet_mapping import (
    coerce_resolved_mapping,
    mapping_key,
    resolve_outlet_cap_mapping,
)

# this is where the logic for assigning boundary conditions to the 3D model is implemented


def _create_impedance_bc(tree, bc_name, outlet_id, pd, *, verbose):
    try:
        return tree.create_impedance_bc(
            bc_name,
            outlet_id,
            pd,
            verbose=verbose,
        )
    except TypeError as exc:
        if "verbose" not in str(exc):
            raise
        return tree.create_impedance_bc(bc_name, outlet_id, pd)


def _attach_tree_metadata(
    tree,
    params,
    *,
    generation_mode,
    side,
    bc_names,
    outlet_names,
    resolved_mapping=None,
    mapping_records=None,
    reference_diameter=None,
):
    tree.inductance = float(params.inductance)
    tree.generation_mode = generation_mode
    tree.outlet_mapping = mapping_payload = {
        "mode": generation_mode,
        "side": side,
        "bc_names": list(bc_names),
        "outlet_names": list(outlet_names),
    }
    if reference_diameter is not None:
        mapping_payload["reference_diameter"] = dict(reference_diameter)
    if resolved_mapping is not None:
        mapping_payload["strategy"] = resolved_mapping.strategy
        mapping_payload["pairs"] = [
            record.to_dict()
            for record in (mapping_records or resolved_mapping.records)
        ]
    return tree.to_dict()


def _mapping_key(value):
    return mapping_key(value)


def _outlet_bc_names(config_handler):
    return [
        name
        for name, bc in config_handler.bcs.items()
        if "inflow" not in str(getattr(bc, "name", name)).lower()
    ]


def _metadata_cap_to_bc(config_handler):
    mapping = {}
    for tree_payload in getattr(config_handler, "tree_params", {}).values():
        if not isinstance(tree_payload, dict):
            continue
        outlet_mapping = tree_payload.get("outlet_mapping")
        if not isinstance(outlet_mapping, dict):
            continue
        outlet_names = outlet_mapping.get("outlet_names") or []
        bc_names = outlet_mapping.get("bc_names") or []
        if len(outlet_names) == len(bc_names):
            pairs = zip(outlet_names, bc_names)
        elif len(bc_names) == 1:
            pairs = ((outlet_name, bc_names[0]) for outlet_name in outlet_names)
        else:
            continue
        for outlet_name, bc_name in pairs:
            mapping[_mapping_key(outlet_name)] = str(bc_name)
    return mapping


def resolve_cap_to_bc_mapping(
    config_handler,
    cap_info,
    *,
    bc_prefix,
    allow_ordered_outlet_mapping=False,
    outlet_mapping_mode=None,
    outlet_mapping=None,
    mode=None,
    explicit_mapping=None,
):
    """Resolve mesh cap names to 0D outlet BC names.

    This wrapper retains the historical dict-shaped return for direct callers;
    callers that need provenance should use ``resolve_outlet_cap_mapping``.
    """

    requested_mode = outlet_mapping_mode if outlet_mapping_mode is not None else mode
    requested_mode = requested_mode or "auto"
    if allow_ordered_outlet_mapping:
        if str(requested_mode).strip().lower() not in {
            "auto",
            "serialized_cap_order",
        }:
            raise ValueError("allow_ordered_outlet_mapping conflicts with mode")
        # Translate the deprecated direct-caller flag at this boundary.  The
        # canonical resolver receives only outlet_mapping_mode.
        requested_mode = "serialized_cap_order"

    try:
        resolved = resolve_outlet_cap_mapping(
            config_handler,
            cap_info,
            bc_prefix=bc_prefix,
            outlet_mapping_mode=requested_mode,
            outlet_mapping=outlet_mapping,
            explicit_mapping=explicit_mapping,
        )
    except ValueError as exc:
        raise ValueError(
            f"could not deterministically map mesh caps to outlet BCs: {exc}"
        ) from exc
    return dict(resolved.items())


def validate_cap_to_bc_mapping(
    config_handler,
    mesh_surfaces_path,
    *,
    outlet_mapping_mode="auto",
    outlet_mapping=None,
    resolved_mapping=None,
    centerline=None,
    seed_payload=None,
    convert_to_cm=False,
    is_pulmonary=False,
    bc_prefix=None,
):
    """Validate and resolve one cap-to-outlet mapping for a mesh.

    ``outlet_mapping_mode`` and ``outlet_mapping`` are the canonical inputs.
    ``centerline`` is the centerline VTP the 0D model was generated from; it
    enables geometric cap matching for ``auto`` and ``centerline`` modes, and
    ``seed_payload`` is the serialized 0D config it checks the centerline
    against.
    A pre-resolved mapping may be supplied when a caller has already frozen
    the identity for the remainder of an iteration, but it cannot be mixed
    with a second mapping request.  Legacy ordered mapping is translated by
    configuration/direct-call adapters before reaching this validator.
    """

    if outlet_mapping is not None and resolved_mapping is not None:
        raise ValueError(
            "outlet_mapping and resolved_mapping conflict and are mutually exclusive; "
            "provide only one mapping input"
        )

    if is_pulmonary:
        rpa_info, lpa_info, _ = vtp_info(
            mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True
        )
        cap_info = dict(sorted((lpa_info | rpa_info).items(), key=lambda item: str(item[0])))
    else:
        cap_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=False)

    if resolved_mapping is not None:
        return coerce_resolved_mapping(
            resolved_mapping,
            config_handler,
            cap_info,
            bc_prefix=bc_prefix,
        )

    return resolve_outlet_cap_mapping(
        config_handler,
        cap_info,
        bc_prefix=bc_prefix,
        outlet_mapping_mode=outlet_mapping_mode,
        outlet_mapping=outlet_mapping,
        centerline=centerline,
        seed_payload=seed_payload,
        convert_to_cm=convert_to_cm,
    )

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
                              diameter_std_cap=None,
                              reference_diameter="arithmetic_mean",
                              allow_ordered_outlet_mapping=False,
                              verbose=True,
                              plot_stiffness=True,
                              resolved_mapping=None,
                              mapping=None,
                              outlet_mapping=None,
                              cap_to_bc_mapping=None):
    '''
    construct impedance trees for outlet BCs
    
    :param k2: stiffness parameter 2
    :param k3: stiffness parameter 3
    :param use_mean: when True, build only two trees (LPA/RPA) and reuse for all outlets
    :param diameter_scale: for unique trees, shrink diameter spread toward the mean (0=all mean, 1=full spread)
    :param diameter_std_cap: optional cap in std deviations on diameter deviation before scaling
    :param reference_diameter: shared-tree diameter when use_mean is True.
        'arithmetic_mean' (default) uses the tree parameter diameter
        (specify_diameter) or the mean cap diameter.  'conductance_matched'
        uses the diameter whose DC conductance, repeated once per outlet,
        equals that of per-outlet trees at the diameters defined by
        diameter_scale/diameter_std_cap (see conductance_matched_diameter).
    :param plot_stiffness: write LPA/RPA stiffness plots when using shared trees'''

    if reference_diameter not in ("arithmetic_mean", "conductance_matched"):
        raise ValueError(
            "reference_diameter must be 'arithmetic_mean' or 'conductance_matched'"
        )
    if reference_diameter == "conductance_matched" and not (use_mean and is_pulmonary):
        raise ValueError(
            "reference_diameter='conductance_matched' requires use_mean=True "
            "and is_pulmonary=True"
        )

    # svZeroDSolver's steady-initial pass uses a fixed 10-step cycle. That is
    # incompatible with reconstructed IMPEDANCE kernels unless the production
    # model also happens to use 11 points per cycle.
    config_handler.simparams.steady_initial = False

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

        # vtp_info scans and sorts the source files before splitting by side.
        # Re-sort after recombining to retain that canonical filename order.
        cap_info = dict(sorted((lpa_info | rpa_info).items(), key=lambda item: str(item[0])))
    else:
        cap_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=False)
    
    # get the mean and standard deviation of the cap areas
    # lpa_areas = np.array(list(lpa_info.values()))
    # rpa_areas = np.array(list(rpa_info.values()))

    supplied_mapping = (
        resolved_mapping
        if resolved_mapping is not None
        else mapping
        if mapping is not None
        else outlet_mapping
        if outlet_mapping is not None
        else cap_to_bc_mapping
    )
    if supplied_mapping is None:
        mapping_mode = (
            "serialized_cap_order"
            if allow_ordered_outlet_mapping
            else "auto"
        )
        resolved_mapping = resolve_outlet_cap_mapping(
            config_handler,
            cap_info,
            bc_prefix="IMPEDANCE",
            outlet_mapping_mode=mapping_mode,
        )
    else:
        resolved_mapping = coerce_resolved_mapping(
            supplied_mapping,
            config_handler,
            cap_info,
            bc_prefix="IMPEDANCE",
        )
    cap_to_bc = resolved_mapping
    if not hasattr(config_handler, "bc_inductance"):
        config_handler.bc_inductance = {}

    lpa_diameters = np.array([_cap_diameter(area) for area in lpa_info.values()]) if is_pulmonary else np.array([])
    rpa_diameters = np.array([_cap_diameter(area) for area in rpa_info.values()]) if is_pulmonary else np.array([])

    lpa_mean_dia = np.mean(lpa_diameters) if lpa_diameters.size > 0 else None
    rpa_mean_dia = np.mean(rpa_diameters) if rpa_diameters.size > 0 else None
    lpa_std_dia = np.std(lpa_diameters) if lpa_diameters.size > 0 else 0.0
    rpa_std_dia = np.std(rpa_diameters) if rpa_diameters.size > 0 else 0.0

    side_reference = {"lpa": None, "rpa": None}
    if use_mean:
        '''use the mean diameter of the cap surfaces to construct the lpa and rpa trees and use these trees for all outlets'''
        if reference_diameter == "conductance_matched":
            # Match each side's total DC conductance to the per-outlet trees
            # that diameter_scale/diameter_std_cap would build.
            for side, params, diameters, mean_d, std_d in (
                ("lpa", lpa_params, lpa_diameters, lpa_mean_dia, lpa_std_dia),
                ("rpa", rpa_params, rpa_diameters, rpa_mean_dia, rpa_std_dia),
            ):
                targets = [_scaled_diameter(d, mean_d, std_d) for d in diameters]
                d_ref, residual = conductance_matched_diameter(
                    targets,
                    d_min=params.d_min,
                    alpha=params.alpha,
                    beta=params.beta,
                )
                side_reference[side] = {
                    "mode": "conductance_matched",
                    "diameter": float(d_ref),
                    "relative_conductance_residual": float(residual),
                    "arithmetic_mean_diameter": float(mean_d),
                    "diameter_scale": float(diameter_scale),
                    "diameter_std_cap": diameter_std_cap,
                }
            lpa_mean_dia = side_reference["lpa"]["diameter"]
            rpa_mean_dia = side_reference["rpa"]["diameter"]
        elif specify_diameter:
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
        print(f'building LPA tree at initial_d={lpa_mean_dia:.4f} with lpa parameters: {lpa_params.summary()}')

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
        if plot_stiffness:
            lpa_tree.plot_stiffness(path='lpa_stiffness_plot.png')

        rpa_tree = StructuredTree(name='RPA', time=time_array, simparams=config_handler.simparams, compliance_model=rpa_params.compliance_model)
        print(f'building RPA tree at initial_d={rpa_mean_dia:.4f} with rpa parameters: {rpa_params.summary()}')

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
        if plot_stiffness:
            rpa_tree.plot_stiffness(path='rpa_stiffness_plot.png')

        lpa_bc_names = []
        lpa_outlet_names = []
        rpa_bc_names = []
        rpa_outlet_names = []

        # distribute the impedance to lpa and rpa specifically
        for idx, record in enumerate(resolved_mapping.records):
            cap_name = record.cap_path
            area = record.area
            if verbose:
                print(f'generating tree {idx + 1} of {len(cap_info)} for cap {cap_name}...')
            if record.side == "lpa":
                bc_name = record.bc_name
                config_handler.bcs[bc_name] = _create_impedance_bc(
                    lpa_tree,
                    bc_name,
                    0,
                    wedge_pressure * 1333.2,
                    verbose=verbose,
                )
                config_handler.bc_inductance[bc_name] = lpa_params.inductance
                lpa_bc_names.append(bc_name)
                lpa_outlet_names.append(cap_name)
            elif record.side == "rpa":
                bc_name = record.bc_name
                config_handler.bcs[bc_name] = _create_impedance_bc(
                    rpa_tree,
                    bc_name,
                    1,
                    wedge_pressure * 1333.2,
                    verbose=verbose,
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
            resolved_mapping=resolved_mapping,
            mapping_records=[
                record.with_scaled_diameter(lpa_mean_dia)
                for record in resolved_mapping.records
                if record.side == "lpa"
            ],
            reference_diameter=side_reference["lpa"],
        )
        config_handler.tree_params[rpa_tree.name] = _attach_tree_metadata(
            rpa_tree,
            rpa_params,
            generation_mode="shared_by_side",
            side="rpa",
            bc_names=rpa_bc_names,
            outlet_names=rpa_outlet_names,
            resolved_mapping=resolved_mapping,
            mapping_records=[
                record.with_scaled_diameter(rpa_mean_dia)
                for record in resolved_mapping.records
                if record.side == "rpa"
            ],
            reference_diameter=side_reference["rpa"],
        )
            
    else:
        '''build a unique tree for each outlet'''
        _, _, kernel_steps = resolve_impedance_timepoint_contract(
            config_handler.simparams.to_dict()
        )
        for idx, record in enumerate(resolved_mapping.records):
            cap_name = record.cap_path
            area = record.area

            if verbose:
                print(f'generating tree {idx} of {len(cap_info)} for cap {cap_name}...')
            cap_d = _cap_diameter(area)
            if record.side == "lpa":
                if verbose:
                    print(f'building tree with lpa parameters: {lpa_params.summary()}')
                params = lpa_params
                mean_d = lpa_mean_dia
                std_d = lpa_std_dia
            elif record.side == "rpa":
                if verbose:
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

            bc_name = record.bc_name

            config_handler.bcs[bc_name] = _create_impedance_bc(
                tree,
                bc_name,
                idx,
                wedge_pressure * 1333.2,
                verbose=verbose,
            )
            config_handler.bc_inductance[bc_name] = params.inductance
            config_handler.tree_params[tree.name] = _attach_tree_metadata(
                tree,
                params,
                generation_mode="per_outlet",
                side=record.side,
                bc_names=[bc_name],
                outlet_names=[cap_name],
                resolved_mapping=resolved_mapping,
                mapping_records=[record.with_scaled_diameter(scaled_d)],
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

    if not is_pulmonary:
        raise ValueError("assign_rcr_bcs currently supports pulmonary LPA/RPA models only")

    # get outlet areas
    rpa_info, lpa_info, inflow_info = vtp_info(mesh_surfaces_path, convert_to_cm=convert_to_cm, pulmonary=True)
    cap_info = lpa_info | rpa_info
    
    # get the mean and standard deviation of the cap areas
    lpa_total_area = np.sum(np.array(list(lpa_info.values())))
    rpa_total_area = np.sum(np.array(list(rpa_info.values())))
    cap_to_bc = resolve_cap_to_bc_mapping(config_handler, cap_info, bc_prefix="RCR")

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
