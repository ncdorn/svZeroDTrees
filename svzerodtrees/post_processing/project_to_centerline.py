import vtk
import os
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk
import svzerodplus
from svsuperestimator.reader._centerline_handler import CenterlineHandler


def get_branch_result(qoi, result_handler, config_handler):
    '''get the desired quantity of interest in branch result form 
     
    Args:
        qoi (str): quantity of interest
        result_handler (ResultHandler): result handler
        config_handler (ConfigHandler): config handler

    Returns:
        branch_result (dict): dictionary of branch results
    '''

    result = {}

    branches = list(result_handler.clean_results.keys())

    if qoi == 'flow':
        pass
    elif qoi == 'pressure':
        pass
    elif qoi == 'resistance':
        pass
    elif qoi == 'flow adaptation':
        pass
    elif qoi == 'pressure adaptation':
        pass
    else:
        raise Exception('qoi not recognized')
    
    results = result_handler.format_result_for_cl_projection(timestep)

    pass





def _map_0d_on_centerline(centerline, config_handler, result_handler, timestep, output_folder):
        """Map 0D result on centerline.

        TODO: This functions has been mainly copied from SimVascular, and has now been adopted from svsuperestimator. A cleanup
            would increase the readability a lot.

        Args:
            qoi (str): Quantity of interest to map. Can be "flow" "pressure" "resistance" or adaptation thereof.
            centerline (str): Path to centerline file.
            config_handler (ConfigHandler): Config handler.
            result_handler (ResultHandler): Result handler.
            timestep (str): Timestep to map. Can be "preop" "postop" "final" or "adaptation"
            output_folder (str): Path to output folder.
        
        Returns:
            None
        """

        print("Mapping 0D solution on centerline")

        # create centerline handler
        # cl_handler = reader.CenterlineHandler.from_file(centerline)
        cl_handler = CenterlineHandler.from_file(centerline)

        # unformatted result in the result_handler should already be in branch result form
        # this is where we need to figure out exactly what we will be projecting onto the centerline
         

        # centerline points
        points = cl_handler.points
        branch_ids = cl_handler.get_point_data_array("BranchId")
        path = cl_handler.get_point_data_array("Path")
        cl_id = cl_handler.get_point_data_array("CenterlineId")
        bif_id = cl_handler.get_point_data_array("BifurcationId")

        # all branch ids in centerline
        ids_cent = np.unique(branch_ids).tolist()
        ids_cent.remove(-1)

        # get zero d result
        results = result_handler.format_result_for_cl_projection(timestep)
        results["time"] = config_handler.get_time_series()

        # add path distance
        for vessel in config_handler.config["vessels"]:
            br, seg = vessel["vessel_name"].split("_")
            br = int(br[6:])
            seg = int(seg[3:])
            if seg == 0:
                results["distance"][br] = [0, 0]
            l_new = (
                results["distance"][br][-1] + vessel["vessel_length"]
            )
            results["distance"][br][1] += l_new

        # assemble output dict
        def rec_dd() -> defaultdict:
            return defaultdict(rec_dd)

        arrays = rec_dd()

        # loop all result fields
        for f in ["flow", "pressure"]:
            if f not in results:
                continue

            # check if ROM branch has same ids as centerline
            ids_rom = list(results[f].keys())
            ids_rom.sort()
            assert (
                ids_cent == ids_rom
            ), "Centerline and ROM results have different branch ids"

            # initialize output arrays
            array_f = np.zeros((path.shape[0], len(results["time"])))
            n_outlet = np.zeros(path.shape[0])

            # loop all branches
            for br in results[f].keys():
                # results of this branch
                res_br = results[f][br]

                # get centerline path
                path_cent = path[branch_ids == br]

                # get node locations from 0D results
                path_1d_res = results["distance"][br]
                f_res = res_br

                # interpolate ROM onto centerline
                # limit to interval [0,1] to avoid extrapolation error interp1d
                # due to slightly incompatible lenghts
                f_cent = np.asarray(interp1d(np.asarray(path_1d_res) / path_1d_res[-1], np.asarray(f_res).T)(
                    path_cent / path_cent[-1]
                )).T

                # store results of this path
                array_f[branch_ids == br] = f_cent

                # add upstream part of branch within junction
                if br == 0:
                    continue

                # first point of branch
                ip = np.where(branch_ids == br)[0][0]

                # centerline that passes through branch (first occurence)
                cid = np.where(cl_id[ip])[0][0]

                # id of upstream junction
                jc = bif_id[ip - 1]

                # centerline within junction
                is_jc = bif_id == jc
                jc_cent = np.where(np.logical_and(is_jc, cl_id[:, cid]))[0]

                # length of centerline within junction
                jc_path = np.append(
                    0,
                    np.cumsum(
                        np.linalg.norm(
                            np.diff(points[jc_cent], axis=0), axis=1
                        )
                    ),
                )
                jc_path /= jc_path[-1]

                # results at upstream branch
                res_br_u = results[f][branch_ids[jc_cent[0] - 1]]

                # results at beginning and end of centerline within junction
                f0 = res_br_u[-1]
                f1 = res_br[0]

                # map 1d results to centerline using paths
                array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T)(
                    jc_path
                ).T

                # count number of outlets of this junction
                n_outlet[jc_cent] += 1

            # normalize results within junctions by number of junction outlets
            is_jc = n_outlet > 0
            array_f[is_jc] = (array_f[is_jc].T / n_outlet[is_jc]).T

            # assemble time steps
            arrays[f] = array_f[:, 0]

        # add arrays to centerline and write to file
        for f, a in arrays.items():
            out_array = numpy_to_vtk(a)
            out_array.SetName(f)
            cl_handler.data.GetPointData().AddArray(out_array)

        target = os.path.join(output_folder, "initial_centerline.vtp")
        cl_handler.to_file(target)