from .base import AdaptationModel
from ..utils import estimate_steady_tree_hemodynamics
import numpy as np
from typing import Sequence

class CWSSIMSAdaptation(AdaptationModel):
    event_reason_label = "geometry_converged"

    def __init__(self, K_arr: Sequence[float]):
        if not isinstance(K_arr, (list, tuple, np.ndarray)):
            raise TypeError(f"K_arr must be a list, tuple, or ndarray, but got {type(K_arr)}.")
        if len(K_arr) != 4:
            raise ValueError(
                f"K_arr must contain exactly 4 elements: "
                f"[K_tau_r, K_sig_r, K_tau_h, K_sig_h], but got {len(K_arr)}."
            )
        super().__init__(K_arr)
    
    def compute_rhs(self, t, y, simple_pa, _vessels, last_update_y, last_t_holder, flow_log, solver_trace):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"State vector must be 1-D, received shape {y.shape}.")
        if not np.all(y > 0.0):
            bad_indices = np.where(y <= 0.0)[0]
            raise ValueError(
                f"Invalid values in y: all values must be > 0. "
                f"Found {len(bad_indices)} non-positive values at indices: {bad_indices.tolist()}"
            )

        lpa_tree = getattr(simple_pa, "lpa_tree", None)
        rpa_tree = getattr(simple_pa, "rpa_tree", None)
        if lpa_tree is None or rpa_tree is None:
            raise AttributeError("PA configuration is missing LPA or RPA structured tree objects.")
        if not hasattr(lpa_tree, "store") or lpa_tree.store is None:
            raise AttributeError("LPA structured tree has not been built; call build() before adaptation.")
        if not hasattr(rpa_tree, "store") or rpa_tree.store is None:
            raise AttributeError("RPA structured tree has not been built; call build() before adaptation.")

        n_lpa = int(lpa_tree.store.n_nodes())
        n_rpa = int(rpa_tree.store.n_nodes())
        n_total = n_lpa + n_rpa
        if y.size != 2 * n_total:
            raise ValueError(
                f"State vector length {y.size} != 2 * number of vessels ({n_total}). "
                "Ensure pack_state reflects StructuredTree storage ordering."
            )

        r_all = y[0::2]
        h_all = y[1::2]


        r_lpa, r_rpa = r_all[:n_lpa], r_all[n_lpa:]
        h_lpa, h_rpa = h_all[:n_lpa], h_all[n_lpa:]

        # Update structured-tree diameters in-place to reflect current radii.
        lpa_tree.store.d = (2.0 * r_lpa).astype(lpa_tree.store.d.dtype, copy=False)
        rpa_tree.store.d = (2.0 * r_rpa).astype(rpa_tree.store.d.dtype, copy=False)

        simple_pa.update_bcs()
        simple_pa.simulate()

        geom_rel_change = (y - last_update_y) / last_update_y
        geom_change = float(np.mean(np.abs(geom_rel_change)))

        def _ensure_reference(tree, attr_name, n_expected, label):
            ref = getattr(tree, attr_name, None)
            if ref is None:
                map_attr = getattr(tree, f"_{attr_name}_map", None)
                if map_attr is not None:
                    ids = np.asarray(tree.store.ids, dtype=np.int32)
                    return np.array([map_attr[int(i)] for i in ids], dtype=np.float64)
                raise AttributeError(f"{label} tree is missing required attribute '{attr_name}'.")
            if isinstance(ref, dict):
                ids = np.asarray(tree.store.ids, dtype=np.int32)
                return np.array([ref[int(i)] for i in ids], dtype=np.float64)
            arr = np.asarray(ref, dtype=np.float64)
            if arr.size != n_expected:
                raise ValueError(
                    f"{label} tree attribute '{attr_name}' has size {arr.size}, expected {n_expected}."
                )
            return arr

        lpa_root_flow = float(
            np.mean(simple_pa.result[simple_pa.result.name == "branch2_seg0"]["flow_out"])
        )
        rpa_root_flow = float(
            np.mean(simple_pa.result[simple_pa.result.name == "branch4_seg0"]["flow_out"])
        )
        distal_pressure = float(simple_pa.clinical_targets.wedge_p) * 1333.2

        lpa_hemo = estimate_steady_tree_hemodynamics(
            lpa_tree,
            root_flow=lpa_root_flow,
            distal_pressure=distal_pressure,
        )
        rpa_hemo = estimate_steady_tree_hemodynamics(
            rpa_tree,
            root_flow=rpa_root_flow,
            distal_pressure=distal_pressure,
        )

        tau_lpa = lpa_hemo.wall_shear_stress
        tau_rpa = rpa_hemo.wall_shear_stress
        tau = np.concatenate([tau_lpa, tau_rpa])

        P_lpa = lpa_hemo.pressure_in
        P_rpa = rpa_hemo.pressure_in
        h_safe = np.maximum(h_all, 1e-9)
        sig_lpa = P_lpa * (r_lpa / h_safe[:n_lpa])
        sig_rpa = P_rpa * (r_rpa / h_safe[n_lpa:])
        sig = np.concatenate([sig_lpa, sig_rpa])

        wss_h_lpa = _ensure_reference(lpa_tree, "homeostatic_wss", n_lpa, "LPA")
        wss_h_rpa = _ensure_reference(rpa_tree, "homeostatic_wss", n_rpa, "RPA")
        ims_h_lpa = _ensure_reference(lpa_tree, "homeostatic_ims", n_lpa, "LPA")
        ims_h_rpa = _ensure_reference(rpa_tree, "homeostatic_ims", n_rpa, "RPA")

        wss_h = np.concatenate([wss_h_lpa, wss_h_rpa])
        ims_h = np.concatenate([ims_h_lpa, ims_h_rpa])

        dydt = np.empty_like(y)
        tau_err = tau - wss_h
        sig_err = sig - ims_h
        dydt[0::2] = self.K_arr[0] * tau_err + self.K_arr[1] * sig_err
        dydt[1::2] = -self.K_arr[2] * tau_err + self.K_arr[3] * sig_err

        if t > max(last_t_holder[0], 1e-12):
            rhs_l2 = float(np.linalg.norm(dydt))
            trace_entry = {
                "t": float(t),
                "geom_change_mean": geom_change,
                "rpa_split": float(simple_pa.rpa_split),
                "rhs_l2": rhs_l2,
            }
            print(
                f"Geometry change at t={t:.6f} mean: {geom_change:.3e} "
                f"and rpa split: {simple_pa.rpa_split:.6f} (rhs_l2={rhs_l2:.3e})"
            )
            last_update_y[:] = y
            last_t_holder[0] = float(t)
            flow_log.append({"t": float(t), "rpa_split": float(simple_pa.rpa_split)})
            solver_trace.append(trace_entry)
            if len(flow_log) > 1:
                prev = float(flow_log[-2]["rpa_split"])
                curr = float(flow_log[-1]["rpa_split"])
                rel_change = (curr - prev) / curr if curr != 0.0 else "N/A"
            else:
                rel_change = "N/A"
            print(f" -> flow split change: {rel_change}")

        return dydt


    def event(self, t, y, *args):
        """
        Terminates integration when EITHER geometry OR flow split change is small.
        Guarantees a sign change by tracking sign history.
        """

        # Unpack args
        simple_pa = args[0]
        reference_y = args[1]
        flow_log = args[2]
        event_state = args[3]

        if t <= 1e-12:
            return 1.0

        rpa_split = simple_pa.rpa_split

        # Geometry relative change
        rel_geom_r = np.mean(np.abs((y[0::2] - reference_y[0::2]) / reference_y[0::2]))
        rel_geom_h = np.mean(np.abs((y[1::2] - reference_y[1::2]) / reference_y[1::2]))
        geom_change = max(rel_geom_r, rel_geom_h)

        # Flow split relative change
        if len(flow_log) > 1:
            prev_split = float(flow_log[-2]["rpa_split"])
            flow_split_change = abs(prev_split - rpa_split) / abs(rpa_split)
        else:
            flow_split_change = 1.0

        # Tolerances
        geom_tol = 1e-6 # 1e-5 was better?
        split_tol = 1e-4

        # Determine convergence
        converged = geom_change - geom_tol < 0 # or flow_split_change - split_tol < 0

        if converged:
            event_state["triggered"] = True

        # # Track if function was ever positive
        if not event_state["triggered"]:
            event_state["was_positive"] = True
            val = 1.0  # Safe positive value
        elif event_state["was_positive"]:
            event_state["was_positive"] = False
            val = -1.0  # Force sign change
        else:
            event_state["was_positive"] = True
            val = 1.0  # Defensive: return positive if it was never positive
        
        # val = 1.0 if not triggered[0] else (-1.0 if was_positive[0] else 1.0)
        # print(f"[event] t={t:.4f}, geom_change={geom_change:.2e}, flow_split_change={flow_split_change:.2e}, val={val}")

        return val

    event.terminal = True
    event.direction = -1

    def event_outsidesim(self, t, y, *args):
        """
        Event function for adaptation that checks if the geometry or flow split change is small.
        This is used when the adaptation is run without simulating the model.
        """
        if t <= 1e-12:
            return 1.0
        last_y = args[1]

        rel_geom_r = np.mean(np.abs((y[0::2] - last_y[0::2]) / last_y[0::2]))
        rel_geom_h = np.mean(np.abs((y[1::2] - last_y[1::2]) / last_y[1::2]))

        geom_change = max(rel_geom_r, rel_geom_h)

        return geom_change - 1e-8  # Adjusted tolerance for event

    event_outsidesim.terminal = True
    event_outsidesim.direction = -1
