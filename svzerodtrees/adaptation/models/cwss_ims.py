from .base import AdaptationModel
from ..utils import unpack_state, simulate_outlet_trees
import numpy as np
from typing import Sequence

class CWSSIMSAdaptation(AdaptationModel):
    
    def __init__(self, K_arr: Sequence[float]):
        if not isinstance(K_arr, (list, tuple, np.ndarray)):
            raise TypeError(f"K_arr must be a list, tuple, or ndarray, but got {type(K_arr)}.")
        if len(K_arr) != 4:
            raise ValueError(
                f"K_arr must contain exactly 4 elements: "
                f"[K_tau_r, K_sig_r, K_tau_h, K_sig_h], but got {len(K_arr)}."
            )
        super().__init__(K_arr)
    
    def compute_rhs(self, t, y, simple_pa, vessels, last_update_y, last_t_holder, flow_log):
        if not np.all(y > 0):
            bad_indices = np.where(y <= 0)[0]
            raise ValueError(
                f"Invalid values in y: all values must be > 0. "
                f"Found {len(bad_indices)} non-positive values at indices: {bad_indices.tolist()}"
            )
        unpack_state(y, vessels)
        simple_pa.update_bcs()
        simple_pa.simulate()

        geom_rel_change = (y - last_update_y) / last_update_y
        geom_change = np.max(np.abs(geom_rel_change))
        if geom_change > 1e-3 and t > last_t_holder[0] + 1e-12:
            max_idx = np.argmax(np.abs(geom_rel_change))
            max_val = y[max_idx]
            max_prev_val = last_update_y[max_idx]
            max_change = geom_rel_change[max_idx]
            print(f"Geometry change at t={t:.2f} s: {geom_change:.3e} and rpa split: {simple_pa.rpa_split:.3f}")
            print(f" -> Max change at index {max_idx}: from y = {max_prev_val:.4e} to y = {max_val:.4e} with change {abs(max_change):.3e}")
            simulate_outlet_trees(simple_pa)
            last_update_y[:] = y
            last_t_holder[0] = t
            flow_log.append((t, simple_pa.rpa_split))
            print(f" -> flow split change: {(flow_log[-1][1] - flow_log[-2][1]) / flow_log[-1][1] if len(flow_log) > 1 else 'N/A'}")

        mu = 0.04
        dydt = np.empty_like(y)
        for v in vessels:
            base = 2*v.idx
            tau = np.mean(4*mu*v.Q/(np.pi*v.r**3))
            sig = np.mean(v.P_in*v.r/v.h)
            dydt[base]   = self.K_arr[0]*(tau - v.wss_h) + self.K_arr[1]*(sig - v.ims_h)
            dydt[base+1] = -self.K_arr[2]*(tau - v.wss_h) + self.K_arr[3]*(sig - v.ims_h)
        return dydt
    
    def event(self, t, y, *args, triggered=[False], was_positive=[False]):
        """
        Terminates integration when EITHER geometry OR flow split change is small.
        Guarantees a sign change by tracking sign history.
        """

        # Unpack args
        simple_pa = args[0]
        last_y = args[1]
        flow_log = args[2]

        rpa_split = simple_pa.rpa_split

        # Geometry relative change
        rel_geom_r = np.max(np.abs((y[0::2] - last_y[0::2]) / last_y[0::2]))
        rel_geom_h = np.max(np.abs((y[1::2] - last_y[1::2]) / last_y[1::2]))
        geom_change = max(rel_geom_r, rel_geom_h)

        # Flow split relative change
        if len(flow_log) > 1:
            prev_split = flow_log[-2][1]
            flow_split_change = abs(prev_split - rpa_split) / abs(rpa_split)
        else:
            flow_split_change = 1.0

        # Tolerances
        geom_tol = 1e-6
        split_tol = 1e-4

        # Determine convergence
        converged = geom_change - geom_tol < 0 or flow_split_change - split_tol < 0

        if converged:
            triggered[0] = True

        # # Track if function was ever positive
        if not triggered[0]:
            was_positive[0] = True
            val = 1.0  # Safe positive value
        elif was_positive[0]:
            was_positive[0] = False
            val = -1.0  # Force sign change
        else:
            was_positive[0] = True
            val = 1.0  # Defensive: return positive if it was never positive
        
        # val = 1.0 if not triggered[0] else (-1.0 if was_positive[0] else 1.0)
        print(f"[event] t={t:.4f}, geom_change={geom_change:.2e}, flow_split_change={flow_split_change:.2e}, val={val}")

        return val

    event.terminal = True
    event.direction = -1

    # def event(self, t, y, *args, prev_phi=[np.inf], prev_t=-np.inf, step_id=[-1]):
    #     geom_change = max(
    #         np.max(np.abs((y[0::2] - args[1][0::2]) / args[1][0::2])),   # radii
    #         np.max(np.abs((y[1::2] - args[1][1::2]) / args[1][1::2]))  # thickness (10× looser)
    #     )

    #     flow_split_change = np.abs(args[2][-1][1] - args[0].rpa_split) / args[0].rpa_split if len(args[2]) > 1 else 1


    #     phi = args[0].rpa_split
    #     dphi = abs(phi - prev_phi[0]) / abs(phi)
    #     prev_phi[0] = phi

    #     print(f"  ---> flow_split_change in event: {flow_split_change:.3e}")

    #     # make a condition where when the flow split tolerannce is reached we adjust GEOM CHANGE so that things have the different sign!
    #     if flow_split_change < 1e-4:
    #         geom_change = 1e-7

    #     # return max(geom_change - 1e-5, flow_split_change - 1e-3)
    #     return geom_change - 1e-6

    # event.terminal = True
    # event.direction = -1
