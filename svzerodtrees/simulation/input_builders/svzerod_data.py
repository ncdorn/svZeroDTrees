from .simulation_file import SimulationFile
import numpy as np
import pandas as pd
import math
import warnings


class SvZeroDdata(SimulationFile):
    '''
    class to handle the svZeroD_data file in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the svZeroD_data object'''
        super().__init__(path)

    def initialize(self):
        '''
        initialize the svZeroD_data object'''
        self.df = pd.read_csv(self.path, sep='\s+')

        self.df.rename({self.df.columns[0]: 'time'}, axis=1, inplace=True)

    def write(self):
        '''
        write the svZeroD_data file'''

        pass

    def get_result(
        self,
        block,
        cycle_duration: float = 1.0,
        window: str = "last",
        full_cycle: bool = True,
        n_tsteps: int = 500,
        output_series: bool = False,
        last_cycle_only=None,
        last_full_cycle=None,
    ):
        """
        Get pressure and flow from svZeroD_data for a given CouplingBlock, with optional
        extraction of a final-cycle window and uniform resampling.

        Windowing behavior:
        - window="last":
            * If full_cycle=True, align to the last full cycle of length cycle_duration.
            * If full_cycle=False, take the last cycle_duration worth of data even if incomplete.
        - window="all": use the entire available time span.
        Resampling: outputs exactly n_tsteps samples on [0, T_out), endpoint=False.

        Returns
        -------
        t_out : (n_tsteps,) np.ndarray
            Resampled time in seconds, starting at 0 up to < T_out (cycle_duration for
            window="last", full span otherwise).
        q_out : (n_tsteps,) np.ndarray
            Resampled flow aligned to t_out.
        p_out : (n_tsteps,) np.ndarray
            Resampled pressure aligned to t_out.
        """
        df = self.df

        # Backward compatibility shims
        if last_cycle_only is not None:
            warnings.warn(
                "last_cycle_only is deprecated; use window='last' or window='all' instead.",
                DeprecationWarning,
            )
            window = "last" if last_cycle_only else "all"
        if last_full_cycle is not None:
            warnings.warn(
                "last_full_cycle is deprecated; use full_cycle instead.",
                DeprecationWarning,
            )
            full_cycle = last_full_cycle

        window = window.lower().strip()
        if window not in {"last", "all"}:
            raise ValueError("window must be 'last' or 'all'.")

        # Resolve time column (allow integer-named first column fallback)
        time_col = "time" if "time" in df.columns else df.columns[0]
        t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

        # Resolve flow/pressure column names by block location
        if block.location == "inlet":
            flow_col = f"flow:{block.name}:{block.connected_block}"
            pres_col = f"pressure:{block.name}:{block.connected_block}"
        elif block.location == "outlet":
            flow_col = f"flow:{block.connected_block}:{block.name}"
            pres_col = f"pressure:{block.connected_block}:{block.name}"
        else:
            raise ValueError(f"Unknown block.location '{block.location}' (expected 'inlet' or 'outlet').")

        if flow_col not in df.columns or pres_col not in df.columns:
            raise KeyError(f"Missing expected columns: {flow_col!r} / {pres_col!r}")

        q_raw = pd.to_numeric(df[flow_col], errors="coerce").to_numpy()
        p_raw = pd.to_numeric(df[pres_col], errors="coerce").to_numpy()

        # Sort by time and drop duplicate timestamps
        order = np.argsort(t_raw)
        t_raw = t_raw[order]
        q_raw = q_raw[order]
        p_raw = p_raw[order]
        keep = np.r_[True, np.diff(t_raw) > 0]
        t_raw = t_raw[keep]
        q_raw = q_raw[keep]
        p_raw = p_raw[keep]

        if t_raw.size < 2 or not np.isfinite(t_raw).all():
            raise RuntimeError("Invalid or insufficient time samples in svZeroD_data.")

        if window == "last":
            T = float(cycle_duration)
            if not np.isfinite(T) or T <= 0:
                raise ValueError("cycle_duration must be positive and finite.")

            t_ref = float(t_raw[0])
            t_end_raw = float(t_raw[-1])

            if full_cycle:
                # Estimate the nominal timestep so we can allow for a missing final sample.
                dt_nominal = 0.0
                t_diffs = np.diff(t_raw)
                if t_diffs.size:
                    pos = t_diffs[(t_diffs > 0) & np.isfinite(t_diffs)]
                    if pos.size:
                        dt_nominal = float(np.median(pos))

                # Align end to the nearest *earlier* multiple of T relative to t_ref
                tolerance = dt_nominal if dt_nominal > 0 else 0.0
                k = math.floor(((t_end_raw - t_ref) + tolerance) / T)
                if k < 1:
                    # Not even one full cycle present; fall back to earliest possible [t_ref, t_ref+T)
                    t_end = t_ref + T
                    t_start = t_ref
                    # but clamp to available data
                    t_end = min(t_end, t_end_raw)
                else:
                    t_end = t_ref + k * T
                    t_start = t_end - T

                # Select strictly within [t_start, t_end)
                sel = (t_raw >= t_start) & (t_raw < t_end)
                t_win = t_raw[sel]
                q_win = q_raw[sel]
                p_win = p_raw[sel]

                # If too few points (e.g., sparse sampling at boundary), step back by whole cycles if possible
                while t_win.size < 2 and (t_start - T) >= t_ref:
                    t_end -= T
                    t_start -= T
                    sel = (t_raw >= t_start) & (t_raw < t_end)
                    t_win = t_raw[sel]
                    q_win = q_raw[sel]
                    p_win = p_raw[sel]

                # Final guard: if still <2 points, use whatever we have (last segment), but keep duration T
                if t_win.size < 2:
                    t_win = t_raw
                    q_win = q_raw
                    p_win = p_raw
                    # redefine window origin so interpolation below is well-defined
                    t_start = float(t_win[0])

                # Normalize to [0, T) for interpolation target
                t0 = t_start
                T_out = T
            else:
                # Take the last cycle_duration of data even if it is incomplete.
                t_end = t_end_raw
                t_start = t_end - T
                sel = t_raw >= t_start
                t_win = t_raw[sel]
                q_win = q_raw[sel]
                p_win = p_raw[sel]

                if t_win.size < 2:
                    t_win = t_raw
                    q_win = q_raw
                    p_win = p_raw
                    t_start = float(t_win[0])

                t0 = t_start
                T_out = T
        else:
            # Use entire span; resample across its duration
            t0 = float(t_raw[0])
            T_out = float(t_raw[-1] - t0)
            if not np.isfinite(T_out) or T_out <= 0:
                T_out = float(cycle_duration)
            t_win, q_win, p_win = t_raw, q_raw, p_raw

        # Build uniform output grid [0, T_out) with n_tsteps samples (endpoint=False avoids duplicate start/end)
        n = int(n_tsteps) if n_tsteps and n_tsteps > 1 else 500
        t_out = np.linspace(0.0, T_out, n, endpoint=False)

        # Interpolate onto t_out
        tau = t_win - t0
        ord2 = np.argsort(tau)
        tau = tau[ord2]
        q_win = q_win[ord2]
        p_win = p_win[ord2]
        uniq = np.r_[True, np.diff(tau) > 0]
        tau = tau[uniq]
        q_win = q_win[uniq]
        p_win = p_win[uniq]

        def _interp_safe(x_t, y, x_out):
            if x_t.size >= 2 and np.isfinite(y).sum() >= 2:
                return np.interp(x_out, x_t, y)
            if np.isfinite(y).any():
                last = y[np.isfinite(y)][-1]
                return np.full_like(x_out, float(last))
            return np.zeros_like(x_out)

        q_out = _interp_safe(tau, q_win, t_out)
        p_out = _interp_safe(tau, p_win, t_out)

        if output_series:
            return pd.Series(t_out), pd.Series(q_out), pd.Series(p_out)
        else:
            return t_out, q_out, p_out

    def get_flow(self, block):
        '''
        integrate the flow at the outlet over the last period
        
        :coupling_block: name of the coupling block
        :block_name: name of the block to integrate the flow over'''

        time, flow, pressure = self.get_result(block)

        # only get times and flows over the last cardiac period 1.0s
        if time.max() > 1.0:
            # unsteady simulation, get last period of the pandas dataframd
            time = time[time > time.max() - 1.0]
            # use the indices of the time to get the flow
            flow = flow[time.index]
            return np.trapz(flow, time)
        else:
            # steady simulation, only get last flow value in the pandas dataframe
            flow = flow[-1]
            return flow
