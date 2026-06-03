from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import AdaptationModel
from ..utils import estimate_steady_tree_hemodynamics


class CWSSAdaptation(AdaptationModel):
    """Stable WSS-only structured-tree adaptation with frozen thickness."""

    event_reason_label = "rpa_split_window_converged"
    split_window_duration = 5.0
    min_window_coverage_fraction = 0.9
    min_window_samples = 5
    split_window_tolerance = 1e-3
    split_window_center_tolerance = 5e-4
    min_convergence_time = 10.0

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
        r_lpa, r_rpa = r_all[:n_lpa], r_all[n_lpa:]

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

        wss_h_lpa = _ensure_reference(lpa_tree, "homeostatic_wss", n_lpa, "LPA")
        wss_h_rpa = _ensure_reference(rpa_tree, "homeostatic_wss", n_rpa, "RPA")
        wss_h = np.concatenate([wss_h_lpa, wss_h_rpa])

        dydt = np.zeros_like(y)
        tau_err = tau - wss_h
        dydt[0::2] = self.K_arr[0] * tau_err * r_all

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
        """Terminate integration when the RPA split stabilizes over a trailing window."""
        _simple_pa = args[0]
        _reference_y = args[1]
        flow_log = args[2]
        _event_state = args[3]

        return self.convergence_margin(t, flow_log)

    def convergence_diagnostics(self, t, flow_log):
        diagnostics = {
            "t": float(t),
            "window_duration": float(self.split_window_duration),
            "window_coverage": 0.0,
            "window_samples": 0,
            "window_span": None,
            "center_deviation": None,
            "margin": None,
            "converged": False,
        }

        if t <= max(1e-12, float(self.min_convergence_time)):
            diagnostics["margin"] = float(self.min_convergence_time) - float(t)
            return diagnostics

        window_start = float(t) - float(self.split_window_duration)
        window_entries = [
            entry
            for entry in flow_log
            if window_start <= float(entry["t"]) <= float(t)
        ]
        diagnostics["window_samples"] = len(window_entries)
        if len(window_entries) < int(self.min_window_samples):
            diagnostics["margin"] = 1.0
            return diagnostics

        window_times = np.asarray([float(entry["t"]) for entry in window_entries], dtype=np.float64)
        window_splits = np.asarray([float(entry["rpa_split"]) for entry in window_entries], dtype=np.float64)
        coverage = float(window_times[-1] - window_times[0])
        diagnostics["window_coverage"] = coverage

        min_coverage = float(self.min_window_coverage_fraction) * float(self.split_window_duration)
        coverage_deficit = min_coverage - coverage
        if coverage_deficit > 0.0:
            diagnostics["margin"] = coverage_deficit
            return diagnostics

        window_span = float(np.max(window_splits) - np.min(window_splits))
        window_mean = float(np.mean(window_splits))
        latest_split = float(window_splits[-1])
        center_deviation = abs(window_mean - latest_split)

        span_margin = window_span - float(self.split_window_tolerance)
        center_margin = center_deviation - float(self.split_window_center_tolerance)
        margin = max(span_margin, center_margin)

        diagnostics["window_span"] = window_span
        diagnostics["center_deviation"] = center_deviation
        diagnostics["margin"] = margin
        diagnostics["converged"] = margin <= 0.0
        return diagnostics

    def convergence_margin(self, t, flow_log):
        """Return a signed margin for accepted-step rolling-window convergence."""
        return float(self.convergence_diagnostics(t, flow_log)["margin"])

    event.terminal = True
    event.direction = -1

    def event_outsidesim(self, t, y, *args):
        if t <= 1e-12:
            return 1.0
        last_y = args[1]
        rel_geom_r = np.mean(np.abs((y[0::2] - last_y[0::2]) / last_y[0::2]))
        return rel_geom_r - 1e-8
