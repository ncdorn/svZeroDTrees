"""Minimal toy adaptation systems for debugging feedback intuition."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


_EPS = 1e-12


def _as_2vector(values, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{label} must contain exactly two entries, received shape {arr.shape}.")
    if not np.all(arr > 0.0):
        raise ValueError(f"{label} must be strictly positive.")
    return arr


def _as_4vector(values, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"{label} must contain exactly four entries, received shape {arr.shape}.")
    return arr


def _as_nonnegative_2vector(values, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{label} must contain exactly two entries, received shape {arr.shape}.")
    if not np.all(arr >= 0.0):
        raise ValueError(f"{label} must be nonnegative.")
    return arr


def parallel_branch_no_load_hemodynamics(
    radii,
    *,
    total_flow: float,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
):
    """Compute steady hemodynamics for two parallel resistive branches with no distal load."""
    radii_arr = _as_2vector(radii, label="radii")
    lengths_arr = _as_2vector(lengths, label="lengths")
    total_flow = float(total_flow)
    viscosity = float(viscosity)
    distal_pressure = float(distal_pressure)
    if viscosity <= 0.0:
        raise ValueError("viscosity must be > 0.")

    radii_safe = np.maximum(radii_arr, 1.0e-50)
    resistance = 8.0 * viscosity * lengths_arr / (np.pi * radii_safe ** 4)
    conductance = 1.0 / resistance
    total_conductance = float(np.sum(conductance))
    if total_conductance <= 0.0:
        raise ValueError("total conductance must be > 0.")

    flow = total_flow * conductance / total_conductance
    inlet_pressure = distal_pressure + total_flow / total_conductance
    shear = 4.0 * viscosity * flow / (np.pi * np.maximum(radii_safe ** 3, _EPS))

    return {
        "radii": radii_arr,
        "lengths": lengths_arr,
        "resistance": resistance,
        "conductance": conductance,
        "flow": flow,
        "split": flow / max(total_flow, _EPS),
        "wall_shear_stress": shear,
        "inlet_pressure": float(inlet_pressure),
        "distal_pressure": distal_pressure,
    }


def parallel_branch_terminal_resistance_hemodynamics(
    radii,
    *,
    total_flow: float,
    terminal_resistances,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
):
    """Compute steady hemodynamics for two parallel branches with series terminal loads."""
    radii_arr = _as_2vector(radii, label="radii")
    lengths_arr = _as_2vector(lengths, label="lengths")
    terminal_resistance_arr = _as_nonnegative_2vector(terminal_resistances, label="terminal_resistances")
    total_flow = float(total_flow)
    viscosity = float(viscosity)
    distal_pressure = float(distal_pressure)
    if viscosity <= 0.0:
        raise ValueError("viscosity must be > 0.")

    radii_safe = np.maximum(radii_arr, 1.0e-50)
    vessel_resistance = 8.0 * viscosity * lengths_arr / (np.pi * radii_safe ** 4)
    total_resistance = vessel_resistance + terminal_resistance_arr
    conductance = 1.0 / total_resistance
    total_conductance = float(np.sum(conductance))
    if total_conductance <= 0.0:
        raise ValueError("total conductance must be > 0.")

    flow = total_flow * conductance / total_conductance
    inlet_pressure = distal_pressure + total_flow / total_conductance
    shear = 4.0 * viscosity * flow / (np.pi * np.maximum(radii_safe ** 3, _EPS))

    return {
        "radii": radii_arr,
        "lengths": lengths_arr,
        "vessel_resistance": vessel_resistance,
        "terminal_resistance": terminal_resistance_arr,
        "resistance": total_resistance,
        "conductance": conductance,
        "flow": flow,
        "split": flow / max(total_flow, _EPS),
        "wall_shear_stress": shear,
        "inlet_pressure": float(inlet_pressure),
        "distal_pressure": distal_pressure,
    }


def parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
    adaptive_radii,
    *,
    upstream_radii,
    total_flow: float,
    viscosity: float = 0.04,
    upstream_lengths=(10.0, 10.0),
    adaptive_lengths=(10.0, 10.0),
    terminal_resistances=(0.0, 0.0),
    distal_pressure: float = 0.0,
):
    """Compute branch hemodynamics with fixed upstream segments and adaptive downstream BCs."""
    adaptive_radii_arr = _as_2vector(adaptive_radii, label="adaptive_radii")
    upstream_radii_arr = _as_2vector(upstream_radii, label="upstream_radii")
    upstream_lengths_arr = _as_2vector(upstream_lengths, label="upstream_lengths")
    adaptive_lengths_arr = _as_2vector(adaptive_lengths, label="adaptive_lengths")
    terminal_resistance_arr = _as_nonnegative_2vector(terminal_resistances, label="terminal_resistances")
    total_flow = float(total_flow)
    viscosity = float(viscosity)
    distal_pressure = float(distal_pressure)
    if viscosity <= 0.0:
        raise ValueError("viscosity must be > 0.")

    upstream_radii_safe = np.maximum(upstream_radii_arr, 1.0e-50)
    adaptive_radii_safe = np.maximum(adaptive_radii_arr, 1.0e-50)
    upstream_resistance = 8.0 * viscosity * upstream_lengths_arr / (np.pi * upstream_radii_safe ** 4)
    adaptive_resistance = 8.0 * viscosity * adaptive_lengths_arr / (np.pi * adaptive_radii_safe ** 4)
    total_resistance = upstream_resistance + adaptive_resistance + terminal_resistance_arr
    conductance = 1.0 / total_resistance
    total_conductance = float(np.sum(conductance))
    if total_conductance <= 0.0:
        raise ValueError("total conductance must be > 0.")

    flow = total_flow * conductance / total_conductance
    inlet_pressure = distal_pressure + total_flow / total_conductance
    adaptive_inlet_pressure = distal_pressure + flow * (adaptive_resistance + terminal_resistance_arr)
    adaptive_shear = 4.0 * viscosity * flow / (np.pi * np.maximum(adaptive_radii_safe ** 3, _EPS))
    upstream_shear = 4.0 * viscosity * flow / (np.pi * np.maximum(upstream_radii_safe ** 3, _EPS))

    return {
        "adaptive_radii": adaptive_radii_arr,
        "upstream_radii": upstream_radii_arr,
        "upstream_lengths": upstream_lengths_arr,
        "adaptive_lengths": adaptive_lengths_arr,
        "upstream_resistance": upstream_resistance,
        "adaptive_resistance": adaptive_resistance,
        "terminal_resistance": terminal_resistance_arr,
        "resistance": total_resistance,
        "conductance": conductance,
        "flow": flow,
        "split": flow / max(total_flow, _EPS),
        "wall_shear_stress": adaptive_shear,
        "upstream_wall_shear_stress": upstream_shear,
        "inlet_pressure": float(inlet_pressure),
        "adaptive_inlet_pressure": adaptive_inlet_pressure,
        "distal_pressure": distal_pressure,
    }


def simulate_two_branch_no_load_cwss_toy(
    *,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    total_flow: float = 1.0,
    k_tau_r: float = 1.0e-3,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate a two-branch no-load CWSS-only toy system in log-radius space."""
    initial_radii_arr = _as_2vector(initial_radii, label="initial_radii")
    perturbed_radii_arr = _as_2vector(perturbed_radii, label="perturbed_radii")
    if float(k_tau_r) < 0.0:
        raise ValueError("k_tau_r must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_no_load_hemodynamics(
        initial_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        lengths=lengths,
        distal_pressure=distal_pressure,
    )
    tau_homeostatic = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    y0 = np.log(perturbed_radii_arr)

    def rhs(_t, y):
        radii = np.maximum(np.exp(np.asarray(y, dtype=np.float64)), _EPS)
        hemo = parallel_branch_no_load_hemodynamics(
            radii,
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        return float(k_tau_r) * np.log(tau / tau_homeostatic)

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    radii_history = np.exp(sol.y)
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)

    for idx in range(sol.t.size):
        hemo = parallel_branch_no_load_hemodynamics(
            radii_history[:, idx],
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]

    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "log",
        "k_tau_r": float(k_tau_r),
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "lengths": [float(value) for value in _as_2vector(lengths, label="lengths")],
        "starting_radii": [float(value) for value in initial_radii_arr],
        "post_perturbation_radii": [float(value) for value in perturbed_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "reference_tau": [float(value) for value in tau_homeostatic],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(split_history[0, 0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_radii_arr[0] / initial_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(inlet_pressure_history[0]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - 0.5) > abs(float(split_history[0, 0]) - 0.5)
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
    }


def simulate_two_branch_no_load_cwss_toy_nonlog(
    *,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    total_flow: float = 1.0,
    k_tau_r: float = 1.0e-3,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate a two-branch no-load CWSS-only toy system directly in radius."""
    initial_radii_arr = _as_2vector(initial_radii, label="initial_radii")
    perturbed_radii_arr = _as_2vector(perturbed_radii, label="perturbed_radii")
    if float(k_tau_r) < 0.0:
        raise ValueError("k_tau_r must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_no_load_hemodynamics(
        initial_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        lengths=lengths,
        distal_pressure=distal_pressure,
    )
    tau_reference = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    y0 = perturbed_radii_arr.copy()

    def rhs(_t, y):
        radii = np.maximum(np.asarray(y, dtype=np.float64), _EPS)
        hemo = parallel_branch_no_load_hemodynamics(
            radii,
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        return float(k_tau_r) * np.log(tau / tau_reference)

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    radii_history = np.maximum(sol.y, _EPS)
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)

    for idx in range(sol.t.size):
        hemo = parallel_branch_no_load_hemodynamics(
            radii_history[:, idx],
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]

    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "nonlog",
        "k_tau_r": float(k_tau_r),
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "lengths": [float(value) for value in _as_2vector(lengths, label="lengths")],
        "starting_radii": [float(value) for value in initial_radii_arr],
        "post_perturbation_radii": [float(value) for value in perturbed_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "reference_tau": [float(value) for value in tau_reference],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(split_history[0, 0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_radii_arr[0] / initial_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(inlet_pressure_history[0]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - 0.5) > abs(float(split_history[0, 0]) - 0.5)
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
    }


def simulate_two_branch_terminal_resistance_cwss_toy(
    *,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    total_flow: float = 1.0,
    k_tau_r: float = 1.0e-3,
    terminal_resistances=(0.0, 0.0),
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate a two-branch CWSS-only toy system with series terminal loads."""
    initial_radii_arr = _as_2vector(initial_radii, label="initial_radii")
    perturbed_radii_arr = _as_2vector(perturbed_radii, label="perturbed_radii")
    terminal_resistance_arr = _as_nonnegative_2vector(terminal_resistances, label="terminal_resistances")
    if float(k_tau_r) < 0.0:
        raise ValueError("k_tau_r must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_terminal_resistance_hemodynamics(
        initial_radii_arr,
        total_flow=total_flow,
        terminal_resistances=terminal_resistance_arr,
        viscosity=viscosity,
        lengths=lengths,
        distal_pressure=distal_pressure,
    )
    tau_homeostatic = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    y0 = np.log(perturbed_radii_arr)

    def rhs(_t, y):
        radii = np.maximum(np.exp(np.asarray(y, dtype=np.float64)), _EPS)
        hemo = parallel_branch_terminal_resistance_hemodynamics(
            radii,
            total_flow=total_flow,
            terminal_resistances=terminal_resistance_arr,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        return float(k_tau_r) * np.log(tau / tau_homeostatic)

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    radii_history = np.exp(sol.y)
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)

    for idx in range(sol.t.size):
        hemo = parallel_branch_terminal_resistance_hemodynamics(
            radii_history[:, idx],
            total_flow=total_flow,
            terminal_resistances=terminal_resistance_arr,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]

    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "log",
        "k_tau_r": float(k_tau_r),
        "terminal_resistances": [float(value) for value in terminal_resistance_arr],
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "lengths": [float(value) for value in _as_2vector(lengths, label="lengths")],
        "starting_radii": [float(value) for value in initial_radii_arr],
        "post_perturbation_radii": [float(value) for value in perturbed_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "reference_tau": [float(value) for value in tau_homeostatic],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(split_history[0, 0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_radii_arr[0] / initial_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(inlet_pressure_history[0]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - 0.5) > abs(float(split_history[0, 0]) - 0.5)
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
    }


def simulate_two_branch_fixed_upstream_adaptive_bc_cwss_toy(
    *,
    homeostatic_upstream_radii=(0.1, 0.1),
    perturbed_upstream_radii=(0.105, 0.1),
    initial_adaptive_radii=(0.1, 0.1),
    total_flow: float = 1.0,
    k_tau_r: float = 1.0e-3,
    viscosity: float = 0.04,
    upstream_lengths=(10.0, 10.0),
    adaptive_lengths=(10.0, 10.0),
    terminal_resistances=(0.0, 0.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate CWSS-only adaptation of downstream BC radii after a fixed upstream perturbation."""
    homeostatic_upstream_radii_arr = _as_2vector(homeostatic_upstream_radii, label="homeostatic_upstream_radii")
    perturbed_upstream_radii_arr = _as_2vector(perturbed_upstream_radii, label="perturbed_upstream_radii")
    initial_adaptive_radii_arr = _as_2vector(initial_adaptive_radii, label="initial_adaptive_radii")
    terminal_resistance_arr = _as_nonnegative_2vector(terminal_resistances, label="terminal_resistances")
    if float(k_tau_r) < 0.0:
        raise ValueError("k_tau_r must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        initial_adaptive_radii_arr,
        upstream_radii=homeostatic_upstream_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        upstream_lengths=upstream_lengths,
        adaptive_lengths=adaptive_lengths,
        terminal_resistances=terminal_resistance_arr,
        distal_pressure=distal_pressure,
    )
    tau_homeostatic = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    y0 = np.log(initial_adaptive_radii_arr)

    def rhs(_t, y):
        adaptive_radii = np.maximum(np.exp(np.asarray(y, dtype=np.float64)), _EPS)
        hemo = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
            adaptive_radii,
            upstream_radii=perturbed_upstream_radii_arr,
            total_flow=total_flow,
            viscosity=viscosity,
            upstream_lengths=upstream_lengths,
            adaptive_lengths=adaptive_lengths,
            terminal_resistances=terminal_resistance_arr,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        return float(k_tau_r) * np.log(tau / tau_homeostatic)

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    radii_history = np.exp(sol.y)
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)
    adaptive_inlet_pressure_history = np.empty_like(radii_history)

    for idx in range(sol.t.size):
        hemo = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
            radii_history[:, idx],
            upstream_radii=perturbed_upstream_radii_arr,
            total_flow=total_flow,
            viscosity=viscosity,
            upstream_lengths=upstream_lengths,
            adaptive_lengths=adaptive_lengths,
            terminal_resistances=terminal_resistance_arr,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]
        adaptive_inlet_pressure_history[:, idx] = hemo["adaptive_inlet_pressure"]

    perturbed_initial = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        initial_adaptive_radii_arr,
        upstream_radii=perturbed_upstream_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        upstream_lengths=upstream_lengths,
        adaptive_lengths=adaptive_lengths,
        terminal_resistances=terminal_resistance_arr,
        distal_pressure=distal_pressure,
    )
    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "log",
        "model_topology": "fixed_upstream_adaptive_downstream_bc",
        "k_tau_r": float(k_tau_r),
        "terminal_resistances": [float(value) for value in terminal_resistance_arr],
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "upstream_lengths": [float(value) for value in _as_2vector(upstream_lengths, label="upstream_lengths")],
        "adaptive_lengths": [float(value) for value in _as_2vector(adaptive_lengths, label="adaptive_lengths")],
        "homeostatic_upstream_radii": [float(value) for value in homeostatic_upstream_radii_arr],
        "perturbed_upstream_radii": [float(value) for value in perturbed_upstream_radii_arr],
        "starting_radii": [float(value) for value in initial_adaptive_radii_arr],
        "post_perturbation_radii": [float(value) for value in initial_adaptive_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "reference_tau": [float(value) for value in tau_homeostatic],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(perturbed_initial["split"][0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_adaptive_radii_arr[0] / initial_adaptive_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(perturbed_initial["inlet_pressure"]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - float(homeostatic["split"][0]))
            > abs(float(perturbed_initial["split"][0]) - float(homeostatic["split"][0]))
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
        "adaptive_inlet_pressure_history": adaptive_inlet_pressure_history,
    }


def simulate_two_branch_fixed_upstream_adaptive_bc_cwss_ims_toy(
    *,
    homeostatic_upstream_radii=(0.1, 0.1),
    perturbed_upstream_radii=(0.105, 0.1),
    initial_adaptive_radii=(0.1, 0.1),
    initial_thickness=(0.01, 0.01),
    total_flow: float = 1.0,
    k_arr=(1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2),
    viscosity: float = 0.04,
    upstream_lengths=(10.0, 10.0),
    adaptive_lengths=(10.0, 10.0),
    terminal_resistances=(0.0, 0.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate CWSS-IMS adaptation of downstream BC radii/thickness after a fixed upstream perturbation."""
    homeostatic_upstream_radii_arr = _as_2vector(homeostatic_upstream_radii, label="homeostatic_upstream_radii")
    perturbed_upstream_radii_arr = _as_2vector(perturbed_upstream_radii, label="perturbed_upstream_radii")
    initial_adaptive_radii_arr = _as_2vector(initial_adaptive_radii, label="initial_adaptive_radii")
    initial_thickness_arr = _as_2vector(initial_thickness, label="initial_thickness")
    terminal_resistance_arr = _as_nonnegative_2vector(terminal_resistances, label="terminal_resistances")
    k_arr_vec = _as_4vector(k_arr, label="k_arr")
    if np.any(k_arr_vec < 0.0):
        raise ValueError("k_arr entries must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        initial_adaptive_radii_arr,
        upstream_radii=homeostatic_upstream_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        upstream_lengths=upstream_lengths,
        adaptive_lengths=adaptive_lengths,
        terminal_resistances=terminal_resistance_arr,
        distal_pressure=distal_pressure,
    )
    tau_homeostatic = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    sigma_homeostatic = np.maximum(
        homeostatic["adaptive_inlet_pressure"] * initial_adaptive_radii_arr / np.maximum(initial_thickness_arr, _EPS),
        _EPS,
    )
    y0 = np.log(np.concatenate([initial_adaptive_radii_arr, initial_thickness_arr]))

    def rhs(_t, y):
        state = np.maximum(np.exp(np.asarray(y, dtype=np.float64)), _EPS)
        adaptive_radii = state[:2]
        thickness = state[2:]
        hemo = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
            adaptive_radii,
            upstream_radii=perturbed_upstream_radii_arr,
            total_flow=total_flow,
            viscosity=viscosity,
            upstream_lengths=upstream_lengths,
            adaptive_lengths=adaptive_lengths,
            terminal_resistances=terminal_resistance_arr,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        sigma = np.maximum(hemo["adaptive_inlet_pressure"] * adaptive_radii / np.maximum(thickness, _EPS), _EPS)
        tau_err = np.log(tau / tau_homeostatic)
        sigma_err = np.log(sigma / sigma_homeostatic)

        dydt = np.empty_like(state)
        dydt[:2] = k_arr_vec[0] * tau_err + k_arr_vec[1] * sigma_err
        dydt[2:] = k_arr_vec[2] * tau_err + k_arr_vec[3] * sigma_err
        return dydt

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    state_history = np.exp(sol.y)
    radii_history = state_history[:2, :]
    thickness_history = state_history[2:, :]
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    sigma_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)
    adaptive_inlet_pressure_history = np.empty_like(radii_history)

    for idx in range(sol.t.size):
        hemo = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
            radii_history[:, idx],
            upstream_radii=perturbed_upstream_radii_arr,
            total_flow=total_flow,
            viscosity=viscosity,
            upstream_lengths=upstream_lengths,
            adaptive_lengths=adaptive_lengths,
            terminal_resistances=terminal_resistance_arr,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]
        adaptive_inlet_pressure_history[:, idx] = hemo["adaptive_inlet_pressure"]
        sigma_history[:, idx] = (
            hemo["adaptive_inlet_pressure"] * radii_history[:, idx] / np.maximum(thickness_history[:, idx], _EPS)
        )

    perturbed_initial = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        initial_adaptive_radii_arr,
        upstream_radii=perturbed_upstream_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        upstream_lengths=upstream_lengths,
        adaptive_lengths=adaptive_lengths,
        terminal_resistances=terminal_resistance_arr,
        distal_pressure=distal_pressure,
    )
    perturbed_sigma = np.maximum(
        perturbed_initial["adaptive_inlet_pressure"] * initial_adaptive_radii_arr / np.maximum(initial_thickness_arr, _EPS),
        _EPS,
    )
    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "log",
        "model_topology": "fixed_upstream_adaptive_downstream_bc",
        "k_arr": [float(value) for value in k_arr_vec],
        "terminal_resistances": [float(value) for value in terminal_resistance_arr],
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "upstream_lengths": [float(value) for value in _as_2vector(upstream_lengths, label="upstream_lengths")],
        "adaptive_lengths": [float(value) for value in _as_2vector(adaptive_lengths, label="adaptive_lengths")],
        "homeostatic_upstream_radii": [float(value) for value in homeostatic_upstream_radii_arr],
        "perturbed_upstream_radii": [float(value) for value in perturbed_upstream_radii_arr],
        "starting_radii": [float(value) for value in initial_adaptive_radii_arr],
        "post_perturbation_radii": [float(value) for value in initial_adaptive_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "starting_thickness": [float(value) for value in initial_thickness_arr],
        "post_perturbation_thickness": [float(value) for value in initial_thickness_arr],
        "final_thickness": [float(value) for value in thickness_history[:, -1]],
        "reference_tau": [float(value) for value in tau_homeostatic],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "reference_sigma": [float(value) for value in sigma_homeostatic],
        "post_perturbation_sigma": [float(value) for value in perturbed_sigma],
        "final_sigma": [float(value) for value in sigma_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(perturbed_initial["split"][0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_adaptive_radii_arr[0] / initial_adaptive_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_thickness_ratio_left_to_right": float(initial_thickness_arr[0] / initial_thickness_arr[1]),
        "post_perturbation_thickness_ratio_left_to_right": float(thickness_history[0, 0] / thickness_history[1, 0]),
        "final_thickness_ratio_left_to_right": float(thickness_history[0, -1] / thickness_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(perturbed_initial["inlet_pressure"]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - float(homeostatic["split"][0]))
            > abs(float(perturbed_initial["split"][0]) - float(homeostatic["split"][0]))
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "thickness_history": thickness_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "sigma_history": sigma_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
        "adaptive_inlet_pressure_history": adaptive_inlet_pressure_history,
    }


def simulate_two_branch_no_load_cwss_ims_toy(
    *,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    initial_thickness=(0.01, 0.01),
    perturbed_thickness=(0.01, 0.01),
    total_flow: float = 1.0,
    k_arr=(1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2),
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate the full four-gain CWSS-IMS toy system in log-state space."""
    initial_radii_arr = _as_2vector(initial_radii, label="initial_radii")
    perturbed_radii_arr = _as_2vector(perturbed_radii, label="perturbed_radii")
    initial_thickness_arr = _as_2vector(initial_thickness, label="initial_thickness")
    perturbed_thickness_arr = _as_2vector(perturbed_thickness, label="perturbed_thickness")
    k_arr_vec = _as_4vector(k_arr, label="k_arr")
    if np.any(k_arr_vec < 0.0):
        raise ValueError("k_arr entries must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_no_load_hemodynamics(
        initial_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        lengths=lengths,
        distal_pressure=distal_pressure,
    )
    tau_homeostatic = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    pressure_homeostatic = float(homeostatic["inlet_pressure"])
    sigma_homeostatic = np.maximum(
        pressure_homeostatic * initial_radii_arr / np.maximum(initial_thickness_arr, _EPS),
        _EPS,
    )
    y0 = np.log(np.concatenate([perturbed_radii_arr, perturbed_thickness_arr]))

    def rhs(_t, y):
        state = np.maximum(np.exp(np.asarray(y, dtype=np.float64)), _EPS)
        radii = state[:2]
        thickness = state[2:]
        hemo = parallel_branch_no_load_hemodynamics(
            radii,
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        pressure = np.full(2, float(hemo["inlet_pressure"]), dtype=np.float64)
        sigma = np.maximum(pressure * radii / np.maximum(thickness, _EPS), _EPS)
        tau_err = np.log(tau / tau_homeostatic)
        sigma_err = np.log(sigma / sigma_homeostatic)

        dydt = np.empty_like(state)
        dydt[:2] = k_arr_vec[0] * tau_err + k_arr_vec[1] * sigma_err
        # Match the MATLAB stress-adaptation convention: WSS and IMS both
        # enter the thickness equation with positive gain signs.
        dydt[2:] = k_arr_vec[2] * tau_err + k_arr_vec[3] * sigma_err
        return dydt

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    state_history = np.exp(sol.y)
    radii_history = state_history[:2, :]
    thickness_history = state_history[2:, :]
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    sigma_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)

    for idx in range(sol.t.size):
        hemo = parallel_branch_no_load_hemodynamics(
            radii_history[:, idx],
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]
        sigma_history[:, idx] = (
            float(hemo["inlet_pressure"])
            * radii_history[:, idx]
            / np.maximum(thickness_history[:, idx], _EPS)
        )

    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "log",
        "k_arr": [float(value) for value in k_arr_vec],
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "lengths": [float(value) for value in _as_2vector(lengths, label="lengths")],
        "starting_radii": [float(value) for value in initial_radii_arr],
        "post_perturbation_radii": [float(value) for value in perturbed_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "starting_thickness": [float(value) for value in initial_thickness_arr],
        "post_perturbation_thickness": [float(value) for value in perturbed_thickness_arr],
        "final_thickness": [float(value) for value in thickness_history[:, -1]],
        "reference_tau": [float(value) for value in tau_homeostatic],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "reference_sigma": [float(value) for value in sigma_homeostatic],
        "post_perturbation_sigma": [float(value) for value in sigma_history[:, 0]],
        "final_sigma": [float(value) for value in sigma_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(split_history[0, 0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_radii_arr[0] / initial_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_thickness_ratio_left_to_right": float(initial_thickness_arr[0] / initial_thickness_arr[1]),
        "post_perturbation_thickness_ratio_left_to_right": float(thickness_history[0, 0] / thickness_history[1, 0]),
        "final_thickness_ratio_left_to_right": float(thickness_history[0, -1] / thickness_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(inlet_pressure_history[0]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - 0.5) > abs(float(split_history[0, 0]) - 0.5)
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "thickness_history": thickness_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "sigma_history": sigma_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
    }


def simulate_two_branch_no_load_cwss_ims_toy_nonlog(
    *,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    initial_thickness=(0.01, 0.01),
    perturbed_thickness=(0.01, 0.01),
    total_flow: float = 1.0,
    k_arr=(1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2),
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Integrate the full four-gain CWSS-IMS toy directly in radius/thickness."""
    initial_radii_arr = _as_2vector(initial_radii, label="initial_radii")
    perturbed_radii_arr = _as_2vector(perturbed_radii, label="perturbed_radii")
    initial_thickness_arr = _as_2vector(initial_thickness, label="initial_thickness")
    perturbed_thickness_arr = _as_2vector(perturbed_thickness, label="perturbed_thickness")
    k_arr_vec = _as_4vector(k_arr, label="k_arr")
    if np.any(k_arr_vec < 0.0):
        raise ValueError("k_arr entries must be >= 0.")
    if float(t_end) <= 0.0:
        raise ValueError("t_end must be > 0.")

    homeostatic = parallel_branch_no_load_hemodynamics(
        initial_radii_arr,
        total_flow=total_flow,
        viscosity=viscosity,
        lengths=lengths,
        distal_pressure=distal_pressure,
    )
    tau_reference = np.maximum(homeostatic["wall_shear_stress"], _EPS)
    pressure_reference = float(homeostatic["inlet_pressure"])
    sigma_reference = np.maximum(
        pressure_reference * initial_radii_arr / np.maximum(initial_thickness_arr, _EPS),
        _EPS,
    )
    y0 = np.concatenate([perturbed_radii_arr, perturbed_thickness_arr])

    def rhs(_t, y):
        state = np.maximum(np.asarray(y, dtype=np.float64), _EPS)
        radii = state[:2]
        thickness = state[2:]
        hemo = parallel_branch_no_load_hemodynamics(
            radii,
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        tau = np.maximum(hemo["wall_shear_stress"], _EPS)
        pressure = np.full(2, float(hemo["inlet_pressure"]), dtype=np.float64)
        sigma = np.maximum(pressure * radii / np.maximum(thickness, _EPS), _EPS)
        tau_err = np.log(tau / tau_reference)
        sigma_err = np.log(sigma / sigma_reference)
        dydt = np.empty_like(state)
        dydt[:2] = k_arr_vec[0] * tau_err + k_arr_vec[1] * sigma_err
        # Match the MATLAB stress-adaptation convention: WSS and IMS both
        # enter the thickness equation with positive gain signs.
        dydt[2:] = k_arr_vec[2] * tau_err + k_arr_vec[3] * sigma_err
        return dydt

    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    state_history = np.maximum(sol.y, _EPS)
    radii_history = state_history[:2, :]
    thickness_history = state_history[2:, :]
    flow_history = np.empty_like(radii_history)
    split_history = np.empty_like(radii_history)
    tau_history = np.empty_like(radii_history)
    sigma_history = np.empty_like(radii_history)
    resistance_history = np.empty_like(radii_history)
    inlet_pressure_history = np.empty(sol.t.size, dtype=np.float64)

    for idx in range(sol.t.size):
        hemo = parallel_branch_no_load_hemodynamics(
            radii_history[:, idx],
            total_flow=total_flow,
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
        )
        flow_history[:, idx] = hemo["flow"]
        split_history[:, idx] = hemo["split"]
        tau_history[:, idx] = hemo["wall_shear_stress"]
        resistance_history[:, idx] = hemo["resistance"]
        inlet_pressure_history[idx] = hemo["inlet_pressure"]
        sigma_history[:, idx] = (
            float(hemo["inlet_pressure"])
            * radii_history[:, idx]
            / np.maximum(thickness_history[:, idx], _EPS)
        )

    termination_reason = "t_end_reached" if sol.status == 0 else str(sol.message).strip() or "solver_failed"
    summary = {
        "ode_scale": "nonlog",
        "k_arr": [float(value) for value in k_arr_vec],
        "total_flow": float(total_flow),
        "viscosity": float(viscosity),
        "distal_pressure": float(distal_pressure),
        "lengths": [float(value) for value in _as_2vector(lengths, label="lengths")],
        "starting_radii": [float(value) for value in initial_radii_arr],
        "post_perturbation_radii": [float(value) for value in perturbed_radii_arr],
        "final_radii": [float(value) for value in radii_history[:, -1]],
        "starting_thickness": [float(value) for value in initial_thickness_arr],
        "post_perturbation_thickness": [float(value) for value in perturbed_thickness_arr],
        "final_thickness": [float(value) for value in thickness_history[:, -1]],
        "reference_tau": [float(value) for value in tau_reference],
        "post_perturbation_tau": [float(value) for value in tau_history[:, 0]],
        "final_tau": [float(value) for value in tau_history[:, -1]],
        "reference_sigma": [float(value) for value in sigma_reference],
        "post_perturbation_sigma": [float(value) for value in sigma_history[:, 0]],
        "final_sigma": [float(value) for value in sigma_history[:, -1]],
        "starting_split_left": float(homeostatic["split"][0]),
        "post_perturbation_split_left": float(split_history[0, 0]),
        "final_split_left": float(split_history[0, -1]),
        "starting_radius_ratio_left_to_right": float(initial_radii_arr[0] / initial_radii_arr[1]),
        "post_perturbation_radius_ratio_left_to_right": float(radii_history[0, 0] / radii_history[1, 0]),
        "final_radius_ratio_left_to_right": float(radii_history[0, -1] / radii_history[1, -1]),
        "starting_thickness_ratio_left_to_right": float(initial_thickness_arr[0] / initial_thickness_arr[1]),
        "post_perturbation_thickness_ratio_left_to_right": float(thickness_history[0, 0] / thickness_history[1, 0]),
        "final_thickness_ratio_left_to_right": float(thickness_history[0, -1] / thickness_history[1, -1]),
        "starting_inlet_pressure": float(homeostatic["inlet_pressure"]),
        "post_perturbation_inlet_pressure": float(inlet_pressure_history[0]),
        "final_inlet_pressure": float(inlet_pressure_history[-1]),
        "n_time_points": int(sol.t.size),
        "n_rhs": int(sol.nfev),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message).strip(),
        "termination_reason": termination_reason,
        "symmetry_break_amplified": bool(
            abs(float(split_history[0, -1]) - 0.5) > abs(float(split_history[0, 0]) - 0.5)
        ),
    }

    return {
        "summary": summary,
        "time": np.asarray(sol.t, dtype=np.float64),
        "radii_history": radii_history,
        "thickness_history": thickness_history,
        "flow_history": flow_history,
        "split_history": split_history,
        "tau_history": tau_history,
        "sigma_history": sigma_history,
        "resistance_history": resistance_history,
        "inlet_pressure_history": inlet_pressure_history,
    }


def write_two_branch_no_load_toy_case(case_result, *, output_dir: str | Path):
    """Persist a single two-branch toy run as JSON, CSV, and PNG artifacts."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "toy_summary.json"
    history_csv_path = output_path / "toy_history.csv"
    histories_png_path = output_path / "toy_histories.png"

    summary_path.write_text(json.dumps(case_result["summary"], indent=2, sort_keys=True), encoding="utf-8")

    with history_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "time_s",
                "left_radius",
                "right_radius",
                "left_flow",
                "right_flow",
                "left_split",
                "right_split",
                "left_tau",
                "right_tau",
                "left_sigma",
                "right_sigma",
                "left_thickness",
                "right_thickness",
                "left_resistance",
                "right_resistance",
                "inlet_pressure",
            ],
        )
        writer.writeheader()
        for idx, time_value in enumerate(case_result["time"]):
            writer.writerow(
                {
                    "time_s": float(time_value),
                    "left_radius": float(case_result["radii_history"][0, idx]),
                    "right_radius": float(case_result["radii_history"][1, idx]),
                    "left_flow": float(case_result["flow_history"][0, idx]),
                    "right_flow": float(case_result["flow_history"][1, idx]),
                    "left_split": float(case_result["split_history"][0, idx]),
                    "right_split": float(case_result["split_history"][1, idx]),
                    "left_tau": float(case_result["tau_history"][0, idx]),
                    "right_tau": float(case_result["tau_history"][1, idx]),
                    "left_sigma": float(case_result.get("sigma_history", np.full_like(case_result["tau_history"], np.nan))[0, idx]),
                    "right_sigma": float(case_result.get("sigma_history", np.full_like(case_result["tau_history"], np.nan))[1, idx]),
                    "left_thickness": float(case_result.get("thickness_history", np.full_like(case_result["tau_history"], np.nan))[0, idx]),
                    "right_thickness": float(case_result.get("thickness_history", np.full_like(case_result["tau_history"], np.nan))[1, idx]),
                    "left_resistance": float(case_result["resistance_history"][0, idx]),
                    "right_resistance": float(case_result["resistance_history"][1, idx]),
                    "inlet_pressure": float(case_result["inlet_pressure_history"][idx]),
                }
            )

    has_thickness = "thickness_history" in case_result
    has_sigma = "sigma_history" in case_result

    if has_thickness or has_sigma:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    time_values = case_result["time"]

    axes[0, 0].plot(time_values, case_result["radii_history"][0], label="Left")
    axes[0, 0].plot(time_values, case_result["radii_history"][1], label="Right")
    axes[0, 0].set_title("Radius")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(time_values, case_result["flow_history"][0], label="Left")
    axes[0, 1].plot(time_values, case_result["flow_history"][1], label="Right")
    axes[0, 1].set_title("Flow")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(time_values, case_result["tau_history"][0], label="Left")
    axes[1, 0].plot(time_values, case_result["tau_history"][1], label="Right")
    axes[1, 0].set_title("Wall Shear Stress")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(alpha=0.25)

    if has_thickness or has_sigma:
        axes[1, 1].plot(time_values, case_result["thickness_history"][0], label="Left")
        axes[1, 1].plot(time_values, case_result["thickness_history"][1], label="Right")
        axes[1, 1].set_title("Thickness")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(alpha=0.25)

        axes[2, 0].plot(time_values, case_result["sigma_history"][0], label="Left")
        axes[2, 0].plot(time_values, case_result["sigma_history"][1], label="Right")
        axes[2, 0].set_title("Intramural Stress")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].grid(alpha=0.25)

        axes[2, 1].plot(time_values, case_result["split_history"][0], label="Left split")
        axes[2, 1].plot(time_values, case_result["split_history"][1], label="Right split")
        axes[2, 1].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        axes[2, 1].set_title("Flow Split")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].grid(alpha=0.25)
    else:
        axes[1, 1].plot(time_values, case_result["split_history"][0], label="Left split")
        axes[1, 1].plot(time_values, case_result["split_history"][1], label="Right split")
        axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        axes[1, 1].set_title("Flow Split")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(histories_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_json": str(summary_path),
        "history_csv": str(history_csv_path),
        "histories_png": str(histories_png_path),
    }


def write_two_branch_no_load_cwss_toy_case(case_result, *, output_dir: str | Path):
    return write_two_branch_no_load_toy_case(case_result, output_dir=output_dir)


def run_two_branch_no_load_cwss_toy_benchmark(
    *,
    output_dir: str | Path,
    k_values=(1.0e-4, 1.0e-3, 1.0e-2),
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    total_flow: float = 1.0,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Run a small gain sweep for the two-branch no-load CWSS toy system."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    overlay_fig, overlay_ax = plt.subplots(figsize=(7, 4))

    for k_tau_r in k_values:
        case_name = f"k_tau_r_{float(k_tau_r):.0e}".replace("+", "")
        case_output_dir = output_path / case_name
        case_result = simulate_two_branch_no_load_cwss_toy(
            initial_radii=initial_radii,
            perturbed_radii=perturbed_radii,
            total_flow=total_flow,
            k_tau_r=float(k_tau_r),
            viscosity=viscosity,
            lengths=lengths,
            distal_pressure=distal_pressure,
            t_end=t_end,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            method=method,
        )
        artifacts = write_two_branch_no_load_toy_case(case_result, output_dir=case_output_dir)
        row = dict(case_result["summary"])
        row["case"] = case_name
        row["artifacts"] = artifacts
        rows.append(row)

        overlay_ax.plot(case_result["time"], case_result["split_history"][0], label=case_name)

    overlay_ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    overlay_ax.set_title("Left-branch flow split")
    overlay_ax.set_xlabel("Time (s)")
    overlay_ax.set_ylabel("Split")
    overlay_ax.grid(alpha=0.25)
    overlay_ax.legend()
    overlay_fig.tight_layout()
    overlay_png = output_path / "left_split_overlay.png"
    overlay_fig.savefig(overlay_png, dpi=200, bbox_inches="tight")
    plt.close(overlay_fig)

    summary_csv = output_path / "benchmark_summary.csv"
    summary_json = output_path / "benchmark_summary.json"

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "case": row["case"],
                "k_tau_r": row["k_tau_r"],
                "starting_split_left": row["starting_split_left"],
                "post_perturbation_split_left": row["post_perturbation_split_left"],
                "final_split_left": row["final_split_left"],
                "starting_radius_ratio_left_to_right": row["starting_radius_ratio_left_to_right"],
                "post_perturbation_radius_ratio_left_to_right": row["post_perturbation_radius_ratio_left_to_right"],
                "final_radius_ratio_left_to_right": row["final_radius_ratio_left_to_right"],
                "symmetry_break_amplified": int(bool(row["symmetry_break_amplified"])),
                "termination_reason": row["termination_reason"],
                "n_rhs": row["n_rhs"],
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "output_dir": str(output_path),
        "rows": rows,
        "artifacts": {
            "benchmark_summary_csv": str(summary_csv),
            "benchmark_summary_json": str(summary_json),
            "left_split_overlay_png": str(overlay_png),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_two_branch_no_load_cwss_ims_comparison(
    *,
    output_dir: str | Path,
    initial_radii=(0.1, 0.1),
    perturbed_radii=(0.105, 0.095),
    initial_thickness=(0.01, 0.01),
    perturbed_thickness=(0.01, 0.01),
    total_flow: float = 1.0,
    viscosity: float = 0.04,
    lengths=(10.0, 10.0),
    distal_pressure: float = 0.0,
    t_end: float = 600.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0,
    method: str = "RK45",
):
    """Compare CWSS-only and full CWSS-IMS four-gain dynamics on the same toy system."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    case_specs = [
        {
            "case": "cwss_only_k_tau_r_1e-2",
            "runner": simulate_two_branch_no_load_cwss_toy,
            "kwargs": {
                "initial_radii": initial_radii,
                "perturbed_radii": perturbed_radii,
                "total_flow": total_flow,
                "k_tau_r": 1.0e-2,
                "viscosity": viscosity,
                "lengths": lengths,
                "distal_pressure": distal_pressure,
                "t_end": t_end,
                "rtol": rtol,
                "atol": atol,
                "max_step": max_step,
                "method": method,
            },
        },
        {
            "case": "cwss_ims_all_gains_1e-2",
            "runner": simulate_two_branch_no_load_cwss_ims_toy,
            "kwargs": {
                "initial_radii": initial_radii,
                "initial_thickness": initial_thickness,
                "perturbed_radii": perturbed_radii,
                "perturbed_thickness": perturbed_thickness,
                "total_flow": total_flow,
                "k_arr": (1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2),
                "viscosity": viscosity,
                "lengths": lengths,
                "distal_pressure": distal_pressure,
                "t_end": t_end,
                "rtol": rtol,
                "atol": atol,
                "max_step": max_step,
                "method": method,
            },
        },
    ]

    split_fig, split_ax = plt.subplots(figsize=(7, 4))
    radius_fig, radius_ax = plt.subplots(figsize=(7, 4))
    rows = []

    for spec in case_specs:
        case_result = spec["runner"](**spec["kwargs"])
        case_output_dir = output_path / spec["case"]
        artifacts = write_two_branch_no_load_toy_case(case_result, output_dir=case_output_dir)
        row = dict(case_result["summary"])
        row["case"] = spec["case"]
        row["artifacts"] = artifacts
        rows.append(row)

        split_ax.plot(case_result["time"], case_result["split_history"][0], label=spec["case"])
        radius_ax.plot(case_result["time"], case_result["radii_history"][0] / case_result["radii_history"][1], label=spec["case"])

    split_ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    split_ax.set_title("Left-branch flow split")
    split_ax.set_xlabel("Time (s)")
    split_ax.set_ylabel("Split")
    split_ax.grid(alpha=0.25)
    split_ax.legend()
    split_fig.tight_layout()
    split_overlay_png = output_path / "left_split_overlay.png"
    split_fig.savefig(split_overlay_png, dpi=200, bbox_inches="tight")
    plt.close(split_fig)

    radius_ax.set_title("Radius ratio (left/right)")
    radius_ax.set_xlabel("Time (s)")
    radius_ax.set_ylabel("Ratio")
    radius_ax.grid(alpha=0.25)
    radius_ax.legend()
    radius_fig.tight_layout()
    radius_overlay_png = output_path / "radius_ratio_overlay.png"
    radius_fig.savefig(radius_overlay_png, dpi=200, bbox_inches="tight")
    plt.close(radius_fig)

    summary_csv = output_path / "comparison_summary.csv"
    summary_json = output_path / "comparison_summary.json"
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "case": row["case"],
                "starting_split_left": row["starting_split_left"],
                "post_perturbation_split_left": row["post_perturbation_split_left"],
                "final_split_left": row["final_split_left"],
                "starting_radius_ratio_left_to_right": row["starting_radius_ratio_left_to_right"],
                "post_perturbation_radius_ratio_left_to_right": row["post_perturbation_radius_ratio_left_to_right"],
                "final_radius_ratio_left_to_right": row["final_radius_ratio_left_to_right"],
                "final_inlet_pressure": row["final_inlet_pressure"],
                "symmetry_break_amplified": int(bool(row["symmetry_break_amplified"])),
                "termination_reason": row["termination_reason"],
                "n_rhs": row["n_rhs"],
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "output_dir": str(output_path),
        "rows": rows,
        "artifacts": {
            "comparison_summary_csv": str(summary_csv),
            "comparison_summary_json": str(summary_json),
            "left_split_overlay_png": str(split_overlay_png),
            "radius_ratio_overlay_png": str(radius_overlay_png),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
