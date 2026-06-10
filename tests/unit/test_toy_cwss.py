import numpy as np

from svzerodtrees.adaptation.toy import (
    parallel_branch_fixed_upstream_adaptive_bc_hemodynamics,
    parallel_branch_no_load_hemodynamics,
    parallel_branch_terminal_resistance_hemodynamics,
    simulate_two_branch_fixed_upstream_adaptive_bc_cwss_toy,
    simulate_two_branch_no_load_cwss_toy,
    simulate_two_branch_no_load_cwss_ims_toy,
    simulate_two_branch_no_load_cwss_ims_toy_nonlog,
    simulate_two_branch_no_load_cwss_toy_nonlog,
    simulate_two_branch_terminal_resistance_cwss_toy,
)


def test_parallel_branch_no_load_hemodynamics_balanced_case_has_equal_split():
    hemo = parallel_branch_no_load_hemodynamics(
        (0.1, 0.1),
        total_flow=1.0,
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
    )

    assert np.allclose(hemo["split"], [0.5, 0.5])
    assert np.allclose(hemo["wall_shear_stress"][0], hemo["wall_shear_stress"][1])


def test_parallel_branch_terminal_resistance_reduces_split_sensitivity():
    no_load = parallel_branch_no_load_hemodynamics(
        (0.105, 0.095),
        total_flow=1.0,
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
    )
    loaded = parallel_branch_terminal_resistance_hemodynamics(
        (0.105, 0.095),
        total_flow=1.0,
        terminal_resistances=(1.0e5, 1.0e5),
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
    )

    assert abs(loaded["split"][0] - 0.5) < abs(no_load["split"][0] - 0.5)
    assert loaded["inlet_pressure"] > no_load["inlet_pressure"]


def test_fixed_upstream_adaptive_bc_separates_perturbation_from_adaptive_state():
    homeostatic = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        (0.1, 0.1),
        upstream_radii=(0.1, 0.1),
        total_flow=1.0,
        viscosity=0.04,
        upstream_lengths=(10.0, 10.0),
        adaptive_lengths=(10.0, 10.0),
    )
    perturbed = parallel_branch_fixed_upstream_adaptive_bc_hemodynamics(
        (0.1, 0.1),
        upstream_radii=(0.105, 0.1),
        total_flow=1.0,
        viscosity=0.04,
        upstream_lengths=(10.0, 10.0),
        adaptive_lengths=(10.0, 10.0),
    )

    assert np.allclose(homeostatic["adaptive_radii"], perturbed["adaptive_radii"])
    assert perturbed["split"][0] > homeostatic["split"][0]
    assert not np.allclose(homeostatic["upstream_radii"], perturbed["upstream_radii"])


def test_two_branch_no_load_cwss_toy_amplifies_asymmetry():
    result = simulate_two_branch_no_load_cwss_toy(
        initial_radii=(0.1, 0.1),
        perturbed_radii=(0.105, 0.095),
        total_flow=1.0,
        k_tau_r=1.0e-3,
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
        t_end=200.0,
        max_step=1.0,
    )

    summary = result["summary"]
    assert summary["symmetry_break_amplified"] is True
    assert summary["final_split_left"] > summary["post_perturbation_split_left"]
    assert (
        summary["final_radius_ratio_left_to_right"]
        > summary["post_perturbation_radius_ratio_left_to_right"]
    )


def test_two_branch_terminal_resistance_cwss_toy_can_stabilize_split():
    result = simulate_two_branch_terminal_resistance_cwss_toy(
        initial_radii=(0.1, 0.1),
        perturbed_radii=(0.105, 0.095),
        total_flow=1.0,
        k_tau_r=1.0e-2,
        terminal_resistances=(1.0e5, 1.0e5),
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
        t_end=200.0,
        max_step=1.0,
    )

    summary = result["summary"]
    assert summary["terminal_resistances"] == [1.0e5, 1.0e5]
    assert summary["symmetry_break_amplified"] is False
    assert abs(summary["final_split_left"] - 0.5) < abs(summary["post_perturbation_split_left"] - 0.5)


def test_fixed_upstream_adaptive_bc_cwss_starts_from_symmetric_bc_radii():
    result = simulate_two_branch_fixed_upstream_adaptive_bc_cwss_toy(
        homeostatic_upstream_radii=(0.1, 0.1),
        perturbed_upstream_radii=(0.105, 0.1),
        initial_adaptive_radii=(0.1, 0.1),
        total_flow=1.0,
        k_tau_r=1.0e-2,
        viscosity=0.04,
        upstream_lengths=(10.0, 10.0),
        adaptive_lengths=(10.0, 10.0),
        t_end=50.0,
        max_step=1.0,
    )

    summary = result["summary"]
    assert summary["model_topology"] == "fixed_upstream_adaptive_downstream_bc"
    assert summary["homeostatic_upstream_radii"] == [0.1, 0.1]
    assert summary["perturbed_upstream_radii"] == [0.105, 0.1]
    assert summary["post_perturbation_radii"] == [0.1, 0.1]
    assert summary["post_perturbation_split_left"] > summary["starting_split_left"]


def test_two_branch_no_load_cwss_ims_toy_returns_positive_thickness_state():
    result = simulate_two_branch_no_load_cwss_ims_toy(
        initial_radii=(0.1, 0.1),
        perturbed_radii=(0.105, 0.095),
        initial_thickness=(0.01, 0.01),
        perturbed_thickness=(0.01, 0.01),
        total_flow=1.0,
        k_arr=(1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3),
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
        t_end=50.0,
        max_step=1.0,
    )

    assert np.all(result["thickness_history"] > 0.0)
    assert np.all(result["sigma_history"] > 0.0)
    assert result["summary"]["n_rhs"] > 0


def test_two_branch_no_load_cwss_ims_matlab_scaled_gains_avoid_split_collapse():
    common = {
        "initial_radii": (0.1, 0.1),
        "perturbed_radii": (0.1, 0.105),
        "initial_thickness": (0.01, 0.01),
        "perturbed_thickness": (0.01, 0.01),
        "total_flow": 1.0,
        "k_arr": (1.0e-2, 1.0e-2, 1.0e-1, 1.0e-1),
        "viscosity": 0.04,
        "lengths": (10.0, 10.0),
        "distal_pressure": 0.0,
        "t_end": 1200.0,
        "max_step": 1.0,
    }

    log_result = simulate_two_branch_no_load_cwss_ims_toy(**common)
    nonlog_result = simulate_two_branch_no_load_cwss_ims_toy_nonlog(**common)

    assert log_result["summary"]["final_split_left"] > 0.4
    assert nonlog_result["summary"]["final_split_left"] > 0.4
    assert log_result["summary"]["final_radius_ratio_left_to_right"] > 0.9
    assert nonlog_result["summary"]["final_radius_ratio_left_to_right"] > 0.9


def test_two_branch_no_load_cwss_toy_nonlog_amplifies_asymmetry():
    result = simulate_two_branch_no_load_cwss_toy_nonlog(
        initial_radii=(0.1, 0.1),
        perturbed_radii=(0.1, 0.105),
        total_flow=1.0,
        k_tau_r=1.0e-2,
        viscosity=0.04,
        lengths=(10.0, 10.0),
        distal_pressure=0.0,
        t_end=50.0,
        max_step=1.0,
    )

    summary = result["summary"]
    assert summary["ode_scale"] == "nonlog"
    assert summary["symmetry_break_amplified"] is True
    assert summary["final_split_left"] < summary["post_perturbation_split_left"]
