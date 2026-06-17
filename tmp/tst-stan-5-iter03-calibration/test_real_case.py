from __future__ import annotations

from types import SimpleNamespace

import pytest

from svzerodtrees.calibration.workflow import assemble_calibration_payload


def test_tst_stan_5_iter03_preop_stage1_real_case_hits_current_multisegment_limit():
    calibration = SimpleNamespace(
        data_source=SimpleNamespace(
            mode="mapped_centerline",
            mapped_centerline_result="tmp/tst-stan-5-iter03-calibration/postprocess/resistance_map_mean.vtp",
            centerline="tmp/tst-stan-5-iter03-calibration/case/centerlines.vtp",
            pressure_array="Pressure",
            flow_array="Velocity",
            branch_id_array="BranchId",
            path_array="Path",
        ),
        parameters=SimpleNamespace(
            vessels=SimpleNamespace(default=["R_poiseuille"], overrides={}),
            junctions=SimpleNamespace(default=["R_poiseuille"], overrides={}),
        ),
        solver=SimpleNamespace(
            initial_damping_factor=1.0,
            maximum_iterations=20,
            tolerance_gradient=1e-6,
            tolerance_increment=1e-10,
        ),
    )

    with pytest.raises(ValueError, match="one 0D vessel per centerline branch"):
        assemble_calibration_payload(
            zerod_config_path="tmp/tst-stan-5-iter03-calibration/case/baseline_0d.json",
            calibration=calibration,
        )
