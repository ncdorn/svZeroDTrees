import json

import numpy as np
import pytest

from svzerodtrees.io.blocks.boundary_condition import _resolve_flow_mean_config
from svzerodtrees.io.config_handler import ConfigHandler


def _payload():
    return {
        "simulation_parameters": {
            "density": 1.06,
            "viscosity": 0.04,
            "number_of_cardiac_cycles": 2,
            "number_of_time_pts_per_cardiac_cycle": 5,
        },
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW",
             "bc_values": {"Q": [1.0, 3.0, 2.0, 1.0], "t": [0.0, 0.2, 0.4, 0.8]}},
            {"bc_name": "OUT", "bc_type": "RESISTANCE", "bc_values": {"R": 100.0, "Pd": 0.0}},
        ],
        "vessels": [{
            "vessel_id": 0, "vessel_name": "branch0_seg0", "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {"R_poiseuille": 1.0, "C": 0.0, "L": 0.0, "stenosis_coefficient": 0.0},
            "boundary_conditions": {"inlet": "INFLOW", "outlet": "OUT"},
        }],
        "junctions": [],
    }


def test_scale_flow_bc_survives_serialization(tmp_path):
    handler = ConfigHandler(_payload(), is_pulmonary=False)

    handler.scale_flow_bc(2.0)
    out = tmp_path / "scaled.json"
    handler.to_json(str(out))

    written = json.loads(out.read_text(encoding="utf-8"))
    inflow = next(bc for bc in written["boundary_conditions"] if bc["bc_name"] == "INFLOW")
    assert inflow["bc_values"]["Q"] == pytest.approx([2.0, 6.0, 4.0, 2.0])
    assert _resolve_flow_mean_config(written) == pytest.approx(2.0 * np.mean([1.0, 3.0, 2.0, 1.0]))
    assert handler.bcs["INFLOW"].Q == pytest.approx([2.0, 6.0, 4.0, 2.0])


def test_scale_flow_bc_rejects_non_flow_bc():
    handler = ConfigHandler(_payload(), is_pulmonary=False)

    with pytest.raises(KeyError, match="FLOW boundary condition 'OUT'"):
        handler.scale_flow_bc(2.0, "OUT")
