from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from svzerodtrees.simulation.input_builders.svzerod_data import SvZeroDdata


def test_svzerod_data_resamples_outlet_result(tmp_path):
    data_path = tmp_path / "svZeroD_data"
    data_path.write_text(
        "time flow:vessel:OUT pressure:vessel:OUT\n"
        "0.0 1.0 10.0\n"
        "0.5 2.0 20.0\n"
        "1.0 3.0 30.0\n",
        encoding="utf-8",
    )
    data = SvZeroDdata(str(data_path))
    block = SimpleNamespace(location="outlet", name="OUT", connected_block="vessel")

    time, flow, pressure = data.get_result(
        block,
        cycle_duration=1.0,
        window="all",
        n_tsteps=4,
    )

    np.testing.assert_allclose(time, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(flow, [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(pressure, [10.0, 15.0, 20.0, 25.0])


def test_svzerod_data_rejects_missing_columns(tmp_path):
    data_path = tmp_path / "svZeroD_data"
    data_path.write_text("time unrelated\n0.0 1.0\n1.0 2.0\n", encoding="utf-8")
    data = SvZeroDdata(str(data_path))
    block = SimpleNamespace(location="inlet", name="IN", connected_block="vessel")

    with pytest.raises(KeyError, match="Missing expected columns"):
        data.get_result(block)
