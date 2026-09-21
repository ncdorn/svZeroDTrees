from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.post_processing.centerline_timeseries import (
    publish_centerline_timeseries,
    validate_centerline_timeseries_descriptor,
)


def _write_frame(
    path: Path,
    *,
    pressure: tuple[float, ...] = (10.0, 20.0),
    flow: tuple[float, ...] = (1.0, 2.0),
    offset: float = 0.0,
    flow_components: int = 1,
) -> None:
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0 + offset, 0.0, 0.0)
    points.InsertNextPoint(1.0 + offset, 0.0, 0.0)
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)

    for name, values in (
        ("BranchId", (7.0, 7.0)),
        ("Path", (0.0, 1.0)),
        ("pressure", pressure),
    ):
        array = numpy_to_vtk(np.asarray(values), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    flow_values = np.asarray(flow, dtype=float)
    if flow_components > 1:
        flow_values = np.column_stack((flow_values, flow_values))
    flow_array = numpy_to_vtk(flow_values, deep=True)
    flow_array.SetName("velocity")
    poly.GetPointData().AddArray(flow_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    assert writer.Write() == 1


def _publish_fixture(tmp_path: Path):
    reference = tmp_path / "reference.vtp"
    frame0 = tmp_path / "mapped_0.vtp"
    frame1 = tmp_path / "mapped_1.vtp"
    _write_frame(reference, pressure=(0.0, 0.0), flow=(0.0, 0.0))
    _write_frame(frame0, pressure=(10.0, 20.0), flow=(1.0, 2.0))
    _write_frame(frame1, pressure=(30.0, 40.0), flow=(3.0, 4.0))
    result = publish_centerline_timeseries(
        selected_frames=[
            {"path": str(frame0), "time_s": 0.2, "timestep_id": 20},
            {"path": str(frame1), "time_s": 0.4, "timestep_id": 40},
        ],
        output_dir=tmp_path / "postprocess",
        cycle_duration_s=0.8,
        reference_centerline=reference,
    )
    return result, reference, frame0, frame1


def test_publish_centerline_timeseries_writes_ordered_arrays_and_descriptor(
    tmp_path: Path,
):
    result, reference, _frame0, _frame1 = _publish_fixture(tmp_path)
    output_dir = tmp_path / "postprocess"
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(output_dir / "centerline_timeseries_last_cycle.vtp"))
    reader.Update()
    poly = reader.GetOutput()

    assert poly.GetNumberOfPoints() == 2
    assert poly.GetNumberOfCells() == 1
    assert poly.GetPointData().HasArray("pressure") == 0
    assert poly.GetPointData().HasArray("velocity") == 0
    assert poly.GetPointData().HasArray("pressure_0") == 1
    assert poly.GetPointData().HasArray("pressure_1") == 1
    assert poly.GetPointData().HasArray("flow_0") == 1
    assert poly.GetPointData().HasArray("flow_1") == 1
    assert [poly.GetPointData().GetArray("flow_0").GetTuple1(i) for i in range(2)] == [
        1.0,
        2.0,
    ]
    assert [poly.GetPointData().GetArray("flow_1").GetTuple1(i) for i in range(2)] == [
        3.0,
        4.0,
    ]

    descriptor_path = output_dir / "postprocess_suite_metadata.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    artifact = descriptor["artifacts"]["centerline_timeseries"]
    assert descriptor["schema_version"] == "1.0"
    assert artifact["vtp"] == "centerline_timeseries_last_cycle.vtp"
    assert artifact["metadata"] == "centerline_timeseries_last_cycle_metadata.json"
    assert artifact["reference_centerline"] == "../reference.vtp"
    assert artifact["frame_indices"] == [0, 1]
    assert artifact["timestamps_s"] == [0.2, 0.4]
    assert artifact["data_contract"]["pressure"]["units"] == "mmHg"
    assert artifact["data_contract"]["flow"] == {
        "quantity": "volumetric_flow",
        "units": "cm^3/s",
        "array_prefix": "flow",
        "source_array": "velocity",
        "source_semantics": "svSlicer velocity dot normal integrated over slice",
    }
    assert (
        artifact["vtp_sha256"]
        == hashlib.sha256((output_dir / artifact["vtp"]).read_bytes()).hexdigest()
    )
    assert (
        validate_centerline_timeseries_descriptor(descriptor_path)["artifact"][
            "frame_count"
        ]
        == 2
    )
    assert result["artifact"] == artifact
    assert reference.exists()


def test_publish_centerline_timeseries_records_tuned_config_lineage(tmp_path: Path):
    reference = tmp_path / "reference.vtp"
    frame = tmp_path / "mapped_0.vtp"
    tuned_config = tmp_path / "tuned" / "svzerod_3d_coupling_tuned.json"
    tuned_config.parent.mkdir()
    tuned_config.write_bytes(b'{"exact": "solver bytes"}\n')
    _write_frame(reference)
    _write_frame(frame)

    result = publish_centerline_timeseries(
        selected_frames=[{"path": str(frame), "time_s": 0.2}],
        output_dir=tmp_path / "postprocess",
        cycle_duration_s=0.8,
        reference_centerline=reference,
        tuned_zerod_config_path=tuned_config,
    )

    descriptor_path = tmp_path / "postprocess" / "postprocess_suite_metadata.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    lineage = descriptor["lineage"]
    assert lineage["tuned_zerod_config_path"] == "../tuned/svzerod_3d_coupling_tuned.json"
    assert lineage["tuned_zerod_config_sha256"] == hashlib.sha256(
        tuned_config.read_bytes()
    ).hexdigest()
    assert result["lineage"] == lineage
    assert validate_centerline_timeseries_descriptor(descriptor_path)["artifact"][
        "frame_count"
    ] == 1


def test_publish_centerline_timeseries_rejects_geometry_mismatch_without_descriptor(
    tmp_path: Path,
):
    reference = tmp_path / "reference.vtp"
    frame0 = tmp_path / "mapped_0.vtp"
    frame1 = tmp_path / "mapped_1.vtp"
    _write_frame(reference)
    _write_frame(frame0)
    _write_frame(frame1, offset=0.1)

    with pytest.raises(ValueError, match="point coordinates changed"):
        publish_centerline_timeseries(
            selected_frames=[
                {"path": str(frame0), "time_s": 0.2},
                {"path": str(frame1), "time_s": 0.4},
            ],
            output_dir=tmp_path / "postprocess",
            cycle_duration_s=0.8,
            reference_centerline=reference,
        )
    assert not (tmp_path / "postprocess" / "postprocess_suite_metadata.json").exists()
    assert not (
        tmp_path / "postprocess" / "centerline_timeseries_last_cycle.vtp"
    ).exists()


def test_publish_centerline_timeseries_rejects_order_and_array_contract(tmp_path: Path):
    reference = tmp_path / "reference.vtp"
    frame0 = tmp_path / "mapped_0.vtp"
    frame1 = tmp_path / "mapped_1.vtp"
    _write_frame(reference)
    _write_frame(frame0)
    _write_frame(frame1)

    with pytest.raises(ValueError, match="strictly increasing"):
        publish_centerline_timeseries(
            selected_frames=[
                {"path": str(frame0), "time_s": 0.4},
                {"path": str(frame1), "time_s": 0.2},
            ],
            output_dir=tmp_path / "bad-order",
            cycle_duration_s=0.8,
            reference_centerline=reference,
        )

    vector_frame = tmp_path / "vector.vtp"
    _write_frame(vector_frame, flow_components=2)
    with pytest.raises(ValueError, match="must be scalar"):
        publish_centerline_timeseries(
            selected_frames=[{"path": str(vector_frame), "time_s": 0.2}],
            output_dir=tmp_path / "bad-array",
            cycle_duration_s=0.8,
            reference_centerline=reference,
        )


def test_digest_failure_and_injected_publication_failure_leave_no_descriptor(
    monkeypatch, tmp_path: Path
):
    result, _reference, _frame0, _frame1 = _publish_fixture(tmp_path)
    descriptor_path = tmp_path / "postprocess" / "postprocess_suite_metadata.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["artifacts"]["centerline_timeseries"]["vtp_sha256"] = "bad"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_centerline_timeseries_descriptor(descriptor_path)

    # A validation failure after both staged files exist must not expose either
    # the descriptor or a partial final artifact.
    injected_root = tmp_path / "injected"
    injected_root.mkdir()
    monkeypatch.setattr(
        "svzerodtrees.post_processing.centerline_timeseries.validate_centerline_timeseries_descriptor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected publication failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected publication failure"):
        _publish_fixture(injected_root)[0]
    injected_output = injected_root / "postprocess"
    assert not (injected_output / "postprocess_suite_metadata.json").exists()
    assert not (injected_output / "centerline_timeseries_last_cycle.vtp").exists()
    assert not (
        injected_output / "centerline_timeseries_last_cycle_metadata.json"
    ).exists()


def test_failed_publication_preserves_existing_descriptor_and_unrelated_artifacts(
    tmp_path: Path,
):
    reference = tmp_path / "reference.vtp"
    frame0 = tmp_path / "mapped_0.vtp"
    frame1 = tmp_path / "mapped_1.vtp"
    _write_frame(reference)
    _write_frame(frame0)
    _write_frame(frame1, offset=0.1)

    output_dir = tmp_path / "postprocess"
    output_dir.mkdir()
    descriptor_path = output_dir / "postprocess_suite_metadata.json"
    original_descriptor = {
        "schema_version": "1.0",
        "artifacts": {
            "resistance_map": {
                "vtp": "resistance_map_mean.vtp",
                "sha256": "unrelated-record",
            }
        },
        "producer_note": "preserve this record",
    }
    original_bytes = (json.dumps(original_descriptor, indent=2) + "\n").encode()
    descriptor_path.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="point coordinates changed"):
        publish_centerline_timeseries(
            selected_frames=[
                {"path": str(frame0), "time_s": 0.2},
                {"path": str(frame1), "time_s": 0.4},
            ],
            output_dir=output_dir,
            cycle_duration_s=0.8,
            reference_centerline=reference,
            suite_metadata_path=descriptor_path,
        )

    assert descriptor_path.read_bytes() == original_bytes
    restored = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert restored["artifacts"] == original_descriptor["artifacts"]
    assert "centerline_timeseries" not in restored["artifacts"]
    assert not (output_dir / "centerline_timeseries_last_cycle.vtp").exists()
    assert not (
        output_dir / "centerline_timeseries_last_cycle_metadata.json"
    ).exists()
