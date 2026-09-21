"""Construction and publication of calibration-ready centerline time series.

The pulmonary post-processing suite maps a selected set of result frames with
``svSlicer``.  This module turns those mapped centerlines into one stable VTP
artifact.  The mapped values are copied verbatim: ``velocity`` is the
svSlicer-integrated flow observation and is therefore not multiplied by an
area a second time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


CENTERLINE_TIMESERIES_SCHEMA_VERSION = "1.0"
CENTERLINE_TIMESERIES_KIND = "centerline_timeseries_last_cycle"
_PRESSURE_ALIASES = ("pressure", "Pressure")
_FLOW_ALIASES = ("velocity", "Velocity", "flow", "Flow")


def _read_polydata(path: str | Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    if output.GetPoints() is None or output.GetNumberOfPoints() <= 0:
        raise ValueError(f"centerline VTP has no points: {path}")
    return output


def _write_polydata(poly: vtk.vtkPolyData, path: str | Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write centerline time-series VTP: {path}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_publish_vtp(poly: vtk.vtkPolyData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temporary_path = Path(raw_path)
        _write_polydata(poly, temporary_path)
        if temporary_path.stat().st_size <= 0:
            raise RuntimeError(f"centerline time-series VTP is empty: {temporary_path}")
        # Re-read the temporary artifact before it becomes visible to a
        # consumer.  This also catches writer failures that return success but
        # produce an unreadable file.
        _read_polydata(temporary_path)
        os.replace(str(temporary_path), str(path))
        published_path = path
        temporary_path = None
        return published_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tuned_zerod_lineage(
    tuned_zerod_config_path: str | Path,
    *,
    descriptor_dir: Path,
) -> dict[str, str]:
    """Return the immutable identity of the exact tuned 0D input bytes.

    The path is kept relative to the descriptor so the artifact remains
    portable after staging.  The digest is computed from the file itself,
    rather than from a normalized JSON representation, because calibration
    must consume the same bytes that produced the 3D coupling configuration.
    """

    config_path = Path(tuned_zerod_config_path).expanduser().resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"tuned full 0D config is missing: {config_path}")
    return {
        "tuned_zerod_config_path": os.path.relpath(
            config_path, descriptor_dir
        ).replace(os.sep, "/"),
        "tuned_zerod_config_sha256": _sha256_file(config_path),
    }


def _validate_existing_lineage(
    existing_lineage: Mapping[str, Any],
    new_lineage: Mapping[str, str],
    *,
    descriptor_dir: Path,
) -> None:
    """Prevent a publication from changing an existing model identity."""

    existing_digest = existing_lineage.get("tuned_zerod_config_sha256")
    if existing_digest is not None and existing_digest != new_lineage[
        "tuned_zerod_config_sha256"
    ]:
        raise ValueError(
            "tuned full 0D config lineage does not match the existing descriptor"
        )
    existing_path = existing_lineage.get("tuned_zerod_config_path")
    if existing_path is None:
        return
    if not isinstance(existing_path, str) or Path(existing_path).is_absolute():
        raise ValueError(
            "existing tuned full 0D config lineage path must be descriptor-relative"
        )
    resolved_existing = (descriptor_dir / existing_path).resolve()
    resolved_new = (descriptor_dir / new_lineage["tuned_zerod_config_path"]).resolve()
    if resolved_existing != resolved_new:
        raise ValueError(
            "tuned full 0D config lineage path does not match the existing descriptor"
        )


def _as_records(
    selected_frames: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
) -> list[dict[str, Any]]:
    if hasattr(selected_frames, "to_dict"):
        records = selected_frames.to_dict(orient="records")
    else:
        records = list(selected_frames)
    if not records:
        raise ValueError("selected_frames must contain at least one frame")
    normalized: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"selected_frames[{index}] must be an object")
        record = dict(raw_record)
        raw_path = record.get("mapped_path", record.get("path"))
        if raw_path is None or not str(raw_path).strip():
            raise ValueError(
                f"selected_frames[{index}] is missing mapped centerline path"
            )
        mapped_path = Path(str(raw_path)).expanduser().resolve()
        if not mapped_path.exists() or not mapped_path.is_file():
            raise FileNotFoundError(f"mapped centerline is missing: {mapped_path}")
        record["path"] = str(mapped_path)
        raw_time = record.get("time_s")
        try:
            timestamp = float(raw_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"selected_frames[{index}] has invalid time_s: {raw_time!r}"
            ) from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"selected_frames[{index}] time_s must be finite")
        record["time_s"] = timestamp
        if "timestep_id" in record and record["timestep_id"] is not None:
            try:
                timestep_id = int(record["timestep_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"selected_frames[{index}] has invalid timestep_id: {record['timestep_id']!r}"
                ) from exc
            if float(record["timestep_id"]) != timestep_id:
                raise ValueError(
                    f"selected_frames[{index}] timestep_id must be integral"
                )
            record["timestep_id"] = timestep_id
        normalized.append(record)

    timestamps = [record["time_s"] for record in normalized]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("selected frame timestamps must be strictly increasing")
    timestep_ids = [record.get("timestep_id") for record in normalized]
    if all(value is not None for value in timestep_ids) and len(
        set(timestep_ids)
    ) != len(timestep_ids):
        raise ValueError("selected frame timestep_id values must be unique")
    if all(value is not None for value in timestep_ids) and any(
        left >= right for left, right in zip(timestep_ids, timestep_ids[1:])
    ):
        raise ValueError(
            "selected frame timestep_id values must be strictly increasing"
        )
    return normalized


def _load_selected_frames_metadata(
    path: Path,
) -> tuple[list[dict[str, Any]], float | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"resistance-map metadata was not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"resistance-map metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("resistance-map metadata must contain an object")
    selected_frames = payload.get("selected_frames")
    if not isinstance(selected_frames, list):
        raise ValueError("resistance-map metadata selected_frames must be a list")
    cycle_duration = payload.get("cycle_duration_s")
    if cycle_duration is None:
        resolved_cycle_duration = None
    else:
        try:
            resolved_cycle_duration = float(cycle_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "resistance-map metadata cycle_duration_s must be numeric"
            ) from exc
    return selected_frames, resolved_cycle_duration


def _cell_signature(poly: vtk.vtkPolyData) -> list[tuple[int, tuple[int, ...]]]:
    signatures: list[tuple[int, tuple[int, ...]]] = []
    for cell_index in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_index)
        signatures.append(
            (
                int(cell.GetCellType()),
                tuple(
                    int(cell.GetPointId(offset))
                    for offset in range(cell.GetNumberOfPoints())
                ),
            )
        )
    return signatures


def _validate_matching_geometry(
    reference: vtk.vtkPolyData,
    candidate: vtk.vtkPolyData,
    source_path: Path,
) -> None:
    if reference.GetNumberOfPoints() != candidate.GetNumberOfPoints():
        raise ValueError(f"mapped centerline point count changed: {source_path}")
    if reference.GetNumberOfCells() != candidate.GetNumberOfCells():
        raise ValueError(f"mapped centerline cell count changed: {source_path}")
    reference_points = vtk_to_numpy(reference.GetPoints().GetData())
    candidate_points = vtk_to_numpy(candidate.GetPoints().GetData())
    if not np.array_equal(reference_points, candidate_points):
        raise ValueError(f"mapped centerline point coordinates changed: {source_path}")
    if _cell_signature(reference) != _cell_signature(candidate):
        raise ValueError(f"mapped centerline connectivity changed: {source_path}")


def _find_point_array(
    poly: vtk.vtkPolyData,
    requested_name: str,
    aliases: Sequence[str],
    source_path: Path,
) -> tuple[vtk.vtkDataArray, str]:
    point_data = poly.GetPointData()
    candidates: list[str] = []
    for name in (requested_name, *aliases):
        if name and name.casefold() not in {value.casefold() for value in candidates}:
            candidates.append(name)
    array: vtk.vtkDataArray | None = None
    resolved_name: str | None = None
    for candidate in candidates:
        for array_index in range(point_data.GetNumberOfArrays()):
            available_name = point_data.GetArrayName(array_index)
            if available_name and available_name.casefold() == candidate.casefold():
                array = point_data.GetArray(array_index)
                resolved_name = str(available_name)
                break
        if array is not None:
            break
    if array is None or resolved_name is None:
        available = [
            point_data.GetArrayName(index)
            for index in range(point_data.GetNumberOfArrays())
            if point_data.GetArrayName(index) is not None
        ]
        raise ValueError(
            f"{source_path}: missing scalar point-data array; tried {candidates}; "
            f"available={available}"
        )
    if array.GetNumberOfComponents() != 1:
        raise ValueError(
            f"{source_path}: point-data array '{resolved_name}' must be scalar; "
            f"components={array.GetNumberOfComponents()}"
        )
    if array.GetNumberOfTuples() != poly.GetNumberOfPoints():
        raise ValueError(
            f"{source_path}: point-data array '{resolved_name}' count does not match "
            f"point count ({array.GetNumberOfTuples()} != {poly.GetNumberOfPoints()})"
        )
    values = np.asarray(vtk_to_numpy(array), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{source_path}: point-data array '{resolved_name}' contains non-finite values"
        )
    return array, resolved_name


def _copy_scalar_array(
    source: vtk.vtkDataArray,
    target_name: str,
) -> vtk.vtkDataArray:
    values = np.asarray(vtk_to_numpy(source), dtype=float).reshape(-1)
    target = numpy_to_vtk(values, deep=True)
    target.SetName(target_name)
    return target


def _strip_timeseries_arrays(point_data: vtk.vtkPointData) -> None:
    remove: list[str] = []
    for array_index in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(array_index)
        if name is None:
            continue
        lowered = name.casefold()
        if lowered in {"pressure", "velocity", "flow"} or lowered.startswith(
            ("pressure_", "velocity_", "flow_")
        ):
            remove.append(name)
    for name in remove:
        point_data.RemoveArray(name)


def _validate_cycle_duration(cycle_duration_s: float) -> float:
    try:
        duration = float(cycle_duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("cycle_duration_s must be numeric") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("cycle_duration_s must be positive and finite")
    return duration


def _validate_data_contract(
    *,
    pressure_units: str,
    flow_units: str,
    flow_quantity: str,
) -> None:
    if pressure_units != "mmHg":
        raise ValueError("centerline timeseries pressure units must be mmHg")
    if flow_units != "cm^3/s" or flow_quantity != "volumetric_flow":
        raise ValueError(
            "centerline timeseries flow must declare quantity=volumetric_flow and units=cm^3/s"
        )


def _build_timeseries_polydata(
    records: Sequence[Mapping[str, Any]],
    *,
    pressure_array: str,
    flow_array: str,
) -> tuple[vtk.vtkPolyData, list[dict[str, Any]], dict[str, str]]:
    reference_poly: vtk.vtkPolyData | None = None
    processed_frames: list[dict[str, Any]] = []
    source_names: dict[str, str] = {}
    for frame_index, record in enumerate(records):
        source_path = Path(str(record["path"]))
        poly = _read_polydata(source_path)
        if reference_poly is None:
            reference_poly = vtk.vtkPolyData()
            reference_poly.DeepCopy(poly)
            _strip_timeseries_arrays(reference_poly.GetPointData())
        else:
            _validate_matching_geometry(reference_poly, poly, source_path)

        pressure, resolved_pressure_name = _find_point_array(
            poly, pressure_array, _PRESSURE_ALIASES, source_path
        )
        flow, resolved_flow_name = _find_point_array(
            poly, flow_array, _FLOW_ALIASES, source_path
        )
        source_names.setdefault("pressure", resolved_pressure_name)
        source_names.setdefault("flow", resolved_flow_name)
        if source_names["pressure"] != resolved_pressure_name:
            # Array spelling may differ by case between VTK files, but changing
            # the semantic source array mid-series is ambiguous and unsafe.
            if source_names["pressure"].casefold() != resolved_pressure_name.casefold():
                raise ValueError(
                    "selected frames use different pressure source array names"
                )
        if source_names["flow"] != resolved_flow_name:
            if source_names["flow"].casefold() != resolved_flow_name.casefold():
                raise ValueError(
                    "selected frames use different flow source array names"
                )

        pressure_name = f"pressure_{frame_index}"
        flow_name = f"flow_{frame_index}"
        reference_poly.GetPointData().AddArray(
            _copy_scalar_array(pressure, pressure_name)
        )
        reference_poly.GetPointData().AddArray(_copy_scalar_array(flow, flow_name))
        frame_result: dict[str, Any] = {
            "frame_index": frame_index,
            "time_s": float(record["time_s"]),
            "point_arrays": [pressure_name, flow_name],
            "source_point_arrays": {
                "pressure": resolved_pressure_name,
                "flow": resolved_flow_name,
            },
        }
        if record.get("timestep_id") is not None:
            frame_result["timestep_id"] = int(record["timestep_id"])
        if record.get("source_frame_path") is not None:
            frame_result["source_frame_path"] = str(record["source_frame_path"])
        processed_frames.append(frame_result)

    if reference_poly is None:  # pragma: no cover - guarded by _as_records
        raise ValueError("selected_frames must contain at least one frame")
    return reference_poly, processed_frames, source_names


def _descriptor_artifact(
    *,
    output_dir: Path,
    vtp_path: Path,
    sidecar_path: Path,
    reference_centerline: Path,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    relative = lambda path: os.path.relpath(path, output_dir).replace(os.sep, "/")
    vtp_digest = _sha256_file(vtp_path)
    metadata_digest = _sha256_file(sidecar_path)
    reference_digest = _sha256_file(reference_centerline)
    artifact: dict[str, Any] = {
        "schema_version": CENTERLINE_TIMESERIES_SCHEMA_VERSION,
        "kind": CENTERLINE_TIMESERIES_KIND,
        "vtp": relative(vtp_path),
        "metadata": relative(sidecar_path),
        "reference_centerline": relative(reference_centerline),
        "vtp_sha256": vtp_digest,
        "metadata_sha256": metadata_digest,
        "reference_centerline_sha256": reference_digest,
        "frame_indices": list(sidecar["frame_indices"]),
        "timestamps_s": list(sidecar["timestamps_s"]),
        "cycle_duration_s": float(sidecar["cycle_duration_s"]),
        "frame_count": int(sidecar["frame_count"]),
        "point_count": int(sidecar["point_count"]),
        "cell_count": int(sidecar["cell_count"]),
        "source_array_names": dict(sidecar["source_array_names"]),
        "pressure_array": sidecar["source_array_names"]["pressure"],
        "flow_array": sidecar["source_array_names"]["flow"],
        "data_contract": dict(sidecar["data_contract"]),
        "digests": {
            "vtp": vtp_digest,
            "metadata": metadata_digest,
            "reference_centerline": reference_digest,
            "vtp_sha256": vtp_digest,
            "metadata_sha256": metadata_digest,
            "reference_centerline_sha256": reference_digest,
        },
    }
    # These explicit aliases make the descriptor convenient to consume without
    # changing the canonical short keys above.
    artifact["vtp_path"] = artifact["vtp"]
    artifact["metadata_json"] = artifact["metadata"]
    artifact["metadata_path"] = artifact["metadata"]
    artifact["sidecar_sha256"] = metadata_digest
    artifact["sidecar"] = artifact["metadata"]
    artifact["sidecar_path"] = artifact["metadata"]
    artifact["reference_centerline_path"] = artifact["reference_centerline"]
    artifact["vtp_relative"] = artifact["vtp"]
    artifact["metadata_relative"] = artifact["metadata"]
    artifact["reference_centerline_relative"] = artifact["reference_centerline"]
    return artifact


def _resolve_artifact_path(descriptor_dir: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"centerline timeseries descriptor is missing {name}")
    if Path(value).is_absolute():
        raise ValueError(f"centerline timeseries descriptor {name} must be relative")
    path = (descriptor_dir / value).resolve()
    return path


def _artifact_from_descriptor(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "artifacts" in descriptor:
        artifacts = descriptor.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("suite descriptor artifacts must be an object")
        artifact = artifacts.get("centerline_timeseries")
        if not isinstance(artifact, Mapping):
            raise ValueError(
                "suite descriptor artifacts.centerline_timeseries is required"
            )
        return artifact
    return descriptor


def validate_centerline_timeseries_descriptor(
    descriptor: str | Path | Mapping[str, Any],
    *,
    descriptor_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate descriptor paths, digests, sidecar identity, and VTP arrays."""

    if isinstance(descriptor, (str, Path)):
        descriptor_path = Path(descriptor).expanduser().resolve()
        try:
            payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"suite descriptor was not found: {descriptor_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"suite descriptor is not valid JSON: {descriptor_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("suite descriptor must contain an object")
        base_dir = descriptor_path.parent
    else:
        payload = descriptor
        base_dir = Path(descriptor_dir or ".").expanduser().resolve()
    if not isinstance(payload, Mapping):
        raise ValueError("centerline timeseries descriptor must contain an object")
    schema_version = payload.get("schema_version")
    if schema_version != CENTERLINE_TIMESERIES_SCHEMA_VERSION:
        raise ValueError(
            "unsupported centerline timeseries descriptor schema version: "
            f"{schema_version!r}"
        )
    artifact = _artifact_from_descriptor(payload)
    if artifact.get("schema_version") != CENTERLINE_TIMESERIES_SCHEMA_VERSION:
        raise ValueError(
            "centerline timeseries artifact has an unsupported schema version: "
            f"{artifact.get('schema_version')!r}"
        )
    if artifact.get("kind") != CENTERLINE_TIMESERIES_KIND:
        raise ValueError("centerline timeseries artifact has an invalid kind")
    vtp_path = _resolve_artifact_path(
        base_dir, artifact.get("vtp", artifact.get("vtp_path")), name="vtp"
    )
    metadata_path = _resolve_artifact_path(
        base_dir,
        artifact.get(
            "metadata", artifact.get("metadata_json", artifact.get("metadata_path"))
        ),
        name="metadata",
    )
    reference_path = _resolve_artifact_path(
        base_dir,
        artifact.get("reference_centerline", artifact.get("reference_centerline_path")),
        name="reference_centerline",
    )
    for path, label in (
        (vtp_path, "VTP"),
        (metadata_path, "sidecar"),
        (reference_path, "reference centerline"),
    ):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"centerline timeseries {label} is missing: {path}")
    digest_fields = artifact.get("digests")
    if not isinstance(digest_fields, Mapping):
        digest_fields = {}
    digest_aliases = {
        vtp_path: (
            artifact.get("vtp_sha256"),
            digest_fields.get("vtp_sha256"),
            digest_fields.get("vtp"),
        ),
        metadata_path: (
            artifact.get("metadata_sha256"),
            artifact.get("sidecar_sha256"),
            digest_fields.get("metadata_sha256"),
            digest_fields.get("metadata"),
            digest_fields.get("sidecar"),
        ),
        reference_path: (
            artifact.get("reference_centerline_sha256"),
            digest_fields.get("reference_centerline_sha256"),
            digest_fields.get("reference_centerline"),
        ),
    }
    resolved_digests: dict[Path, str] = {}
    for path, aliases in digest_aliases.items():
        declared = [value for value in aliases if value is not None]
        if not declared or any(not isinstance(value, str) for value in declared):
            raise ValueError(f"centerline timeseries digest is missing: {path}")
        if len(set(declared)) != 1 or _sha256_file(path) != declared[0]:
            raise ValueError(f"centerline timeseries digest mismatch: {path}")
        resolved_digests[path] = declared[0]

    try:
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"centerline timeseries sidecar is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(sidecar, Mapping):
        raise ValueError("centerline timeseries sidecar must contain an object")
    if sidecar.get("kind") != CENTERLINE_TIMESERIES_KIND:
        raise ValueError("centerline timeseries sidecar has an invalid kind")
    if sidecar.get("schema_version") != CENTERLINE_TIMESERIES_SCHEMA_VERSION:
        raise ValueError(
            "centerline timeseries sidecar has an unsupported schema version"
        )
    if sidecar.get("vtp_sha256") not in (None, resolved_digests[vtp_path]):
        raise ValueError(
            "centerline timeseries sidecar and descriptor disagree on VTP digest"
        )
    if sidecar.get("reference_centerline_sha256") not in (
        None,
        resolved_digests[reference_path],
    ):
        raise ValueError(
            "centerline timeseries sidecar and descriptor disagree on reference digest"
        )
    for field in (
        "frame_indices",
        "timestamps_s",
        "cycle_duration_s",
        "frame_count",
        "point_count",
        "cell_count",
    ):
        if sidecar.get(field) != artifact.get(field):
            raise ValueError(
                f"centerline timeseries sidecar and descriptor disagree on {field}"
            )
    _validate_cycle_duration(sidecar["cycle_duration_s"])
    timestamps = [float(value) for value in sidecar["timestamps_s"]]
    if any(not math.isfinite(value) for value in timestamps) or any(
        left >= right for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(
            "centerline timeseries timestamps must be strictly increasing and finite"
        )
    source_array_names = sidecar.get("source_array_names")
    if source_array_names != artifact.get("source_array_names"):
        raise ValueError(
            "centerline timeseries sidecar and descriptor disagree on source arrays"
        )
    sidecar_contract = sidecar.get("data_contract")
    if not isinstance(sidecar_contract, Mapping) or sidecar_contract != artifact.get(
        "data_contract"
    ):
        raise ValueError(
            "centerline timeseries sidecar and descriptor disagree on data contract"
        )
    pressure_contract = sidecar_contract.get("pressure")
    flow_contract = sidecar_contract.get("flow")
    if not isinstance(pressure_contract, Mapping) or not isinstance(
        flow_contract, Mapping
    ):
        raise ValueError(
            "centerline timeseries sidecar data_contract must declare pressure and flow"
        )
    _validate_data_contract(
        pressure_units=str(pressure_contract.get("units")),
        flow_units=str(flow_contract.get("units")),
        flow_quantity=str(flow_contract.get("quantity")),
    )
    if pressure_contract.get("quantity") != "pressure":
        raise ValueError(
            "centerline timeseries pressure data contract must declare quantity=pressure"
        )

    def _descriptor_count(name: str, *, minimum: int) -> int:
        value = artifact.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"centerline timeseries {name} must be an integer")
        integer = value
        if integer < minimum:
            raise ValueError(
                f"centerline timeseries {name} must be an integer >= {minimum}"
            )
        return integer

    frame_count = _descriptor_count("frame_count", minimum=1)
    point_count = _descriptor_count("point_count", minimum=1)
    cell_count = _descriptor_count("cell_count", minimum=0)
    frame_indices = artifact.get("frame_indices")
    if not isinstance(frame_indices, list) or len(frame_indices) != frame_count:
        raise ValueError(
            "centerline timeseries frame_indices length does not match frame_count"
        )
    if frame_indices != list(range(frame_count)):
        raise ValueError(
            "centerline timeseries frame_indices must be contiguous from zero"
        )
    descriptor_timestamps = artifact.get("timestamps_s")
    if not isinstance(descriptor_timestamps, list) or len(
        descriptor_timestamps
    ) != frame_count:
        raise ValueError(
            "centerline timeseries timestamps_s must match frame_count"
        )
    try:
        timestamps = [float(value) for value in descriptor_timestamps]
        cycle_duration = _validate_cycle_duration(artifact["cycle_duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "centerline timeseries timing fields are malformed"
        ) from exc
    if (
        any(not math.isfinite(value) for value in timestamps)
        or any(left >= right for left, right in zip(timestamps, timestamps[1:]))
        or timestamps[-1] - timestamps[0] >= cycle_duration
    ):
        raise ValueError(
            "centerline timeseries timestamps must be strictly increasing and "
            "contained within one cycle"
        )
    processed_frames = sidecar.get("processed_frames")
    if not isinstance(processed_frames, list) or len(processed_frames) != frame_count:
        raise ValueError(
            "centerline timeseries processed_frames must contain one record per frame"
        )
    for index, frame in enumerate(processed_frames):
        if not isinstance(frame, Mapping):
            raise ValueError(
                "centerline timeseries processed_frames entries must be objects"
            )
        if (
            frame.get("frame_index") != index
            or frame.get("time_s") != artifact["timestamps_s"][index]
            or frame.get("point_arrays")
            != [f"pressure_{index}", f"flow_{index}"]
        ):
            raise ValueError(
                "centerline timeseries processed_frames do not match numbered arrays"
            )

    poly = _read_polydata(vtp_path)
    reference_poly = _read_polydata(reference_path)
    _validate_matching_geometry(reference_poly, poly, vtp_path)
    if poly.GetNumberOfPoints() != point_count or poly.GetNumberOfCells() != cell_count:
        raise ValueError(
            "centerline timeseries descriptor geometry counts do not match VTP"
        )
    expected_indices = list(range(frame_count))
    point_data = poly.GetPointData()
    for index in expected_indices:
        for prefix in ("pressure", "flow"):
            name = f"{prefix}_{index}"
            array = point_data.GetArray(name)
            if array is None or array.GetNumberOfComponents() != 1:
                raise ValueError(
                    f"centerline timeseries VTP is missing scalar array {name}"
                )
            values = np.asarray(vtk_to_numpy(array), dtype=float).reshape(-1)
            if (
                values.size != point_count
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"centerline timeseries array {name} has invalid values or count"
                )
    return {
        "descriptor": dict(payload),
        "artifact": dict(artifact),
        "vtp_path": str(vtp_path),
        "metadata_path": str(metadata_path),
        "reference_centerline": str(reference_path),
    }


def publish_centerline_timeseries(
    *,
    selected_frames: Sequence[Mapping[str, Any]]
    | Iterable[Mapping[str, Any]]
    | Any
    | None = None,
    output_dir: str | Path,
    cycle_duration_s: float | None = None,
    reference_centerline: str | Path | None = None,
    centerline: str | Path | None = None,
    resistance_map_metadata_json: str | Path | None = None,
    suite_metadata_path: str | Path | None = None,
    tuned_zerod_config_path: str | Path | None = None,
    pressure_array: str = "pressure",
    flow_array: str = "velocity",
    pressure_units: str = "mmHg",
    flow_units: str = "cm^3/s",
    flow_quantity: str = "volumetric_flow",
) -> dict[str, Any]:
    """Publish the selected mapped centerline frames and suite descriptor.

    ``selected_frames`` must be the records returned by the resistance-map
    computation.  Passing ``resistance_map_metadata_json`` is a convenience for
    the pulmonary suite and reads that exact selected-frame set without making
    another selection decision.
    """

    if resistance_map_metadata_json is not None:
        metadata_frames, metadata_cycle_duration = _load_selected_frames_metadata(
            Path(resistance_map_metadata_json).expanduser().resolve()
        )
        if selected_frames is not None:
            raise ValueError(
                "provide selected_frames or resistance_map_metadata_json, not both"
            )
        selected_frames = metadata_frames
        if cycle_duration_s is None:
            cycle_duration_s = metadata_cycle_duration
    if selected_frames is None:
        raise ValueError("selected_frames or resistance_map_metadata_json is required")
    records = _as_records(selected_frames)
    if cycle_duration_s is None:
        raise ValueError("cycle_duration_s is required")
    duration = _validate_cycle_duration(cycle_duration_s)
    _validate_data_contract(
        pressure_units=pressure_units,
        flow_units=flow_units,
        flow_quantity=flow_quantity,
    )
    reference_value = (
        reference_centerline if reference_centerline is not None else centerline
    )
    if reference_value is None:
        raise ValueError("reference_centerline is required")
    reference_path = Path(reference_value).expanduser().resolve()
    if not reference_path.exists() or not reference_path.is_file():
        raise FileNotFoundError(f"reference centerline is missing: {reference_path}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    vtp_path = output_path / "centerline_timeseries_last_cycle.vtp"
    sidecar_path = output_path / "centerline_timeseries_last_cycle_metadata.json"
    descriptor_path = (
        Path(suite_metadata_path).expanduser().resolve()
        if suite_metadata_path is not None
        else output_path / "postprocess_suite_metadata.json"
    )

    # Parse the caller-supplied descriptor before publishing any files.  Its
    # bytes are retained so a failed update can restore the exact payload,
    # including unrelated artifacts published by another producer.
    existing_descriptor_bytes: bytes | None = None
    existing_descriptor_payload: Mapping[str, Any] | None = None
    if descriptor_path.exists():
        existing_descriptor_bytes = descriptor_path.read_bytes()
        try:
            parsed_descriptor = json.loads(existing_descriptor_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"suite descriptor is not valid JSON: {descriptor_path}") from exc
        if not isinstance(parsed_descriptor, Mapping):
            raise ValueError("suite descriptor must contain an object")
        existing_descriptor_payload = parsed_descriptor

    lineage: dict[str, str] | None = None
    if tuned_zerod_config_path is not None:
        lineage = _tuned_zerod_lineage(
            tuned_zerod_config_path,
            descriptor_dir=descriptor_path.parent,
        )
        if existing_descriptor_payload is not None:
            existing_lineage = existing_descriptor_payload.get("lineage")
            if existing_lineage is not None:
                if not isinstance(existing_lineage, Mapping):
                    raise ValueError("suite descriptor lineage must contain an object")
                _validate_existing_lineage(
                    existing_lineage,
                    lineage,
                    descriptor_dir=descriptor_path.parent,
                )
    elif existing_descriptor_payload is not None:
        existing_lineage = existing_descriptor_payload.get("lineage")
        if existing_lineage is not None:
            if not isinstance(existing_lineage, Mapping):
                raise ValueError("suite descriptor lineage must contain an object")
            # Preserve an existing identity in the returned result so a
            # pulmonary-suite finalization cannot accidentally drop lineage
            # when it rewrites the merged descriptor.
            lineage = dict(existing_lineage)

    # Atomic replacement is used for the public files below.  Keep the old
    # bytes, when present, so a later validation failure does not replace a
    # previously published centerline series with a partial attempt.
    previous_public_files: dict[Path, bytes | None] = {
        path: path.read_bytes() if path.exists() else None
        for path in (vtp_path, sidecar_path)
    }
    try:
        poly, processed_frames, source_names = _build_timeseries_polydata(
            records,
            pressure_array=pressure_array,
            flow_array=flow_array,
        )
        timestamps = [float(record["time_s"]) for record in records]
        frame_indices = list(range(len(records)))
        if len(processed_frames) != len(frame_indices):  # pragma: no cover
            raise RuntimeError(
                "centerline timeseries frame construction lost a selected frame"
            )
        data_contract = {
            "pressure": {
                "quantity": "pressure",
                "units": pressure_units,
                "array_prefix": "pressure",
                "source_array": source_names["pressure"],
            },
            "flow": {
                "quantity": flow_quantity,
                "units": flow_units,
                "array_prefix": "flow",
                "source_array": source_names["flow"],
                "source_semantics": "svSlicer velocity dot normal integrated over slice",
            },
        }
        sidecar: dict[str, Any] = {
            "kind": CENTERLINE_TIMESERIES_KIND,
            "schema_version": CENTERLINE_TIMESERIES_SCHEMA_VERSION,
            "output_path": str(vtp_path),
            "metadata_json": str(sidecar_path),
            "frame_count": len(records),
            "selected_frame_count": len(records),
            "point_count": int(poly.GetNumberOfPoints()),
            "cell_count": int(poly.GetNumberOfCells()),
            "frame_indices": frame_indices,
            "timestamps_s": timestamps,
            "cycle_duration_s": duration,
            "source_array_names": source_names,
            "zerod_point_arrays": [
                name for frame in processed_frames for name in frame["point_arrays"]
            ],
            "pressure_array": source_names["pressure"],
            "flow_array": source_names["flow"],
            "data_contract": data_contract,
            "processed_frames": processed_frames,
        }
        _atomic_publish_vtp(poly, vtp_path)
        sidecar["vtp"] = vtp_path.name
        sidecar["reference_centerline"] = os.path.relpath(
            reference_path, sidecar_path.parent
        ).replace(os.sep, "/")
        sidecar["source_resistance_map_metadata_json"] = (
            str(Path(resistance_map_metadata_json).expanduser().resolve())
            if resistance_map_metadata_json is not None
            else None
        )
        sidecar["vtp_sha256"] = _sha256_file(vtp_path)
        sidecar["reference_centerline_sha256"] = _sha256_file(reference_path)
        # The sidecar is itself staged atomically.  Its contents intentionally
        # exclude its own digest because a self-digest would be a fixed-point
        # problem; the suite descriptor records the sidecar digest.
        _atomic_write_json(sidecar_path, sidecar)
        artifact = _descriptor_artifact(
            output_dir=descriptor_path.parent,
            vtp_path=vtp_path,
            sidecar_path=sidecar_path,
            reference_centerline=reference_path,
            sidecar=sidecar,
        )
        # Validate the bytes and all geometry before exposing the descriptor.
        descriptor_payload: dict[str, Any] = (
            dict(existing_descriptor_payload)
            if existing_descriptor_payload is not None
            else {
                "schema_version": CENTERLINE_TIMESERIES_SCHEMA_VERSION,
                "artifacts": {},
            }
        )
        descriptor_payload["schema_version"] = CENTERLINE_TIMESERIES_SCHEMA_VERSION
        artifacts = descriptor_payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            artifacts = {}
        descriptor_payload["artifacts"] = dict(artifacts)
        descriptor_payload["artifacts"]["centerline_timeseries"] = artifact
        if lineage is not None:
            descriptor_payload["lineage"] = lineage
        validate_centerline_timeseries_descriptor(
            descriptor_payload,
            descriptor_dir=descriptor_path.parent,
        )
        _atomic_write_json(descriptor_path, descriptor_payload)
        return {
            "kind": CENTERLINE_TIMESERIES_KIND,
            "schema_version": CENTERLINE_TIMESERIES_SCHEMA_VERSION,
            "vtp": str(vtp_path),
            "output_path": str(vtp_path),
            "metadata": str(sidecar_path),
            "metadata_json": str(sidecar_path),
            "source_resistance_map_metadata_json": (
                str(Path(resistance_map_metadata_json).expanduser().resolve())
                if resistance_map_metadata_json is not None
                else None
            ),
            "descriptor": str(descriptor_path),
            "artifact": artifact,
            "frame_count": len(records),
            "selected_frame_count": len(records),
            "point_count": int(poly.GetNumberOfPoints()),
            "cell_count": int(poly.GetNumberOfCells()),
            "frame_indices": frame_indices,
            "timestamps_s": timestamps,
            "cycle_duration_s": duration,
            "source_array_names": source_names,
            "zerod_point_arrays": [
                name for frame in processed_frames for name in frame["point_arrays"]
            ],
            "data_contract": data_contract,
            "processed_frames": processed_frames,
            "lineage": dict(lineage) if lineage is not None else None,
        }
    except Exception:
        # A descriptor is the publication gate.  If anything after staging
        # fails, restore caller-owned content and remove only files created by
        # this attempt.  In particular, never unlink a descriptor that also
        # contains resistance-map or other independently published records.
        if existing_descriptor_bytes is None:
            descriptor_path.unlink(missing_ok=True)
        else:
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor_path.write_bytes(existing_descriptor_bytes)
        for path, previous_bytes in previous_public_files.items():
            if previous_bytes is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(previous_bytes)
        raise


def write_centerline_timeseries(**kwargs: Any) -> dict[str, Any]:
    """Supported public alias for :func:`publish_centerline_timeseries`."""

    return publish_centerline_timeseries(**kwargs)


__all__ = [
    "CENTERLINE_TIMESERIES_KIND",
    "CENTERLINE_TIMESERIES_SCHEMA_VERSION",
    "publish_centerline_timeseries",
    "validate_centerline_timeseries_descriptor",
    "write_centerline_timeseries",
]
