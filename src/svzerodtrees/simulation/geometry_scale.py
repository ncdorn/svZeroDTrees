"""Scaled copies of a patient's geometry inputs.

``mesh_scale_factor`` s says the true anatomy is the mesh scaled by s.  Applying
s only inside the 3D solver leaves every other consumer (learned 0D seed,
tuning cap areas, proximal compliance, svSlicer cross-sections, calibration
lengths) at the unscaled size: slice flows come out s**-2 too small and the 0D
and 3D models describe different geometry (TST-STAN-1 s=1.147, TST-STAN-9
s=1.425).  ``stage_scaled_geometry`` writes scaled copies of the mesh-complete
directories, the centerline, and the source 0D config instead, so every stage
reads the same geometry and the 3D solver runs with a scale factor of 1.

Scaling rules (lengths times s, same fluid and wall material):

- mesh and centerline coordinates, ``Path``, radii, section sizes: x s;
  ``CenterlineSectionArea``: x s**2;
- 0D vessels: ``vessel_length`` x s, ``R_poiseuille`` (8 mu L / pi r^4) x s**-3,
  ``L`` (rho L / pi r^2) x s**-1, ``C`` (3 A L / 2 (Eh/r)) x s**3 at constant
  Eh/r (the proximal-compliance and large-radius Olufsen wall model),
  ``stenosis_coefficient`` (~ 1 / A^2) x s**-4; ``geometric_params`` areas x s**2
  and lengths/radii x s; junction ``areas`` x s**2 and ``lengths`` x s;
- boundary conditions, simulation parameters, tangents, ids: unchanged.

The 0D rules are exact for Poiseuille/ROM element values; the scaled config is
meant as the seed for learnedZeroD, which is regenerated from the scaled
centerline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

SCALED_GEOMETRY_MANIFEST = "scaled_geometry.json"

_CENTERLINE_LENGTH_ARRAYS = (
    "Path",
    "MaximumInscribedSphereRadius",
    "CenterlineSectionMaxSize",
    "CenterlineSectionMinSize",
)
_CENTERLINE_AREA_ARRAYS = ("CenterlineSectionArea",)
_VESSEL_VALUE_EXPONENTS = {
    "R_poiseuille": -3,
    "L": -1,
    # s**4 (constant E h) raised STAN-9's baseline compliance enough to tip its
    # near-zero LPA flow split negative in learnedZeroD's features (2026-10-15).
    "C": 3,
    "stenosis_coefficient": -4,
}
_GEOMETRIC_AREA_KEYS = ("inlet_area", "outlet_area", "area")
_GEOMETRIC_LENGTH_KEYS = ("length", "radius", "inlet_radius", "outlet_radius")


def _check_scale(scale: float) -> float:
    value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return value


def _scaled_number(value: Any, factor: float) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return value  # inf (rigid / unset) and nan stay as they are
    return float(value) * factor


def _scaled_values(values: Any, factor: float) -> Any:
    if isinstance(values, list):
        return [_scaled_number(v, factor) for v in values]
    return _scaled_number(values, factor)


def scale_zerod_payload(payload: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Return a copy of a 0D config with its geometry scaled by ``scale``."""
    s = _check_scale(scale)
    scaled = copy.deepcopy(dict(payload))
    for vessel in scaled.get("vessels") or []:
        if "vessel_length" in vessel:
            vessel["vessel_length"] = _scaled_number(vessel["vessel_length"], s)
        values = vessel.get("zero_d_element_values")
        if isinstance(values, dict):
            for key, exponent in _VESSEL_VALUE_EXPONENTS.items():
                if key in values:
                    values[key] = _scaled_values(values[key], s**exponent)
        geometry = vessel.get("geometric_params")
        if isinstance(geometry, dict):
            for key in _GEOMETRIC_AREA_KEYS:
                if key in geometry:
                    geometry[key] = _scaled_values(geometry[key], s**2)
            for key in _GEOMETRIC_LENGTH_KEYS:
                if key in geometry:
                    geometry[key] = _scaled_values(geometry[key], s)
    for junction in scaled.get("junctions") or []:
        if "areas" in junction:
            junction["areas"] = _scaled_values(junction["areas"], s**2)
        if "lengths" in junction:
            junction["lengths"] = _scaled_values(junction["lengths"], s)
        values = junction.get("junction_values")
        if isinstance(values, dict):
            for key, exponent in _VESSEL_VALUE_EXPONENTS.items():
                if key in values:
                    values[key] = _scaled_values(values[key], s**exponent)
    return scaled


def _read_vtk(path: Path):
    reader = vtk.vtkXMLUnstructuredGridReader() if path.suffix == ".vtu" else vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"cannot read VTK geometry: {path}")
    return data


def _write_vtk(data, path: Path) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter() if path.suffix == ".vtu" else vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {path}")


def _scale_points(data, scale: float) -> None:
    coordinates = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64) * scale
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(coordinates), deep=True))
    data.SetPoints(points)


def _scale_point_array(data, name: str, factor: float) -> None:
    array = data.GetPointData().GetArray(name)
    if array is None:
        return
    values = vtk_to_numpy(array).astype(np.float64) * factor
    scaled = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=array.GetDataType())
    scaled.SetName(name)
    data.GetPointData().RemoveArray(name)
    data.GetPointData().AddArray(scaled)


def scale_centerline(source: str | Path, output: str | Path, scale: float) -> None:
    s = _check_scale(scale)
    data = _read_vtk(Path(source))
    _scale_points(data, s)
    for name in _CENTERLINE_LENGTH_ARRAYS:
        _scale_point_array(data, name, s)
    for name in _CENTERLINE_AREA_ARRAYS:
        _scale_point_array(data, name, s**2)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    _write_vtk(data, Path(output))


def scale_mesh_complete(source_dir: str | Path, output_dir: str | Path, scale: float) -> int:
    """Copy a mesh-complete directory with every .vtu/.vtp point scaled.

    Other files are copied unchanged.  Returns the number of scaled VTK files.
    """
    s = _check_scale(scale)
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"mesh-complete directory not found: {source_dir}")
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if path.is_dir():
            continue
        target = output_dir / path.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".vtu", ".vtp"}:
            data = _read_vtk(path)
            _scale_points(data, s)
            _write_vtk(data, target)
            count += 1
        else:
            shutil.copy2(path, target)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def stage_scaled_geometry(
    staged_root: str | Path,
    scale: float,
    *,
    sources: Mapping[str, str | None],
    targets: Mapping[str, str | None],
) -> dict[str, Any]:
    """Write scaled geometry inputs once and reuse them afterwards.

    ``sources``/``targets`` map ``preop_mesh_complete``, ``postop_mesh_complete``
    (directories), ``centerline`` and ``zerod_config`` (files) to their source
    and scaled paths; a ``None`` source is skipped.  Targets must lie inside
    ``staged_root``.  The record in ``staged_root/scaled_geometry.json`` holds
    the scale and the source digests; an existing record with the same scale
    and sources is reused, a different one is an error (never overwritten).
    The copies are built in a temporary sibling directory and moved into place.
    """
    s = _check_scale(scale)
    root = Path(staged_root)
    kinds = {"preop_mesh_complete": "dir", "postop_mesh_complete": "dir", "centerline": "file", "zerod_config": "file"}
    unknown = set(sources) - set(kinds)
    if unknown:
        raise ValueError(f"unknown geometry inputs: {sorted(unknown)}")
    entries = {}
    for name, source in sources.items():
        if source is None:
            continue
        target = targets.get(name)
        if not target:
            raise ValueError(f"no target path for {name}")
        relative = Path(target).relative_to(root)  # raises if outside staged_root
        source_path = Path(source)
        digest = _directory_digest(source_path) if kinds[name] == "dir" else _sha256(source_path)
        entries[name] = {"source": str(source_path), "source_sha256": digest, "target": str(relative)}
    record = {"scale": s, "inputs": entries}

    manifest_path = root / SCALED_GEOMETRY_MANIFEST

    def _reuse() -> dict[str, Any] | None:
        if not manifest_path.exists():
            return None
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {
            name: {key: entry.get(key) for key in ("source", "source_sha256", "target")}
            for name, entry in (existing.get("inputs") or {}).items()
        }
        if existing.get("scale") == s and identity == entries:
            return {**existing, "staged_root": str(root), "reused": True}
        raise ValueError(
            f"{manifest_path} was staged from other inputs or another scale; "
            "remove it explicitly to restage"
        )

    reused = _reuse()
    if reused is not None:
        return reused
    if root.exists():
        raise ValueError(f"{root} exists without {SCALED_GEOMETRY_MANIFEST}; remove it to restage")

    building = root.with_name(root.name + f".building-{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        for name, entry in entries.items():
            target = building / entry["target"]
            if name.endswith("mesh_complete"):
                entry["vtk_files"] = scale_mesh_complete(entry["source"], target, s)
            elif name == "centerline":
                scale_centerline(entry["source"], target, s)
            else:
                payload = json.loads(Path(entry["source"]).read_text(encoding="utf-8"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(scale_zerod_payload(payload, s), indent=2) + "\n", encoding="utf-8")
        record = {"scale": s, "inputs": entries}
        (building / SCALED_GEOMETRY_MANIFEST).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(building, root)
        except OSError:
            # Another job staged the same inputs first; use its copy.
            shutil.rmtree(building, ignore_errors=True)
            reused = _reuse()
            if reused is None:
                raise
            return reused
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return {**record, "staged_root": str(root), "reused": False}
