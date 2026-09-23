"""Physically based pairing of mesh caps with centerline-generated outlet BCs.

A 0D model generated from a centerline names its vessels ``branch{N}_seg{K}``,
where ``N`` is the centerline ``BranchId``.  Each outlet cap of the 3D mesh is
paired with the centerline outlet endpoint nearest to the cap's area-weighted
centroid.  That endpoint's ``BranchId`` selects the terminal 0D vessel, and
therefore the outlet BC attached to it.

Suffixes added by learnedZeroD (``branch{N}_seg{K}_connectorEL``) keep the
``branch{N}`` prefix and are accepted.

Every step is checked rather than assumed:

- the 0D branch set must match the centerline branch set, and the 0D model
  must be traceable to this centerline: outlet vessels that record
  ``centerline_node_ids`` must end on their branch's terminal node
  (learnedZeroD seeds); otherwise every 0D branch length must match the
  centerline branch length;
- each cap centroid must lie within ``max_offset`` cap radii of its endpoint;
- the runner-up endpoint must be at least ``min_margin`` cap radii farther
  away than the matched one, so no pairing is a near tie;
- the resulting pairing must be one-to-one.

Distances are compared in units of the cap's equivalent radius
``sqrt(area / pi)`` computed from the same raw geometry, so the checks are
independent of the mesh length unit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


DEFAULT_MAX_OFFSET = 1.0
DEFAULT_MIN_MARGIN = 1.0
DEFAULT_LENGTH_RTOL = 0.02

_VESSEL_NAME = re.compile(r"^branch(\d+)_seg\d+(?:_\w+)?$")


def _read_polydata(path: str | Path):
    import vtk

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VTP file not found: {path}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"VTP file contains no points: {path}")
    return poly


def cap_geometry(path: str | Path) -> tuple[np.ndarray, float]:
    """Return the area-weighted centroid and area of a cap surface.

    Both values are in the raw units of the VTP file.
    """

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(_read_polydata(path))
    triangles.Update()
    mesh = triangles.GetOutput()
    if mesh.GetNumberOfPolys() == 0:
        raise ValueError(f"cap surface has no polygons: {path}")
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = (points[faces[:, k]] for k in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"cap surface has zero area: {path}")
    centroid = (areas[:, None] * (a + b + c) / 3.0).sum(axis=0) / total
    return centroid, total


def centerline_branches(
    path: str | Path,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, float]]:
    """Return terminal endpoints and arc lengths per centerline ``BranchId``.

    Terminal endpoints are degree-1 nodes of the centerline line graph, as
    ``(node_id, xyz)`` where ``node_id`` is the ``GlobalNodeId`` when present
    and the point index otherwise.  A terminal node without a branch id
    (``BranchId < 0``) is attributed to the first branch reached by walking
    inward.  Branch length is the summed length of unique edges whose two
    nodes share that ``BranchId``.
    """

    from vtk.util.numpy_support import vtk_to_numpy

    poly = _read_polydata(path)
    point_data = poly.GetPointData()
    branch_array = point_data.GetArray("BranchId")
    if branch_array is None:
        raise ValueError(f"centerline has no BranchId point array: {path}")
    branch = vtk_to_numpy(branch_array).astype(int)
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    global_ids = point_data.GetArray("GlobalNodeId")
    node_ids = (
        vtk_to_numpy(global_ids).astype(int)
        if global_ids is not None
        else np.arange(len(points))
    )

    lines = poly.GetLines()
    connectivity = vtk_to_numpy(lines.GetConnectivityArray())
    offsets = vtk_to_numpy(lines.GetOffsetsArray())
    edges = set()
    for start, stop in zip(offsets[:-1], offsets[1:]):
        chain = connectivity[start:stop]
        for u, v in zip(chain[:-1], chain[1:]):
            if u != v:
                edges.add((int(min(u, v)), int(max(u, v))))
    if not edges:
        raise ValueError(f"centerline has no line segments: {path}")

    neighbours: dict[int, list[int]] = {}
    lengths: dict[int, float] = {}
    for u, v in edges:
        neighbours.setdefault(u, []).append(v)
        neighbours.setdefault(v, []).append(u)
        if branch[u] == branch[v] and branch[u] >= 0:
            lengths[int(branch[u])] = lengths.get(int(branch[u]), 0.0) + float(
                np.linalg.norm(points[u] - points[v])
            )

    endpoints: dict[int, list[tuple[int, np.ndarray]]] = {}
    for node, adjacent in neighbours.items():
        if len(adjacent) != 1:
            continue
        previous, current = None, node
        while branch[current] < 0:
            onward = [n for n in neighbours[current] if n != previous]
            if len(onward) != 1:
                raise ValueError(
                    f"centerline terminal point {node} is not attached to a branch: {path}"
                )
            previous, current = current, onward[0]
        endpoints.setdefault(int(branch[current]), []).append(
            (int(node_ids[node]), points[node])
        )
    return endpoints, lengths


def _outlet_branches(
    vessels: Any, outlet_names: Sequence[str]
) -> tuple[dict[str, tuple[int, str, int | None]], dict[int, float]]:
    """Map each outlet BC to ``(branch_id, vessel_name, outlet_node_id)``.

    ``outlet_node_id`` is the vessel's recorded ``centerline_node_ids.outlet``
    or ``None``.  Also returns the summed 0D vessel length per branch.
    """

    if not isinstance(vessels, Sequence) or not vessels:
        raise ValueError("the 0D config has no serialized vessels")

    by_bc: dict[str, list[tuple[int, str, int | None]]] = {}
    lengths: dict[int, float] = {}
    for vessel in vessels:
        name = str(vessel.get("vessel_name", ""))
        match = _VESSEL_NAME.match(name)
        if match is None:
            raise ValueError(
                "centerline mapping requires centerline-generated vessel names "
                f"'branch<N>_seg<K>'; found '{name}'"
            )
        branch_id = int(match.group(1))
        lengths[branch_id] = lengths.get(branch_id, 0.0) + float(vessel.get("vessel_length", 0.0))
        outlet = (vessel.get("boundary_conditions") or {}).get("outlet")
        if outlet is not None:
            node_id = (vessel.get("centerline_node_ids") or {}).get("outlet")
            by_bc.setdefault(str(outlet), []).append(
                (branch_id, name, None if node_id is None else int(node_id))
            )

    result = {}
    for bc_name in outlet_names:
        attached = by_bc.get(bc_name, [])
        if len(attached) != 1:
            raise ValueError(
                f"outlet BC '{bc_name}' must attach to exactly one 0D vessel; "
                f"found {len(attached)}"
            )
        result[bc_name] = attached[0]
    branch_ids = [branch_id for branch_id, _, _ in result.values()]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("more than one outlet BC attaches to the same 0D branch")
    return result, lengths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centerline_cap_pairs(
    config_handler: Any,
    cap_paths: Sequence[str],
    outlet_names: Sequence[str],
    centerline: str | Path,
    *,
    seed_payload: Mapping[str, Any] | None = None,
    convert_to_cm: bool = False,
    max_offset: float = DEFAULT_MAX_OFFSET,
    min_margin: float = DEFAULT_MIN_MARGIN,
    length_rtol: float = DEFAULT_LENGTH_RTOL,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Pair caps with outlet BCs through the centerline geometry.

    Vessels are read from ``seed_payload``, the serialized 0D config, when
    given, and otherwise from ``config_handler._config``.  ``ConfigHandler``
    does not retain ``centerline_node_ids``, so pass the payload for
    learnedZeroD seeds.  ``convert_to_cm`` declares that the mesh and centerline are in mm while
    the 0D model is in cm, so 0D branch lengths are compared with 0.1 times
    the centerline lengths.  Returns ``(pairs, provenance)``; raises
    ``ValueError`` with every failed check when the pairing is not certain.
    """

    centerline_path = Path(centerline)
    source = seed_payload
    if source is None:
        source = getattr(config_handler, "_config", None)
    vessels = source.get("vessels") if isinstance(source, Mapping) else None
    bc_branches, zerod_lengths = _outlet_branches(vessels, outlet_names)
    endpoints, centerline_lengths = centerline_branches(centerline_path)

    zerod_ids = set(zerod_lengths)
    centerline_ids = set(centerline_lengths) | set(endpoints)
    if zerod_ids != centerline_ids:
        raise ValueError(
            "0D branches do not match the centerline branches; the 0D model was "
            "not generated from this centerline "
            f"(0D only: {sorted(zerod_ids - centerline_ids)}, "
            f"centerline only: {sorted(centerline_ids - zerod_ids)})"
        )

    candidates = []
    for bc_name, (branch_id, vessel_name, node_id) in bc_branches.items():
        found = endpoints.get(branch_id, [])
        if len(found) != 1:
            raise ValueError(
                f"outlet branch {branch_id} ({bc_name}) must have exactly one "
                f"centerline endpoint; found {len(found)}"
            )
        candidates.append((bc_name, branch_id, vessel_name, node_id, *found[0]))
    if len(candidates) < 2:
        raise ValueError("centerline mapping requires at least two outlet endpoints")

    scale = 0.1 if convert_to_cm else 1.0
    consistency: dict[str, Any]
    if all(candidate[3] is not None for candidate in candidates):
        # learnedZeroD folds branch length into junctions, so the recorded
        # outlet node is the evidence that ties the seed to this centerline.
        mismatched = [
            f"{bc_name} (branch {branch_id}: node {node_id}, endpoint {endpoint_id})"
            for bc_name, branch_id, _, node_id, endpoint_id, _ in candidates
            if node_id != endpoint_id
        ]
        if mismatched:
            raise ValueError(
                "0D outlet vessels do not end on their centerline branch endpoints; "
                "the 0D model was not generated from this centerline: "
                + ", ".join(mismatched)
            )
        consistency = {"seed_consistency_check": "centerline_node_ids"}
    else:
        if any(zerod_lengths[branch_id] <= 0.0 for _, branch_id, *_ in candidates):
            raise ValueError(
                "0D outlet vessels have zero length and record no centerline_node_ids, "
                "so the 0D model cannot be checked against this centerline; for a "
                "learnedZeroD seed, pass the serialized seed payload"
            )
        length_errors = {
            branch_id: abs(zerod_lengths[branch_id] - scale * centerline_lengths.get(branch_id, 0.0))
            / max(zerod_lengths[branch_id], 1e-12)
            for branch_id in zerod_ids
        }
        worst_branch = max(length_errors, key=length_errors.get)
        if length_errors[worst_branch] > length_rtol:
            raise ValueError(
                "0D branch lengths do not match the centerline; the 0D model was not "
                f"generated from this centerline (branch {worst_branch}: 0D length "
                f"{zerod_lengths[worst_branch]:.6g}, centerline length x {scale:g} = "
                f"{scale * centerline_lengths.get(worst_branch, 0.0):.6g})"
            )
        consistency = {
            "seed_consistency_check": "branch_lengths",
            "length_rtol": float(length_rtol),
            "length_scale": scale,
            "max_branch_length_rel_error": float(length_errors[worst_branch]),
        }
    endpoint_xyz = np.array([candidate[5] for candidate in candidates])

    pairs = []
    evidence = []
    failures = []
    claimed: dict[str, str] = {}
    for cap_path in cap_paths:
        stem = Path(cap_path).stem
        centroid, area = cap_geometry(cap_path)
        radius = math.sqrt(area / math.pi)
        distances = np.linalg.norm(endpoint_xyz - centroid, axis=1)
        nearest, runner_up = np.argsort(distances)[:2]
        bc_name, branch_id, vessel_name, _, _, endpoint = candidates[nearest]
        offset = float(distances[nearest]) / radius
        margin = float(distances[runner_up] - distances[nearest]) / radius
        if offset > max_offset:
            failures.append(
                f"cap '{stem}' is {offset:.2f} cap radii from the nearest outlet "
                f"endpoint (branch {branch_id}); limit is {max_offset:g}"
            )
        if margin < min_margin:
            failures.append(
                f"cap '{stem}' is ambiguous between branch {branch_id} and branch "
                f"{candidates[runner_up][1]} (margin {margin:.2f} cap radii; "
                f"limit is {min_margin:g})"
            )
        if bc_name in claimed:
            failures.append(
                f"caps '{claimed[bc_name]}' and '{stem}' both match branch {branch_id}"
            )
        claimed[bc_name] = stem
        pairs.append((cap_path, bc_name))
        evidence.append(
            {
                "cap_stem": stem,
                "bc_name": bc_name,
                "branch_id": branch_id,
                "vessel_name": vessel_name,
                "cap_centroid": [float(v) for v in centroid],
                "endpoint": [float(v) for v in endpoint],
                "cap_radius": radius,
                "offset_cap_radii": offset,
                "margin_cap_radii": margin,
            }
        )
    if failures:
        raise ValueError("; ".join(failures))

    provenance = {
        "centerline": str(centerline_path),
        "centerline_sha256": _sha256(centerline_path),
        **consistency,
        "max_offset": float(max_offset),
        "min_margin": float(min_margin),
        "max_offset_observed": max(item["offset_cap_radii"] for item in evidence),
        "min_margin_observed": min(item["margin_cap_radii"] for item in evidence),
        "pairs": evidence,
    }
    return pairs, provenance


__all__ = [
    "DEFAULT_LENGTH_RTOL",
    "DEFAULT_MAX_OFFSET",
    "DEFAULT_MIN_MARGIN",
    "cap_geometry",
    "centerline_branches",
    "centerline_cap_pairs",
]
