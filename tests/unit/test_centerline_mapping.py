from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svzerodtrees.tune_bcs.centerline_mapping import cap_geometry, centerline_branches
from svzerodtrees.tune_bcs.outlet_mapping import resolve_outlet_cap_mapping


# Inlet branch 0 runs up the z axis to a bifurcation at (0, 0, 1); outlet
# branches 1-3 leave it.  Outlet BCs are serialized in branch order, while the
# caps are named so that sorting by filename gives a different order.
BIFURCATION = np.array([0.0, 0.0, 1.0])
OUTLET_ENDS = {
    1: np.array([-1.0, 0.0, 2.0]),
    2: np.array([1.0, 0.0, 2.0]),
    3: np.array([0.0, 1.0, 2.0]),
}
CAP_RADIUS = 0.1


def _branch_points():
    points = {0: [np.array([0.0, 0.0, z]) for z in np.linspace(0.0, 1.0, 5)]}
    for branch_id, end in OUTLET_ENDS.items():
        points[branch_id] = [BIFURCATION + t * (end - BIFURCATION) for t in np.linspace(0.25, 1.0, 4)]
    return points


def _within_branch_length(points):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:])))


def _write_centerline(path: Path, *, global_node_offset: int | None = None) -> dict[int, int]:
    """Write the centerline and return each outlet branch's terminal node id."""

    xyz, branch_ids, lines = [], [], []
    for branch_id, points in _branch_points().items():
        start = len(xyz)
        xyz.extend(points)
        branch_ids.extend([branch_id] * len(points))
        chain = list(range(start, len(xyz)))
        # Outlet branches share the bifurcation point, the last inlet point.
        lines.append(chain if branch_id == 0 else [4, *chain])

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.array(xyz), deep=True))
    poly.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    for chain in lines:
        cells.InsertNextCell(len(chain))
        for node in chain:
            cells.InsertCellPoint(node)
    poly.SetLines(cells)
    branch_array = numpy_to_vtk(np.array(branch_ids, dtype=np.int32), deep=True)
    branch_array.SetName("BranchId")
    poly.GetPointData().AddArray(branch_array)
    offset = 0 if global_node_offset is None else global_node_offset
    if global_node_offset is not None:
        node_array = numpy_to_vtk(np.arange(len(xyz), dtype=np.int32) + offset, deep=True)
        node_array.SetName("GlobalNodeId")
        poly.GetPointData().AddArray(node_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()

    terminals = {}
    for branch_id, chain in enumerate(lines):
        if branch_id:
            terminals[branch_id] = chain[-1] + offset
    return terminals


def _write_cap(path: Path, center, radius=CAP_RADIUS):
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    ring = [np.asarray(center) + radius * np.array([math.cos(a), math.sin(a), 0.0]) for a in angles]
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.array([center, *ring], dtype=float), deep=True))
    poly.SetPoints(vtk_points)
    triangles = vtk.vtkCellArray()
    for k in range(len(ring)):
        triangles.InsertNextCell(3)
        for node in (0, 1 + k, 1 + (k + 1) % len(ring)):
            triangles.InsertCellPoint(node)
    poly.SetPolys(triangles)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()
    return str(path)


def _config(*, vessel_suffix="", node_ids=None, length_scale=1.0):
    points = _branch_points()
    vessels = [
        {
            "vessel_name": f"branch0_seg0{vessel_suffix}",
            "vessel_length": length_scale * _within_branch_length(points[0]),
            "boundary_conditions": {"inlet": "INFLOW"},
        }
    ]
    for branch_id in OUTLET_ENDS:
        vessel = {
            "vessel_name": f"branch{branch_id}_seg0{vessel_suffix}",
            "vessel_length": length_scale * _within_branch_length(points[branch_id]),
            "boundary_conditions": {"outlet": f"OUT_{branch_id - 1}"},
        }
        if node_ids is not None:
            vessel["centerline_node_ids"] = {"inlet": 4, "outlet": node_ids[branch_id]}
        vessels.append(vessel)
    names = [f"OUT_{k}" for k in range(len(OUTLET_ENDS))]
    return SimpleNamespace(
        _config={
            "boundary_conditions": [
                {"bc_name": "INFLOW", "bc_type": "FLOW"},
                *[{"bc_name": name, "bc_type": "RESISTANCE"} for name in names],
            ],
            "vessels": vessels,
        },
        bcs={"INFLOW": SimpleNamespace(name="INFLOW", type="FLOW")}
        | {name: SimpleNamespace(name=name, type="RESISTANCE") for name in names},
        tree_params={},
        vessel_map={},
    )


# Filename order is lpa_a, lpa_b, rpa; branch order puts rpa on branch 1.
CAP_BRANCHES = {"lpa_a": 2, "lpa_b": 3, "rpa": 1}
EXPECTED_BC = {"lpa_a": "OUT_1", "lpa_b": "OUT_2", "rpa": "OUT_0"}


def _caps(tmp_path: Path, placement=None):
    placement = placement or {stem: OUTLET_ENDS[b] for stem, b in CAP_BRANCHES.items()}
    area = math.pi * CAP_RADIUS**2
    return {
        _write_cap(tmp_path / f"{stem}.vtp", center): area
        for stem, center in sorted(placement.items())
    }


@pytest.fixture
def centerline(tmp_path):
    path = tmp_path / "centerlines.vtp"
    _write_centerline(path)
    return path


def _by_stem(resolved):
    return {record.cap_stem: record.bc_name for record in resolved.records}


def test_cap_geometry_returns_area_weighted_centroid(tmp_path):
    centroid, area = cap_geometry(_write_cap(tmp_path / "cap.vtp", [1.0, 2.0, 3.0], radius=0.5))

    assert centroid == pytest.approx([1.0, 2.0, 3.0])
    # A 16-gon inscribed in the circle.
    assert area == pytest.approx(0.5 * 16 * 0.25 * math.sin(2 * math.pi / 16))


def test_centerline_branches_finds_terminals_and_lengths(centerline):
    endpoints, lengths = centerline_branches(centerline)

    assert sorted(endpoints) == [0, 1, 2, 3]
    for branch_id, end in OUTLET_ENDS.items():
        (node_id, xyz), = endpoints[branch_id]
        assert xyz == pytest.approx(end)
    assert lengths[0] == pytest.approx(1.0)


def test_centerline_mode_pairs_caps_by_geometry_not_filename_order(tmp_path, centerline):
    config = _config()
    caps = _caps(tmp_path)

    resolved = resolve_outlet_cap_mapping(config, caps, mode="centerline", centerline=centerline)
    by_order = resolve_outlet_cap_mapping(config, caps, mode="serialized_cap_order")

    assert resolved.strategy == "centerline"
    assert _by_stem(resolved) == EXPECTED_BC
    assert _by_stem(by_order) != EXPECTED_BC
    geometry = resolved.provenance["geometry"]
    assert geometry["seed_consistency_check"] == "branch_lengths"
    assert geometry["centerline"] == str(centerline)
    assert {item["cap_stem"]: item["branch_id"] for item in geometry["pairs"]} == CAP_BRANCHES
    assert geometry["max_offset_observed"] == pytest.approx(0.0, abs=1e-9)
    json.loads(resolved.to_json())


def test_auto_uses_centerline_after_metadata_and_cap_name_fail(tmp_path, centerline):
    resolved = resolve_outlet_cap_mapping(_config(), _caps(tmp_path), centerline=centerline)

    assert resolved.strategy == "centerline"
    assert _by_stem(resolved) == EXPECTED_BC


def test_auto_without_centerline_points_to_geometric_mapping(tmp_path):
    with pytest.raises(ValueError, match="centerline: no centerline was supplied.*supply the centerline"):
        resolve_outlet_cap_mapping(_config(), _caps(tmp_path))


def test_learned_seed_is_checked_through_centerline_node_ids(tmp_path):
    centerline = tmp_path / "centerlines.vtp"
    terminals = _write_centerline(centerline, global_node_offset=100)
    # Learned seeds fold branch length into junctions, so lengths are not usable.
    config = _config(vessel_suffix="_connectorEL", node_ids=terminals, length_scale=0.0)

    resolved = resolve_outlet_cap_mapping(config, _caps(tmp_path), mode="centerline", centerline=centerline)

    assert _by_stem(resolved) == EXPECTED_BC
    assert resolved.provenance["geometry"]["seed_consistency_check"] == "centerline_node_ids"

    # ConfigHandler drops centerline_node_ids; the serialized seed keeps them.
    handler = _config(vessel_suffix="_connectorEL", length_scale=0.0)
    with pytest.raises(ValueError, match="pass the serialized seed payload"):
        resolve_outlet_cap_mapping(handler, _caps(tmp_path), mode="centerline", centerline=centerline)
    from_payload = resolve_outlet_cap_mapping(
        handler,
        _caps(tmp_path),
        mode="centerline",
        centerline=centerline,
        seed_payload=config._config,
    )
    assert _by_stem(from_payload) == EXPECTED_BC

    terminals[1], terminals[2] = terminals[2], terminals[1]
    swapped = _config(vessel_suffix="_connectorEL", node_ids=terminals, length_scale=0.0)
    with pytest.raises(ValueError, match="do not end on their centerline branch endpoints"):
        resolve_outlet_cap_mapping(swapped, _caps(tmp_path), mode="centerline", centerline=centerline)


def test_rejects_centerline_the_seed_was_not_generated_from(tmp_path, centerline):
    with pytest.raises(ValueError, match="branch lengths do not match the centerline"):
        resolve_outlet_cap_mapping(
            _config(length_scale=1.5), _caps(tmp_path), mode="centerline", centerline=centerline
        )


def test_mm_geometry_is_compared_with_cm_seed_lengths(tmp_path, centerline):
    resolved = resolve_outlet_cap_mapping(
        _config(length_scale=0.1),
        _caps(tmp_path),
        mode="centerline",
        centerline=centerline,
        convert_to_cm=True,
    )

    assert resolved.provenance["geometry"]["length_scale"] == pytest.approx(0.1)


def test_rejects_cap_far_from_every_outlet_endpoint(tmp_path, centerline):
    placement = {stem: OUTLET_ENDS[b] for stem, b in CAP_BRANCHES.items()}
    placement["rpa"] = OUTLET_ENDS[1] + np.array([0.0, 0.0, 0.5])

    with pytest.raises(ValueError, match=r"cap 'rpa' is 5\.\d+ cap radii from the nearest"):
        resolve_outlet_cap_mapping(_config(), _caps(tmp_path, placement), mode="centerline", centerline=centerline)


def test_rejects_two_caps_on_one_endpoint(tmp_path, centerline):
    placement = {"lpa_a": OUTLET_ENDS[2], "lpa_b": OUTLET_ENDS[2], "rpa": OUTLET_ENDS[1]}

    with pytest.raises(ValueError, match="both match branch 2"):
        resolve_outlet_cap_mapping(_config(), _caps(tmp_path, placement), mode="centerline", centerline=centerline)


def test_rejects_ambiguous_cap_between_two_endpoints(tmp_path, centerline):
    placement = {stem: OUTLET_ENDS[b] for stem, b in CAP_BRANCHES.items()}
    placement["lpa_b"] = 0.5 * (OUTLET_ENDS[2] + OUTLET_ENDS[3])

    with pytest.raises(ValueError, match="cap 'lpa_b' is ambiguous"):
        resolve_outlet_cap_mapping(_config(), _caps(tmp_path, placement), mode="centerline", centerline=centerline)


def test_failed_centerline_never_falls_back_to_serialized_order(tmp_path, centerline):
    with pytest.raises(ValueError, match="centerline: 0D branch lengths do not match"):
        resolve_outlet_cap_mapping(
            _config(length_scale=1.5),
            _caps(tmp_path),
            centerline=centerline,
            allow_serialized_fallback=True,
        )


def test_rejects_non_centerline_vessel_names(tmp_path, centerline):
    config = _config()
    config._config["vessels"][0]["vessel_name"] = "mpa"

    with pytest.raises(ValueError, match="centerline-generated vessel names"):
        resolve_outlet_cap_mapping(config, _caps(tmp_path), mode="centerline", centerline=centerline)


def test_centerline_mode_contract(tmp_path, centerline):
    with pytest.raises(ValueError, match="'centerline' requires a centerline"):
        resolve_outlet_cap_mapping(_config(), _caps(tmp_path), mode="centerline")
    with pytest.raises(ValueError, match="used only by outlet_mapping_mode 'auto' or 'centerline'"):
        resolve_outlet_cap_mapping(
            _config(), _caps(tmp_path), mode="serialized_cap_order", centerline=centerline
        )
