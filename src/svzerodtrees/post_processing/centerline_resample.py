"""Insert centerline points into branches too short to observe.

Calibration samples a vessel's boundary interface at the cross-section next to
the branch end (``external_upstream`` uses the branch's second sample) and needs
at least three samples per branch; otherwise it falls back to the end sample.
VMTK centerlines can give a short MPA only two points, one on the inflow cap,
whose cross-section misses most of the flow (TST-STAN-2, -9).  Inserting evenly
spaced points into such a branch gives svSlicer more cross-sections without
changing the geometry:

- positions, ``Path`` and other float arrays are interpolated linearly;
- ``CenterlineSectionNormal`` is interpolated and renormalized;
- integer arrays (branch/bifurcation/centerline ids, flags) come from the nearer
  endpoint, and ``GlobalNodeId`` gets new ids after the largest existing one;
- each line cell between two consecutive points of the branch is replaced by a
  chain through the new points.

New points are appended after the original points, so original point ids are
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

NORMAL_ARRAY = "CenterlineSectionNormal"
NODE_ID_ARRAY = "GlobalNodeId"


def _read(path: str | Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"centerline has no points: {path}")
    return polydata


def _segments(polydata: vtk.vtkPolyData) -> list[list[int]]:
    lines = polydata.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    cells = []
    while lines.GetNextCell(ids):
        cells.append([int(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
    return cells


def densify_short_branches(
    centerline_path: str | Path,
    output_path: str | Path,
    *,
    min_points_per_branch: int,
    branch_id_array: str = "BranchId",
) -> dict[str, Any]:
    """Write a centerline whose branches have at least ``min_points_per_branch`` points.

    Returns a summary with ``points_added`` and, per densified branch,
    ``{"before": n, "after": m}``.  Branches already long enough, and single-point
    branches (no segment to subdivide), are left unchanged.
    """
    minimum = int(min_points_per_branch)
    if minimum < 2:
        raise ValueError("min_points_per_branch must be at least 2")
    polydata = _read(centerline_path)
    point_data = polydata.GetPointData()
    if point_data.GetArray(branch_id_array) is None:
        raise ValueError(f"centerline needs point array {branch_id_array!r}")

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    branch_ids = vtk_to_numpy(point_data.GetArray(branch_id_array))
    cells = _segments(polydata)

    # Two-point segments joining consecutive points of one branch.
    branch_segments: dict[int, list[int]] = {}
    for index, cell in enumerate(cells):
        if len(cell) == 2 and branch_ids[cell[0]] == branch_ids[cell[1]]:
            branch_segments.setdefault(int(branch_ids[cell[0]]), []).append(index)

    # (segment index, fractions in (0, 1)) for every segment that gets points.
    insertions: dict[int, list[float]] = {}
    densified: dict[str, dict[str, int]] = {}
    for branch in sorted(int(b) for b in np.unique(branch_ids)):
        count = int(np.count_nonzero(branch_ids == branch))
        segments = branch_segments.get(branch, [])
        if count >= minimum or not segments:
            continue
        lengths = {s: float(np.linalg.norm(points[cells[s][1]] - points[cells[s][0]])) for s in segments}
        extra = {s: 0 for s in segments}
        for _ in range(minimum - count):
            # Next point goes where the resulting spacing would be largest.
            target = max(segments, key=lambda s: lengths[s] / (extra[s] + 1))
            extra[target] += 1
        for segment, n_new in extra.items():
            if n_new:
                insertions[segment] = [k / (n_new + 1) for k in range(1, n_new + 1)]
        densified[str(branch)] = {"before": count, "after": minimum}

    n_original = len(points)
    new_points: list[np.ndarray] = []
    sources: list[tuple[int, int, float]] = []  # (start id, end id, fraction)
    new_cells: list[list[int]] = []
    for index, cell in enumerate(cells):
        fractions = insertions.get(index)
        if not fractions:
            new_cells.append(cell)
            continue
        start, end = cell
        chain = [start]
        for fraction in fractions:
            new_points.append(points[start] + fraction * (points[end] - points[start]))
            sources.append((start, end, fraction))
            chain.append(n_original + len(new_points) - 1)
        chain.append(end)
        new_cells.extend([chain[i], chain[i + 1]] for i in range(len(chain) - 1))

    output = vtk.vtkPolyData()
    all_points = np.vstack([points, np.asarray(new_points)]) if new_points else points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(all_points), deep=True))
    output.SetPoints(vtk_points)
    lines = vtk.vtkCellArray()
    for cell in new_cells:
        lines.InsertNextCell(len(cell))
        for point_id in cell:
            lines.InsertCellPoint(point_id)
    output.SetLines(lines)

    for index in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(index)
        name = array.GetName()
        values = vtk_to_numpy(array)
        if sources:
            starts = np.array([s for s, _, _ in sources])
            ends = np.array([e for _, e, _ in sources])
            fractions = np.array([f for _, _, f in sources])
            if np.issubdtype(values.dtype, np.floating):
                weights = fractions.reshape((-1,) + (1,) * (values.ndim - 1))
                added = values[starts] + weights * (values[ends] - values[starts])
                if name == NORMAL_ARRAY and added.ndim == 2:
                    norms = np.linalg.norm(added, axis=1, keepdims=True)
                    added = np.divide(added, norms, out=np.copy(added), where=norms > 0)
            elif name == NODE_ID_ARRAY:
                added = np.arange(len(sources), dtype=values.dtype) + values.max() + 1
            else:
                added = np.where(
                    (fractions < 0.5).reshape((-1,) + (1,) * (values.ndim - 1)),
                    values[starts],
                    values[ends],
                )
            values = np.concatenate([values, added.astype(values.dtype)])
        new_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=array.GetDataType())
        new_array.SetName(name)
        output.GetPointData().AddArray(new_array)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(output)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write resampled centerline: {output_path}")
    return {
        "source": str(centerline_path),
        "output": str(output_path),
        "min_points_per_branch": minimum,
        "points_added": len(new_points),
        "branches": densified,
    }
