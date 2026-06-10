"""Minimal local pyvista compatibility layer for test workflows.

This covers the subset of the API exercised by svzerodtrees tests without
requiring the optional external pyvista dependency in the local workspace.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


class PolyData:
    def __init__(self, points=None):
        if points is None:
            points = np.zeros((0, 3), dtype=float)
        self.points = np.asarray(points, dtype=float)
        self.lines = np.asarray([], dtype=np.int64)
        self.point_data: dict[str, np.ndarray] = {}

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        if self.points.size == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    def sample(self, _mesh):
        return self

    def tube(self, radius=None):
        return self

    def extract_surface(self):
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            "points": self.points.tolist(),
            "lines": self.lines.tolist(),
            "point_data": {
                name: np.asarray(values).tolist()
                for name, values in self.point_data.items()
            },
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


DataSet = PolyData


def read(path: str | Path) -> PolyData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    poly = PolyData(payload.get("points", []))
    poly.lines = np.asarray(payload.get("lines", []), dtype=np.int64)
    poly.point_data = {
        name: np.asarray(values)
        for name, values in payload.get("point_data", {}).items()
    }
    return poly


def Line(start, end) -> PolyData:
    poly = PolyData(np.asarray([start, end], dtype=float))
    poly.lines = np.asarray([2, 0, 1], dtype=np.int64)
    return poly


class Plotter:
    def __init__(self, off_screen=True, window_size=None):
        self.off_screen = off_screen
        self.window_size = window_size
        self.camera_position = (
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    def add_mesh(self, mesh, scalars=None, cmap=None, show_scalar_bar=True, **kwargs):
        return None

    def view_isometric(self):
        self.camera_position = (
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        return None

    def reset_camera_clipping_range(self):
        return None

    def set_background(self, color):
        return None

    def screenshot(self, path: str | Path):
        Path(path).write_bytes(b"")

    def close(self):
        return None
