from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, Tuple

@dataclass
class StructuredTreeResults:
    # coordinates / metadata
    time: np.ndarray                 # (T,)
    vessel_ids: np.ndarray           # (N,)
    names: Dict[int, str]            # vessel_id -> name
    gen: np.ndarray                  # (N,)
    d: np.ndarray                    # (N,) diameters [cm]
    eta: float                       # viscosity [g/(cm·s)]
    rho: float                       # density  [g/cm^3]

    # time-series (N, T) — all contiguous, SoA
    flow_in: np.ndarray              # Q_in
    flow_out: np.ndarray             # Q_out
    pressure_in: np.ndarray          # P_in
    pressure_out: np.ndarray         # P_out

    # lazy caches
    _wss_cache: Optional[np.ndarray] = None  # (N, T)

    @property
    def n_vessels(self): return self.vessel_ids.shape[0]

    @property
    def n_time(self): return self.time.shape[0]

    # ---- Derived fields ----
    def wss_timeseries(self, use_flow: str = "in") -> np.ndarray:
        """
        Compute wall shear stress τ_w(t) per vessel using Poiseuille:
        τ = 4 μ Q / (π r^3). By default uses inlet flow.
        """
        if self._wss_cache is not None and use_flow == "in":
            return self._wss_cache

        eps = np.finfo(np.float64).tiny
        r = 0.5 * self.d[:, None]                         # (N,1)
        Q = self.flow_in if use_flow == "in" else self.flow_out
        tau = 4.0 * self.eta * Q / (np.pi * np.maximum(r**3, eps))  # (N,T)

        if use_flow == "in":
            self._wss_cache = tau
        return tau

    # ---- Slicing helpers ----
    def ts(self, vessel: Optional[int] = None, name: Optional[str] = None,
           field: str = "pressure_in") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (time, y(t)) for a single vessel by id or name.
        field ∈ {'flow_in','flow_out','pressure_in','pressure_out','wss'}.
        """
        if (vessel is None) == (name is None):
            raise ValueError("Provide exactly one of vessel or name")

        if name is not None:
            # invert names dict once if you’ll call this a lot
            inv = getattr(self, "_name_to_id", None)
            if inv is None:
                inv = {vname: vid for vid, vname in self.names.items()}
                self._name_to_id = inv
            vessel = inv[name]

        idx = int(np.where(self.vessel_ids == vessel)[0][0])

        if field == "wss":
            y = self.wss_timeseries()[idx]
        else:
            y = getattr(self, field)[idx]
        return self.time, y

    # ---- Group/aggregate examples ----
    def mean_by_generation(self, field: str = "wss") -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (gens, mean_over_vessels_at_gen(t)) as (G,), (G,T).
        """
        vals = self.wss_timeseries() if field == "wss" else getattr(self, field)
        gens, inv = np.unique(self.gen, return_inverse=True)
        G, T = gens.size, self.n_time
        out = np.zeros((G, T), dtype=np.float64)
        counts = np.bincount(inv)
        for g_idx in range(G):
            sel = (inv == g_idx)
            out[g_idx] = vals[sel].mean(axis=0) if counts[g_idx] else 0.0
        return gens, out

    def bin_by_diameter(self, bins: np.ndarray, field: str = "wss") -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin vessels by diameter and average time-series in each bin.
        Returns (bin_centers, (B, T) array).
        """
        vals = self.wss_timeseries() if field == "wss" else getattr(self, field)
        which = np.digitize(self.d, bins) - 1  # [0..B-1]
        B, T = len(bins) - 1, self.n_time
        out = np.zeros((B, T), dtype=np.float64)
        for b in range(B):
            sel = (which == b)
            out[b] = vals[sel].mean(axis=0) if np.any(sel) else 0.0
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, out