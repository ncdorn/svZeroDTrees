import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from .builder import *
from .asymmetry import resolve_branch_scaling
from ..treevessel import TreeVessel
from ...utils import *
from ..utils import *
from ...io.blocks import *
from .io import _json_sanitize, to_block_dict
from .results import StructuredTreeResults
from ..compliance import *
from multiprocessing import Pool
from typing import Optional, Literal
import json
import pickle
import math
from functools import partial
import time
from collections import deque

class StructuredTree:
    """
    Structured tree representing microvascular adaptation at the outlet of a 0D model.
    Can be initialized from primitive inputs or from a pre-defined config tree.

    Required inputs:
    - name: a unique identifier for the structured tree
    - time: time vector (list or np.ndarray) used for pressure and flow inputs
    - simparams: simulation parameter object containing physical constants
    """

    def __init__(self,
                 name: str,
                 time: list[float],
                 simparams: SimParams,
                 density: float = 1.055,
                 viscosity: float = 0.04,
                 diameter: float = 0.5,
                 compliance_model: ComplianceModel = None,
                 R: float = None,
                 C: float = None,
                 Pd: float = 0.0,
                 P_in: list[float] = None,
                 Q_in: list[float] = None,
                 tree_config: dict = None,
                 root: TreeVessel = None):
        # --- Required core parameters ---
        self.name = name
        self.time = time

        self.simparams = simparams

        # --- Physical constants ---
        self.viscosity = viscosity
        self.density = density

        # --- Geometry and hemodynamics ---
        self.diameter = diameter
        self._R = R
        self.C = C
        self.Pd = Pd
        self.P_in = P_in
        self.Q_in = Q_in

        # --- Dimensionless reference values ---
        self.q = 10.0    # reference flow [cm³/s]
        self.Lr = 1.0    # reference length [cm]
        self.g = 981.0   # gravity [cm/s²]

        # --- Tree structure ---
        self.generations = 0
        self.root = None

        # --- Compliance model ---
        self.compliance_model = compliance_model if compliance_model else ConstantCompliance(1e5)

        # --- Homeostatic references (populated after compute_homeostatic_state) ---
        self.homeostatic_wss = None
        self.homeostatic_ims = None
        self._homeostatic_wss_map = None
        self._homeostatic_ims_map = None

        if tree_config:
            if root is None:
                raise ValueError("tree_config provided but no root TreeVessel instance passed.")
            self.root = root
            self.block_dict = tree_config
        else:
            self.block_dict = {
                "name": name,
                "initial_d": diameter,
                "P_in": P_in,
                "Q_in": Q_in,
                "boundary_conditions": [],
                "simulation_parameters": {},
                "vessels": [],
                "junctions": [],
                "adaptations": 0
            }


    @classmethod
    def from_outlet_vessel(cls,
                        vessel: Vessel,
                        simparams: SimParams,
                        bc: BoundaryCondition,
                        tree_exists: bool = False,
                        root: TreeVessel = None,
                        P_outlet=0.0,
                        Q_outlet=0.0,
                        time: list = None) -> "StructuredTree":
        """
        Create StructuredTree from a 0D outlet vessel and boundary condition.
        """
        P_outlet, _ = _ensure_list_signal(P_outlet)
        Q_outlet, time = _ensure_list_signal(Q_outlet, time or [0.0, 1.0])

        if "Rp" in bc.values:
            R = bc.values["Rp"] + bc.values["Rd"]
            C = bc.values.get("C", 0.0)
        else:
            R = bc.values["R"]
            C = None

        name = f"OutletTree{vessel.branch}"
        Pd = bc.values.get("Pd", 0.0)

        return cls(
            name=name,
            diameter=vessel.diameter,
            R=R,
            C=C,
            Pd=Pd,
            P_in=P_outlet,
            Q_in=Q_outlet,
            time=time,
            simparams=simparams,
            tree_config=vessel if tree_exists else None,
            root=root if tree_exists else None
        )

    @classmethod
    def from_bc_config(cls,
                    bc: BoundaryCondition,
                    simparams: SimParams,
                    diameter: float,
                    P_outlet=0.0,
                    Q_outlet=0.0,
                    time: list = None) -> "StructuredTree":
        """
        Create StructuredTree from a boundary condition only (no vessel metadata).
        """
        P_outlet, _ = _ensure_list_signal(P_outlet)
        Q_outlet, time = _ensure_list_signal(Q_outlet, time or [0.0, 1.0])

        if "Rp" in bc.values:
            R = bc.values["Rp"] + bc.values["Rd"]
            C = bc.values.get("C", 0.0)
        else:
            R = bc.values["R"]
            C = None

        name = f"OutletTree_{bc.name}"
        Pd = bc.values.get("Pd", 0.0)

        return cls(
            name=name,
            diameter=diameter,
            R=R,
            C=C,
            Pd=Pd,
            P_in=P_outlet,
            Q_in=Q_outlet,
            time=time,
            simparams=simparams
        )

# **** I/O METHODS ****

    def to_dict(self):
        '''
        convert the StructuredTree instance parameters to a dictionary
        '''

        params = {
            "name": self.name,
            "initial_d": self.diameter,
            "compliance": {
                "model": self.compliance_model.description(),
                "params": self.compliance_model.params,
            }
        }
        return params


    def to_json(self, filename, *, strict=True, nonfinite_as=None):
        """
        Write the structured tree to a JSON file with a fully serializable block_dict.

        Parameters
        ----------
        filename : str
            Output path.
        strict : bool
            If True (default), disallow NaN/Inf by converting them to `nonfinite_as`.
        nonfinite_as : Any
            Replacement for NaN/Inf when strict=True (default: None).
            Set to 0.0 if you prefer numeric placeholders.
        """
        sanitized = _json_sanitize(self.block_dict, strict=strict, nonfinite_as=nonfinite_as)
        with open(filename, "w") as f:
            # allow_nan=False enforces strict RFC 8259 JSON (no NaN/Inf literals)
            json.dump(sanitized, f, indent=4, ensure_ascii=False, allow_nan=not strict)

    def to_pickle(self, filename):
        '''
        write the structured tree to a pickle file

        :param filename: name of the pickle file
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def to_block_dict(self):
        return to_block_dict(self.store)

# **** END OF I/O METHODS ****

    def build(self, **build_kwargs):
        # explicit, side-effectful convenience method
        xi = build_kwargs.pop("xi", None)
        eta_sym = build_kwargs.pop("eta_sym", None)
        max_nodes = int(build_kwargs.pop("max_nodes", 200_000))
        alpha, beta = resolve_branch_scaling(
            alpha=build_kwargs.get("alpha"),
            beta=build_kwargs.get("beta"),
            xi=xi,
            eta_sym=eta_sym,
        )
        build_kwargs["alpha"] = alpha
        build_kwargs["beta"] = beta

        # stash the latest scaling inputs for downstream reference
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.eta_sym = eta_sym if eta_sym is not None else (beta / alpha if alpha else None)

        self.max_nodes = max_nodes
        self.store = build_tree_soa(**build_kwargs,
                                    density=self.density,
                                    eta=self.viscosity,
                                    compliance_model=self.compliance_model,
                                    name=self.name,
                                    max_nodes=self.max_nodes)
        self.homeostatic_wss = None
        self.homeostatic_ims = None
        self._homeostatic_wss_map = None
        self._homeostatic_ims_map = None
        return self.store

    def segment_resistances(self) -> np.ndarray:
        """
        Return Poiseuille segment resistances for every vessel currently stored.
        """
        if not hasattr(self, "store") or self.store is None:
            raise RuntimeError("StructuredTree.segment_resistances() requires a built store.")

        st = self.store
        r = 0.5 * np.asarray(st.d, dtype=np.float64)
        L = float(st.lrr) * r
        eta = float(st.eta)

        eps = np.finfo(np.float64).tiny
        r4 = np.maximum(r ** 4, eps)
        return 8.0 * eta * L / (np.pi * r4)

    def equivalent_resistance(self) -> float:
        """
        Compute the equivalent resistance seen at the root of the structured tree.
        """
        if not hasattr(self, "store") or self.store is None:
            raise RuntimeError("StructuredTree.equivalent_resistance() requires a built storage object.")

        st = self.store
        R_seg = self.segment_resistances()
        if R_seg.size == 0:
            return 0.0

        left = np.asarray(st.left, dtype=np.int32)
        right = np.asarray(st.right, dtype=np.int32)
        gen = np.asarray(st.gen, dtype=np.int32)

        order = np.argsort(gen)[::-1]  # deepest generation first
        R_eq = np.array(R_seg, dtype=np.float64)

        for idx in order:
            children = []
            li = int(left[idx])
            ri = int(right[idx])
            if li >= 0:
                children.append(li)
            if ri >= 0:
                children.append(ri)

            if not children:
                continue

            inv_sum = 0.0
            for child_idx in children:
                child_R = float(R_eq[child_idx])
                if child_R <= 0.0:
                    continue
                inv_sum += 1.0 / child_R

            if inv_sum > 0.0:
                R_eq[idx] = float(R_seg[idx]) + 1.0 / inv_sum
            else:
                R_eq[idx] = float(R_seg[idx])

        return float(R_eq[0])


    def reset_tree(self, keep_root=False):
        """
        reset the block dict if you are generating many iterations of the structured tree to optimize the diameter

        :param keep_root: bool to decide whether to keep the root TreeVessel instance
        """
        if keep_root:
            pass
        else:
            self.root = None
        self.block_dict["vessels"] = []
        self.block_dict["junctions"] = []
        self.vesselDlist = []


    def create_block_dict(self):
        '''
        create the block dict from a pre-existing root, 
        for example in the case of adapting the diameter of the vessels
        '''
        self.reset_tree(keep_root=True)
        self.block_dict["vessels"].append(self.root.params)
        queue = [self.root]

        while len(queue) > 0:
            q_id = 0
            current_vessel = queue.pop(q_id)
            # create the block dict for the left vessel
            if not current_vessel.collapsed:
                queue.append(current_vessel.left)
                self.block_dict["vessels"].append(current_vessel.left.params)
                # create the block dict for the right vessel
                queue.append(current_vessel.right)
                self.block_dict["vessels"].append(current_vessel.right.params)


    def compute_olufsen_impedance(
        self,
        n_procs: Optional[int] = None,     # kept for API compatibility (unused)
        tsteps: Optional[int] = None,
        chunk_size: int = 512,
        *,
        # ---- the 3 switches ----
        dc_mode: Literal["segment_only","poiseuille_network"] = "poiseuille_network",
        fft_layout: Literal["legacy_shift","hermitian"] = "legacy_shift",
        leaf_termination: Literal["zero","reflectionless"] = "reflectionless",
    ):
        """
        Vectorized, NaN-safe Olufsen structured-tree impedance with toggles to
        match legacy behavior or use more physical alternatives.

        Requires:
        - self.store: SoA with d, gen, left, right, collapsed, lrr, density, compliance_model
        - self.time (monotonic)
        - self.eta  (dynamic viscosity, g/(cm·s))

        Returns:
        self.Z_t (real time-domain kernel via IFFT), self.time
        """
        st = self.store

        # ---------------- time & frequency grid ----------------
        time = np.asarray(self.time, dtype=np.float64)
        if tsteps is None: tsteps = int(time.size)
        if tsteps < 2:     raise ValueError("Need at least two time samples")

        dt_nominal = float(np.mean(np.diff(time)))
        scale_t = float(self.q) / (float(self.Lr) ** 3) if hasattr(self, "q") and hasattr(self, "Lr") else 1.0
        dt = dt_nominal * scale_t
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Computed dt must be positive/finite, got {dt}")

        # Standard frequency grid (+/-) and its legacy-shifted version
        omega_std   = 2.0 * np.pi * np.fft.fftfreq(tsteps, d=dt)      # ifft expects this order
        omega_legacy = np.fft.fftshift(omega_std)                      # old code built ω this way
        F = tsteps
        pos_stop = F//2 + 1               # number of nonnegative freqs (DC..Nyquist)
        dc_std_idx = int(np.where(omega_std == 0.0)[0][0])             # index of DC in std order (usually 0)

        # ---------------- geometry & material (vectorized) ----------------
        d = st.d.astype(np.float32)
        r = 0.5 * d
        A = (np.pi * r * r).astype(np.float32)
        L = (np.float32(st.lrr) * r).astype(np.float32)

        rho64 = float(st.density)
        eta64 = float(st.eta)
        if rho64 <= 0 or not np.isfinite(rho64): raise ValueError(f"rho must be positive/finite, got {rho64}")
        if eta64 <= 0 or not np.isfinite(eta64): raise ValueError(f"eta must be positive/finite, got {eta64}")

        # compliance model Eh/r (vectorized if possible)
        r64 = r.astype(np.float64)
        try:
            Eh_over_r = np.asarray(st.compliance_model.evaluate(r64), dtype=np.float64)
            if Eh_over_r.shape != r64.shape: raise ValueError
        except Exception:
            Eh_over_r = np.asarray([st.compliance_model.evaluate(float(ri)) for ri in r64], dtype=np.float64)

        A64 = A.astype(np.float64)
        L64 = L.astype(np.float64)
        C64  = (3.0 * A64) / (2.0 * Eh_over_r)       # your scalar code’s definition
        pref_g = np.sqrt((C64 * A64) / rho64)        # appears in g_omega
        pref_c = np.sqrt((A64 / (C64 * rho64)))      # appears in c_omega

        # topology
        gens = st.gen.astype(np.int32)
        max_gen = int(gens.max(initial=0))
        collapsed = st.collapsed.astype(bool)
        left_idx  = st.left.astype(np.int32)
        right_idx = st.right.astype(np.int32)

        # nodes by generation (exclude collapsed in spectral path)
        idx_by_gen = [np.where((gens == g) & (~collapsed))[0] for g in range(max_gen + 1)]
        # absolute index -> row-in-gen map
        n_nodes = d.size
        pos_map_by_gen = []
        for g in range(max_gen + 1):
            idx = idx_by_gen[g]
            mp = np.full(n_nodes, -1, dtype=np.int32)
            if idx.size: mp[idx] = np.arange(idx.size, dtype=np.int32)
            pos_map_by_gen.append(mp)

        EPS = 1e-14

        # ---------------- DC handling (switch) ----------------
        # Poiseuille segment resistance
        r64_safe = np.maximum(r64, np.finfo(np.float64).tiny)
        R_seg = (8.0 * eta64 * L64) / (np.pi * (r64_safe ** 4))

        def dc_root_value() -> float:
            if dc_mode == "segment_only":
                # Match OLD behavior: z(0) = R_eq of the ROOT SEGMENT ONLY
                return float(R_seg[0])
            elif dc_mode == "poiseuille_network":
                # Physically correct network DC: series (add) + parallel (combine) bottom-up
                Z_dc_next, next_idx = None, None
                for g in range(max_gen, -1, -1):
                    idx = idx_by_gen[g]
                    if idx.size == 0:
                        Z_dc_next, next_idx = None, idx
                        continue
                    li = left_idx[idx];  ri = right_idx[idx]
                    hasL = (li >= 0) & (~collapsed[li]);  hasR = (ri >= 0) & (~collapsed[ri])
                    if g == max_gen or Z_dc_next is None or next_idx.size == 0:
                        Z_load = np.zeros(idx.size, dtype=np.float64)
                    else:
                        pos_child = pos_map_by_gen[g + 1]
                        Z1 = np.zeros(idx.size, dtype=np.float64)
                        Z2 = np.zeros(idx.size, dtype=np.float64)
                        if np.any(hasL): Z1[hasL] = Z_dc_next[pos_child[li[hasL]]]
                        if np.any(hasR): Z2[hasR] = Z_dc_next[pos_child[ri[hasR]]]
                        Zsum = Z1 + Z2
                        Z_load = np.zeros_like(Z1)
                        onlyL = hasL & (~hasR); onlyR = hasR & (~hasL); both = hasL & hasR
                        if np.any(onlyL): Z_load[onlyL] = Z1[onlyL]
                        if np.any(onlyR): Z_load[onlyR] = Z2[onlyR]
                        if np.any(both):
                            num = (Z1[both] * Z2[both]); den = Zsum[both]
                            zero = (np.abs(den) < EPS) & (np.abs(num) < EPS)
                            safe = ~zero
                            Zb = np.zeros_like(num); Zb[safe] = num[safe] / den[safe]
                            Z_load[both] = Zb
                    Z_dc_next = R_seg[idx] + Z_load
                    next_idx = idx
                root_row = 0 if next_idx.size == 1 else int(np.where(next_idx == 0)[0][0])
                return float(Z_dc_next[root_row])
            else:
                raise ValueError(f"Unknown dc_mode: {dc_mode}")

        Z_dc_root = dc_root_value()

        # ---------------- robust Womersley + term ----------------
        def womersley(r_vec: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
            r_in = np.asarray(r_vec, dtype=np.float64)
            w_in = np.asarray(w_vec, dtype=np.float64)
            arg = (np.abs(w_in)[None, :] * rho64) / eta64
            arg = np.maximum(arg, 0.0)
            out = r_in[:, None] * np.sqrt(arg, dtype=np.float64)
            out[~np.isfinite(out)] = 0.0
            return out

        inv_sqrt_i = 1.0 / np.sqrt(1j)  # complex128

        def form_term(wom: np.ndarray) -> np.ndarray:
            wom = wom.astype(np.float64)
            term = np.zeros_like(wom, dtype=np.complex128)
            m0  = wom == 0.0
            m1  = wom > 3.0
            m23 = (wom > 2.0) & (~m1)
            m2  = (~m1) & (~m23) & (~m0)
            if np.any(m1) or np.any(m23):
                t_inv = (2.0 * inv_sqrt_i) / wom
                B = 1.0 + (1.0 / (2.0 * wom))
                S2 = 1.0 - t_inv * B
            if np.any(m2) or np.any(m23):
                S1 = 1j * (wom ** 2) / 8.0 + (wom ** 4) / 48.0
            if np.any(m1):  term[m1]  = np.sqrt(S2[m1])
            if np.any(m2):  term[m2]  = np.sqrt(S1[m2])
            if np.any(m23): term[m23] = ((3.0 - wom[m23]) * np.sqrt(S1[m23])
                                    + (wom[m23] - 2.0) * np.sqrt(S2[m23]))
            mag = np.abs(term); tiny = mag < EPS
            if np.any(tiny): term[tiny] = (term[tiny] / (mag[tiny] + 0.0)) * EPS
            return term

        # ---------------- inner solver for arbitrary positive ω vector ----------------
        def solve_root_for_pos_omegas(w_pos: np.ndarray) -> np.ndarray:
            """Return Z_root(ω) for ω in w_pos (ω>=0). Shape [len(w_pos)]."""
            if w_pos.size == 0:
                return np.empty(0, dtype=np.complex64)

            out = np.empty(w_pos.size, dtype=np.complex64)
            # chunk to cap memory
            for s in range(0, w_pos.size, chunk_size):
                e = min(s + chunk_size, w_pos.size)
                w = w_pos[s:e].astype(np.float64)     # >0
                Fc = w.size

                wom = womersley(r64, w)
                term = form_term(wom)
                g_omega = (pref_g[:, None] * term).astype(np.complex128)   # [n,Fc]
                c_omega = (pref_c[:, None] * term).astype(np.complex128)   # [n,Fc]
                # clamp tiny |g| & |c|
                gm = np.abs(g_omega); mask = gm < EPS
                if np.any(mask): g_omega[mask] *= (EPS / (gm[mask] + 0.0))
                cm = np.abs(c_omega); mask = cm < EPS
                if np.any(mask): c_omega[mask] *= (EPS / (cm[mask] + 0.0))

                kappa = (w[None, :] * L64[:, None]) / c_omega              # [n,Fc]

                Z_next, next_idx = None, None
                for g in range(max_gen, -1, -1):
                    idx = idx_by_gen[g]
                    if idx.size == 0:
                        Z_next, next_idx = None, idx
                        continue

                    li = left_idx[idx];  ri = right_idx[idx]
                    hasL = (li >= 0) & (~collapsed[li])
                    hasR = (ri >= 0) & (~collapsed[ri])

                    if g == max_gen or Z_next is None or next_idx.size == 0:
                        if leaf_termination == "zero":
                            ZL = np.zeros((idx.size, Fc), dtype=np.complex128)
                        elif leaf_termination == "reflectionless":
                            # reflectionless continuation: Z_char = 1/g
                            gk_leaf = g_omega[idx, :]
                            ZL = 1.0 / gk_leaf
                        else:
                            raise ValueError(f"Unknown leaf_termination: {leaf_termination}")
                    else:
                        pos_child = pos_map_by_gen[g + 1]
                        Z1 = np.zeros((idx.size, Fc), dtype=np.complex128)
                        Z2 = np.zeros((idx.size, Fc), dtype=np.complex128)
                        if np.any(hasL): Z1[hasL, :] = Z_next[pos_child[li[hasL]], :]
                        if np.any(hasR): Z2[hasR, :] = Z_next[pos_child[ri[hasR]], :]

                        Zsum = Z1 + Z2
                        ZL = np.zeros_like(Z1)
                        onlyL = hasL & (~hasR); onlyR = hasR & (~hasL); both = hasL & hasR
                        if np.any(onlyL): ZL[onlyL, :] = Z1[onlyL, :]
                        if np.any(onlyR): ZL[onlyR, :] = Z2[onlyR, :]
                        if np.any(both):
                            num = (Z1[both, :] * Z2[both, :]); den = Zsum[both, :]
                            zero = (np.abs(den) < EPS) & (np.abs(num) < EPS)
                            safe = ~zero
                            Zb = np.zeros_like(num); Zb[safe] = num[safe] / den[safe]
                            ZL[both, :] = Zb

                    sin_k = np.sin(kappa[idx, :])
                    cos_k = np.cos(kappa[idx, :])
                    gk = g_omega[idx, :]

                    den = (cos_k + 1j * gk * ZL * sin_k)
                    num = (1j * sin_k / gk + cos_k * ZL)

                    small_den = (np.abs(den) < EPS)
                    Z_seg = np.empty_like(den)
                    Z_seg[~small_den] = num[~small_den] / den[~small_den]
                    Z_seg[small_den] = (1j * np.tan(kappa[idx, :][small_den]) / gk[small_den])

                    Z_next = Z_seg
                    next_idx = idx

                root_row = 0 if next_idx.size == 1 else int(np.where(next_idx == 0)[0][0])
                out[s:e] = Z_next[root_row, :].astype(np.complex64)

            return out

        # ---------------- assemble spectrum according to fft_layout (switch) ----------------
        if fft_layout == "hermitian":
            # Positive nonnegative ω in std order: [0 .. Nyquist]
            w_pos = omega_std[:pos_stop]
            Z_pos = np.empty(pos_stop, dtype=np.complex64)
            # DC
            Z_pos[0] = np.complex64(Z_dc_root)
            # ω>0
            if pos_stop > 1:
                Z_pos[1:] = solve_root_for_pos_omegas(w_pos[1:])
            # Build full Hermitian spectrum for a real ifft
            Z_full = np.empty(F, dtype=np.complex64)
            Z_full[:pos_stop] = Z_pos
            if F > 2:
                Z_full[pos_stop:] = np.conjugate(Z_pos[1:pos_stop-1][::-1])

        elif fft_layout == "legacy_shift":
            # Build on fftshifted frequencies: first half is [-fs/2, ..., -df, 0] (length pos_stop)
            w_first_half = np.abs(omega_legacy[:pos_stop])
            Z_first = np.empty(pos_stop, dtype=np.complex64)

            # DC at the last slot of the first half
            Z_first[-1] = np.complex64(Z_dc_root)

            # ω>0 entries for the rest of the first half
            if pos_stop > 1:
                Z_first[:-1] = solve_root_for_pos_omegas(w_first_half[:-1])

            # Fill the shifted spectrum:
            #   first half (negative freqs + DC) is the conjugate of the positives we just computed
            Z_legacy = np.zeros(F, dtype=np.complex64)
            Z_legacy[:pos_stop] = np.conjugate(Z_first)

            # Second half (positive freqs) is the conjugate mirror of the interior of the first half
            if F % 2 == 0:
                # even N: exclude Nyquist (index 0 of first half) AND DC (last index)
                mirror_src = Z_legacy[1:pos_stop-1]
            else:
                # odd N: exclude DC only (no Nyquist bin)
                mirror_src = Z_legacy[:pos_stop-1]

            Z_legacy[pos_stop:] = np.conjugate(mirror_src[::-1])

            # Shift back to standard order for IFFT
            Z_full = np.fft.ifftshift(Z_legacy)

        else:
            raise ValueError(f"Unknown fft_layout: {fft_layout}")

        # ---------------- sanity check + IFFT ----------------
        if not np.all(np.isfinite(Z_full)):
            bad = ~np.isfinite(Z_full)
            first_bad = int(np.argmax(bad))
            raise FloatingPointError(
                f"Non-finite Z_full at index {first_bad}, ω={omega_std[first_bad]:.3e}. "
                f"min|Z|={np.nanmin(np.abs(Z_full)):.3e}, max|Z|={np.nanmax(np.abs(Z_full)):.3e}"
            )

        print(f"Z(w=0) = {Z_full[dc_std_idx]}")
        self.Z_t = np.fft.ifft(Z_full).real.astype(np.float64)
        return self.Z_t, self.time


    def create_impedance_bc(self, name, tree_id, Pd: float = 0.0):
        '''
        create an impedance BC object
        
        :param name: name of the boundary condition
        :param tree_id: id of the tree in the list of trees
        :param Pd: distal pressure in dyn/cm2'''

        print(f'creating impedance bc for tree {self.name}')

        impedance_bc = BoundaryCondition({
            "bc_name": f"{name}",
            "bc_type": "IMPEDANCE",
            "bc_values": {
                "tree": tree_id,
                "Z": self.Z_t.tolist(),
                "t": self.time,
                "Pd": Pd,
            }
        })

        return impedance_bc
    
    def create_resistance_bc(self, name, Pd: float = 0.0):
        '''
        create a resistance bc from the tree using the trees root equivalent resistance'''

        self.Pd = Pd

        resistance_bc = BoundaryCondition({
            "bc_name": f"{name}",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": self.equivalent_resistance(),
                "Pd": Pd,
            }
        })

        return resistance_bc
    
    def compute_homeostatic_state(self, Q):
        """
        Simulate the structured tree under a reference flow Q and cache
        the corresponding wall shear stress and intramural stress profiles.
        """

        if not hasattr(self, "store") or self.store is None:
            raise RuntimeError("StructuredTree must be built before computing the homeostatic state.")

        if Q is None:
            raise ValueError("Flow Q must be provided to compute the homeostatic state.")

        print("computing homeostatic state for structured tree...")
        q_val = float(Q)
        Pd_val = float(self.Pd) if self.Pd is not None else 0.0
        result_df = self.simulate(Q_in=[q_val, q_val], Pd=Pd_val)

        if not hasattr(self, "results") or self.results is None:
            raise RuntimeError("StructuredTree simulation did not populate results; cannot compute homeostatic state.")

        tau_ts = self.results.wss_timeseries()
        tau_mean = np.mean(tau_ts, axis=1)

        pressure_mean = np.mean(self.results.pressure_in, axis=1)
        radii = 0.5 * np.asarray(self.store.d, dtype=np.float64)
        # Legacy TreeVessel models assume wall thickness is r/10.
        thickness = np.maximum(radii / 10.0, 1e-9)
        ims_mean = pressure_mean * radii / thickness

        self.homeostatic_wss = tau_mean
        self.homeostatic_ims = ims_mean

        ids = np.asarray(self.store.ids, dtype=np.int32)
        self._homeostatic_wss_map = {int(vid): float(tau_mean[idx]) for idx, vid in enumerate(ids)}
        self._homeostatic_ims_map = {int(vid): float(ims_mean[idx]) for idx, vid in enumerate(ids)}

        return result_df

    def match_RCR_to_impedance(self):
        '''
        find the RCR parameters to match the impedance from an impedance tree.'''

        # get the impedance of the structured tree if self.Z_t is None
        if self.Z_t is None:
            self.compute_olufsen_impedance()

        
        # loss function for optimizing RCR parameters
        def loss_function(params):
            '''
            loss function for optimizing RCR parameters

            :param params: RCR parameters [Rp, C, Rd]
            '''
            # compute impedance from the RCR parameters
            Rp, C, Rd = params
            # calculate the impedance from the RCR parameters
            tsteps = len(self.time)
            period = max(self.time) * self.q / self.Lr**3
            df = 1 / period
            omega = [i * df * 2 * np.pi for i in range(-tsteps//2, tsteps//2)] # angular frequency vector

            Z_om = np.zeros(len(omega), dtype=complex)

            # Z_om[:tsteps//2+1] = np.conjugate([((1j * w * Rp * Rd * C) + (Rp + Rd)) / ((1j * w * Rd * C) + 1) for w in omega[:tsteps//2+1]])
            # def Z(w, Rp, C, Rd):
            #     return ((1j * w * Rp * Rd * C) + (Rp + Rd)) / ((1j * w * Rd * C) + 1)
            Z_om[:tsteps//2+1] = np.array([np.sqrt(((Rd + Rp) ** 2 + (w * Rp * Rd * C) ** 2) / (1 + (w * Rd * C) ** 2)) for w in omega[:tsteps//2+1]])

            # apply self-adjoint property of the impedance
            Z_om_half = Z_om[:tsteps//2]
            # add negative frequencies
            Z_om[tsteps//2+1:] = np.conjugate(np.flipud(Z_om_half[:-1]))


            # dimensionalize omega
            omega = [w * self.q / self.Lr**3 for w in omega]

            Z_om = np.fft.ifftshift(Z_om)

            print(f'Z(w=0) = {Z_om[0]}')

            Z_rcr = np.fft.ifft(Z_om)

            self.Z_rcr = np.real(Z_rcr)

            # calculate the squared difference between the impedance from the RCR parameters and the impedance from the structured tree
            loss = np.sum((self.Z_t - Z_rcr)**2)

            print(f'loss: {loss}')

            return loss
        
        # initial guess for RCR parameters
        initial_guess = [100.0, 0.0001, 900.0]

        # optimize the RCR parameters
        bounds = Bounds(lb=[0.0, 0.0, 0.0], ub=[np.inf, np.inf, np.inf])
        # bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf], ub=[np.inf, np.inf, np.inf])
        result = minimize(loss_function, initial_guess, method='Nelder-Mead', bounds=bounds)

        print(f'optimized RCR parameters: {result.x}')

        return result.x


    def adapt_constant_wss(self, Q, Q_new, method='cwss', n_iter=1):
        R_old = self.root.R_eq  # calculate pre-adaptation resistance

        def constant_wss(d, Q=Q, Q_new=Q_new):
            '''
            function for recursive algorithm to update the vessel diameter based on constant wall shear stress assumption

            :param d: diameter of the vessel
            :param Q: original flowrate through the vessel
            :param Q_new: post-operative flowrate through the model
            
            :return: length of the updated diameter
            '''
            # adapt the diameter of the vessel based on the constant shear stress assumption

            return (Q_new / Q) ** (1 / 3) * d
        
        def constant_wss_ims(d, Q=Q, Q_new=Q_new):
            '''
            update the diameter based on constant wall shear stress and intramural stress'''

            # dDdt = K_tau_d * ()

            pass

        def update_diameter(vessel, update_func):
            '''
            preorder traversal to update the diameters of all the vessels in the tree  
            
            :param vessel: TreeVessel instance
            :param update_func: function to update vessel diameter based on constant wall shear stress asssumption
            '''

            if vessel:
                # recursive step
                update_diameter(vessel.left, update_func)
                update_diameter(vessel.right, update_func)

                vessel.d = update_func(vessel.d)

        # recursive step
        for i in range(n_iter):
            print(f'performing cwss adaptation iteration {i}')
            self.initial_d = constant_wss(self.initial_d, Q=Q, Q_new=Q_new)
            update_diameter(self.root, constant_wss)

        self.create_block_dict()

        R_new = self.root.R_eq  # calculate post-adaptation resistance

        return R_old, R_new


    def adapt_wss_ims(self, Q, Q_new, n_iter=100):
        '''
        adapt the diameter of the structured tree based on the flowrate through the model

        :param Q: original flowrate through the vessel
        :param Q_new: post-operative flowrate through the model
        :param method: adaptation method to use
        :param n_iter: number of iterations to perform

        :return: pre-adaptation and post-adaptation resistance
        '''

        preop_result = self.simulate(Q_in=[Q, Q], Pd=self.Pd)

        assign_flow_to_root(preop_result, self.root)

        postop_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)

        def adapt_vessel(vessel):
            '''
            recursive step to adapt the vessel diameter'''

            if vessel:
                # get flow from postop result
                vessel_name = f"branch{vessel.id}_seg0"
                Q_new = get_branch_result(postop_result, 'flow_in', vessel_name)
                P_new = get_branch_result(postop_result, 'pressure_in', vessel_name)
                # adapt the diameter based on the flow
                vessel.adapt_cwss_ims(Q_new, P_new, n_iter=n_iter, verbose=False)
                # recursive step
                adapt_vessel(vessel.left)
                adapt_vessel(vessel.right)
        
        print(f"adapting tree diameter with Q = {Q} Q_new = {Q_new}")

        adapt_vessel(self.root)

        adapted_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
        assign_flow_to_root(adapted_result, self.root)


    def adapt_wss_ims_method2(self, Q, Q_new, n_iter=100):
        '''
        adapt the diameter of the structured tree based on the flowrate through the model using a different method

        :param Q: original flowrate through the vessel
        :param Q_new: post-operative flowrate through the model
        :param n_iter: number of iterations to perform

        :return: pre-adaptation and post-adaptation resistance
        '''
        print(f"running preop tree simulation with Q = {Q} and Pd = {self.Pd}")
        preop_result = self.simulate(Q_in=[Q, Q], Pd=self.Pd)
        assign_flow_to_root(preop_result, self.root)

        print(f"running postop tree simulation with Q = {Q_new} and Pd = {self.Pd}")
        iteration_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)

        def adapt_vessel(vessel):
            '''
            recursive step to adapt the vessel diameter'''

            if vessel:
                # get flow from postop result
                vessel_name = f"branch{vessel.id}_seg0"
                Q_new = get_branch_result(iteration_result, 'flow_in', vessel_name)
                P_new = get_branch_result(iteration_result, 'pressure_in', vessel_name)
                # adapt the diameter based on the flow
                vessel.adapt_cwss_ims(Q_new, P_new, n_iter=1)
                # recursive step
                adapt_vessel(vessel.left)
                adapt_vessel(vessel.right)
        
        print(f"adapting tree diameter with Q = {Q} Q_new = {Q_new}")

        for i in range(n_iter):
            print(f"adapting {self.count_vessels()} vessel diameters for tree {self.name}...")
            adapt_vessel(self.root)
            # after adapting the vessel diameters, we need to simulate the tree again to get the new flow and pressure values
            iteration_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
            # assign_flow_to_root(iteration_result, self.root)

            print(f"adaptation iteration {i+1} of {n_iter}, root diameter = {self.root.d}, root resistance = {self.root.R_eq}")
        


        adapted_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
        assign_flow_to_root(adapted_result, self.root)


    def optimize_tree_diameter(self, resistance=None, d_min=0.01, alpha=0.9, beta=0.6, lrr=10.0):
        """ 
        Use Nelder-Mead to optimize the diameter and number of vessels with respect to the desired resistance
        
        :param resistance: resistance value to optimize against
        :param log_file: optional path to log file
        :param d_min: minimum diameter of the vessels
        :param pries_secomb: True if the pries and secomb model is used to adapt the vessels, so pries and secomb integration
            is performed at every optimization iteration
        """
        
        # write_to_log(log_file, "Optimizing tree diameter for resistance " + str(self.params["bc_values"]["R"]) + " with d_min = " + str(d_min) + "...")
        # initial guess is oulet r
        d_guess = self.diameter / 2

        # get the resistance if it is RCR or Resistance BC
        if resistance is None:
            if "Rp" in self.params["bc_values"].keys():
                R0 = self.params["bc_values"]["Rp"] + self.params["bc_values"]["Rd"]
            else:
                R0 = self.params["bc_values"]["R"]
        else:
            R0 = resistance

        # define the objective function to be minimized
        def r_min_objective(diameter, d_min, R0, alpha, beta, lrr):
            '''
            objective function for optimization

            :param diameter: inlet diameter of the structured tree

            :return: squared difference between target resistance and built tree resistance
            '''
            # build tree
            self.build(initial_d = diameter[0],
                        d_min = d_min,
                        alpha = alpha,
                        beta = beta,
                        lrr = lrr)


            # get equivalent resistance
            R = self.root.R_eq

            # calculate squared relative difference
            loss = ((R0 - R) / R0) ** 2

            print(f"d = {diameter[0]}, R = {R}, loss = {loss}")
            
            return loss

        # define optimization bound (lower bound = r_min, which is the termination diameter)
        bounds = Bounds(lb=0.005)

        # perform Nelder-Mead optimization
        d_final = minimize(r_min_objective,
                           d_guess,
                           args=(d_min, R0, alpha, beta, lrr),
                           options={"disp": True},
                           method='Nelder-Mead',
                           bounds=bounds)
        
        R_final = self.root.R_eq

        print(f"There are {self.store.n_nodes()} vessels in the tree")


        return d_final.x, R_final
    

    def add_hemodynamics_from_outlet(self, Q_outlet, P_outlet):
        '''
        add hemodynamics from the outlet of the 0D model to the structured tree
        
        :param Q_outlet: flow at the outlet of the 0D model
        :param P_outlet: pressure at the outlet of the 0D model
        '''

        # make the array length 2 for steady state bc
        if len(Q_outlet) == 1:
            Q_outlet = [Q_outlet[0],] * 2
        
        if len(P_outlet) == 1:
            P_outlet = [P_outlet[0],] * 2

        # add the flow and pressure values to the structured tree
        self.params["Q_in"] = Q_outlet
        self.params["P_in"] = P_outlet

        # this is redundant but whatever
        self.block_dict["Q_in"] = Q_outlet
        self.block_dict["P_in"] = P_outlet
    

    def optimize_alpha_beta(self, Resistance=5.0, log_file=None):
        """ 
        use constrained optimization to optimize the diameter, alpha and beta values of the tree
        
        :param Resistance: resistance value to optimize against
        :param log_file: optional path to log file
        """

        def r_min_objective(params):
            '''
            objective function for optimization

            :param radius: inlet radius of the structured tree

            :return: squared difference between target resistance and built tree resistance
            '''

            # build structured tree
            self.build_tree(params[0], optimizing=True, alpha=params[1], beta=params[2])
            
            # get the equivalent resistance
            R = self.root.R_eq
            
            # calculate squared difference to minimize
            R_diff = (Resistance - R) ** 2

            return R_diff

        # initial guess is outlet r and alpha, beta values from literature
        r_guess = self.initialD / 2
        params_guess = np.array([r_guess, 0.9, 0.6]) # r, alpha, beta initial guess

        # define optimization constraints
        param_constraints = LinearConstraint([[0, 0, 0], [0, 1, 1], [0, 1, -1.5]], [0.0, 1, 0], [np.inf, np.inf, 0])
        param_bounds = Bounds(lb=[0.049, 0, 0], ub=[np.inf, 1, 1], keep_feasible=True)

        # optimization step: use trust-constr since the optimization is constrained
        r_final = minimize(r_min_objective,
                           params_guess,
                           options={"disp": True},
                           method='trust-constr',
                           constraints=param_constraints,
                           bounds=param_bounds)
        
        R_final = self.root.R_eq

        # write the optimization results to log file
        write_to_log(log_file, "     Resistance after optimization is " + str(R_final) + "\n")
        write_to_log(log_file, "     the optimized radius is " + str(r_final.x[0]) + "\n")
        write_to_log(log_file, "     the optimized alpha value is " + str(r_final.x[1])  + "\n")
        write_to_log(log_file, "     the optimized alpha value is " + str(r_final.x[2])  + "\n")

        return r_final.x[0], R_final


    def create_bcs(self):
        ''''
        create the inflow and distal pressure BCs. This function will prepare a block_dict to be run by svzerodplus
        '''
        self.block_dict["boundary_conditions"] = [] # erase the previous boundary conditions
        timesteps = len(self.Q_in) # identify the number of timesteps in the flow boundary condition

        self.block_dict["boundary_conditions"].append(
            {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": self.Q_in,
                        "t": np.linspace(0.0, 1.0, num=timesteps).tolist()
                    }
                },
        )

        for vessel_config in self.block_dict["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    self.block_dict["boundary_conditions"].append(
                        {
                        "bc_name": "P_d" + str(vessel_config["vessel_id"]),
                        "bc_type": "PRESSURE",
                        "bc_values": {
                            "P": [self.Pd,] * 2,
                            "t": [0.0, 1.0]
                            }
                        }
                    )


    def count_vessels(self):
        '''
            count the number vessels in the tree
        '''
        return len(self.store.ids)


    def plot_stiffness(self, path='stiffness_plot.png'):
        '''
        plot the value of Eh/r from d_root to d_min for the tree stiffness value
        '''

        d = np.linspace(self.initial_d, self.d_min, 100)
        Eh_r = np.zeros(len(d))

        # compute Eh/r for each d
        for i in range(len(d)):
            Eh_r[i] = self.compliance_model.evaluate(d[i] / 2)

        if path is None:
            return d, Eh_r
        else:
            plt.figure()
            plt.plot(d, Eh_r)
            plt.yscale('log')
            plt.xlabel('diameter (cm)')
            plt.ylabel('Eh/r (mmHg)')
            plt.title('Eh/r vs. diameter')
            plt.savefig(path)

        
    @property
    def R(self):
        '''
        :return: the equivalent resistance of the tree

        tree.root.R_eq may work better in most cases since that is a value rather than a method
        '''
        if self.root is not None:
            self._R = self.root.R_eq
        return self._R
    

    def simulate(self, 
                 Q_in: list = [1.0, 1.0],
                 Pd: float = 1.0,
                 number_of_cardiac_cycles=1,
                 number_of_time_pts_per_cardiac_cycle = 100,
                 json_path=None
                 ):
        '''
        simulate the structured tree

        :param Q_in: flow at the inlet of the tree
        :param P_d: pressure at the distal outlets of the tree
        ** simulation parameters: **
        :param density: density of the blood [g/cm3]
        :param number_of_cardiac_cycles: number of cardiac cycles to simulate
        :param number_of_time_pts_per_cardiac_cycle: number of time points per cardiac cycle
        :param viscosity: viscosity of the blood [g/cm/s]

        :return result: result dictionary from svzerodsolver
        '''

        self.block_dict = self.to_block_dict()

        if self.simparams is None:
            self.simparams = SimParams({
                "density": self.density,
                "model_name": self.name,
                "number_of_cardiac_cycles": number_of_cardiac_cycles,
                "number_of_time_pts_per_cardiac_cycle": number_of_time_pts_per_cardiac_cycle,
                "viscosity": self.viscosity
            })
            
        self.block_dict["simulation_parameters"] = self.simparams.to_dict()

        self.block_dict["initial_d"] = float(max(self.store.d))

        self.Q_in = Q_in

        self.Pd = Pd
        
        # create solver config from StructuredTree and get tree flow result
        self.create_bcs()

        # result = run_svzerodplus(self.block_dict)

        if json_path is not None:
            self.to_json(json_path)

        result = pysvzerod.simulate(self.block_dict)

        # assigning flow values to the TreeVessel instances
        print("attaching result to tree")
        self.attach_results_from_solver(result)

        # assign flow result to TreeVessel instances to allow for visualization, adaptation, etc.
        # currently this conflicts with adaptation computation, where we do not want to assign flow to root every time we simulate
        # assign_flow_to_root(result, self.root)

        return result
    
    def attach_results_from_solver(self, df_like):
        """
        `df_like` is the table you showed: columns name,time,flow_in,flow_out,pressure_in,pressure_out
        """
        # build index maps
        names = {}                       # vessel_id -> name
        for i in range(self.store.n_nodes()):
            vid = int(self.store.ids[i])
            names[vid] = self.store.name if i == 0 else f"{self.store.name}_seg{vid}"

        # sort vessels by ids to get stable (N,T) layout
        vids = np.array(sorted(names.keys()), dtype=np.int32)
        ordered_names = [names[int(v)] for v in vids]

        # collect solver signals in one pass
        if df_like.empty:
            raise ValueError("Structured tree solver results are empty.")

        fields = ("flow_in", "flow_out", "pressure_in", "pressure_out")
        df_sorted = df_like.sort_values(["name", "time"])
        wide = df_sorted.set_index(["name", "time"])[list(fields)].unstack(level="time")
        wide = wide.sort_index(axis=1, level=1)

        missing = [name for name in ordered_names if name not in wide.index]
        if missing:
            raise ValueError(f"Missing solver results for vessels: {missing[:5]}")
        wide = wide.loc[ordered_names]

        time_index = wide.columns.levels[1]
        t = time_index.to_numpy(dtype=np.float64)

        def extract(field: str) -> np.ndarray:
            frame = wide[field]
            frame = frame.reindex(columns=time_index, copy=False)
            return frame.to_numpy(dtype=np.float64)

        store_ids = np.asarray(self.store.ids, dtype=np.int32)
        idx_lookup = {int(vid): idx for idx, vid in enumerate(store_ids)}
        order = [idx_lookup[int(v)] for v in vids]
        gen = np.asarray(self.store.gen, dtype=np.int32)[order]
        d = np.asarray(self.store.d, dtype=np.float64)[order]

        res = StructuredTreeResults(
            time=t,
            vessel_ids=vids,
            names=names,
            gen=gen,
            d=d,
            eta=float(self.store.eta),
            rho=float(self.store.density),
            flow_in=extract("flow_in"),
            flow_out=extract("flow_out"),
            pressure_in=extract("pressure_in"),
            pressure_out=extract("pressure_out"),
        )
        self.results = res

    def enumerate_vessels(self, start_idx=0):
        """Return a deterministic DFS ordering and stamp each vessel with .idx."""

        vessel_order = []
        def _dfs(v):
            if v is None:
                return
            v.idx = len(vessel_order) + start_idx       # store once, forever
            vessel_order.append(v)
            _dfs(v.left)
            _dfs(v.right)
        
        _dfs(self.root)
        return vessel_order


from dataclasses import dataclass

@dataclass
class TreeVesselView:
    store: StructuredTreeStorage
    i: int

    # read-only-ish properties (writes are just array assignments)
    @property
    def id(self): return int(self.store.ids[self.i])

    @property
    def gen(self): return int(self.store.gen[self.i])

    @property
    def d(self): return float(self.store.d[self.i])

    @d.setter
    def d(self, val: float):
        self.store.d[self.i] = val  # updates SoA in-place

    @property
    def collapsed(self): return bool(self.store.collapsed[self.i])

    @property
    def left(self):
        j = int(self.store.left[self.i])
        return None if j < 0 else TreeVesselView(self.store, j)

    @property
    def right(self):
        j = int(self.store.right[self.i])
        return None if j < 0 else TreeVesselView(self.store, j)

    @property
    def parent(self):
        j = int(self.store.parent[self.i])
        return None if j < 0 else TreeVesselView(self.store, j)
    
