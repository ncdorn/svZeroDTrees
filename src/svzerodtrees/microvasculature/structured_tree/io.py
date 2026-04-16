from .storage import StructuredTreeStorage
import numpy as np
import math

def to_block_dict(store: StructuredTreeStorage) -> dict:
    """
    Build a block dictionary with per-vessel 0D parameters.

    Required fields on `store`:
      - ids:          (N,) int
      - gen:          (N,) int
      - d:            (N,) float  (diameter, cm)
      - left, right:  (N,) int    (child vessel ids or -1)
      - collapsed:    (N,) bool
      - name:         str         (root name prefix)
      - lrr:          float       (length-to-radius ratio)
      - density:      float       (rho, g/cm^3)
      - eta:          float       (dynamic viscosity, g/(cm·s))
      - compliance_model: object with evaluate(r) or evaluate_vectorized(r)
                           returning Eh/r (pressure·length), consistent with your z0 code
    """

    n = int(store.n_nodes())
    ids = np.asarray(store.ids, dtype=np.int32)
    d   = np.asarray(store.d,   dtype=np.float64)
    r   = 0.5 * d                              # radius [cm]
    L_v = store.lrr * r                        # length [cm]
    A   = np.pi * r * r                        # cross-sectional area [cm^2]
    rho = float(store.density)                 # [g/cm^3]
    eta = float(store.eta)                     # [g/(cm·s)]

    # --- Eh/r (vectorized if possible) ---
    if hasattr(store.compliance_model, "evaluate_vectorized"):
        Eh_over_r = np.asarray(store.compliance_model.evaluate_vectorized(r), dtype=np.float64)
    else:
        # fallback: scalar loop (still fast for typical N)
        Eh_over_r = np.array([float(store.compliance_model.evaluate(float(ri))) for ri in r], dtype=np.float64)

    # Numerical safety: avoid divide-by-zero on pathological radii
    eps = np.finfo(np.float64).tiny
    r4  = np.maximum(r**4, eps)
    A_  = np.maximum(A,   eps)
    Eh_over_r_ = np.maximum(Eh_over_r, eps)

    # --- 0D element values (R, C, L) ---
    # Poiseuille resistance [g/(cm^4·s)] * cm^3? (consistent with your existing units)
    R_poiseuille = 8.0 * eta * L_v / (np.pi * r4)

    # Volume compliance of the segment:
    # C' (per unit length) = 3*A/(2*(Eh/r))  → C_total = C' * L
    C_total = (3.0 * A / (2.0 * Eh_over_r_)) * L_v

    # Inductance of the segment: L = ρ * L / A
    L_inert = rho * L_v / A_

    # Clean up any residual inf/nan caused by extreme inputs
    R_poiseuille = np.nan_to_num(R_poiseuille, nan=0.0, posinf=0.0, neginf=0.0)
    C_total      = np.nan_to_num(C_total,      nan=0.0, posinf=0.0, neginf=0.0)
    L_inert      = np.nan_to_num(L_inert,      nan=0.0, posinf=0.0, neginf=0.0)

    # --- Vessels block ---
    vessels = []
    for i in range(n):
        vid = int(ids[i])
        vname = store.name if i == 0 else f"{store.name}_seg{vid}"

        v = {
            "vessel_id": vid,
            "vessel_length": float(L_v[i]),
            "vessel_name": vname,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(R_poiseuille[i]),
                "C": float(C_total[i]),
                "L": float(L_inert[i]),
                "stenosis_coefficient": 0.0,
            },
        }

        # Keep existing inlet BC for the root (if you use it elsewhere)
        if i == 0:
            v.setdefault("boundary_conditions", {})
            v["boundary_conditions"]["inlet"] = "INFLOW"

        # Add outlet BC for collapsed terminals
        if bool(store.collapsed[i]):
            v.setdefault("boundary_conditions", {})
            v["boundary_conditions"]["outlet"] = f"P_d{vid}"

        vessels.append(v)

    # --- Junctions block (unchanged shape, but using vessel ids) ---
    junctions = []
    j = 0
    for i in range(n):
        li = int(store.left[i]); ri = int(store.right[i])
        if li >= 0 or ri >= 0:
            junctions.append({
                "junction_name": f"J{j}",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [int(ids[i])],
                "outlet_vessels": [x for x in (li, ri) if x >= 0],
            })
            j += 1

    return {"vessels": vessels, "junctions": junctions}

def _json_sanitize(obj, *, strict=True, nonfinite_as=None):
        """
        Recursively convert obj into JSON-serializable primitives.
        - NumPy scalars -> Python int/float/bool
        - NumPy arrays  -> lists
        - Non-finite floats (NaN/Inf) -> `nonfinite_as` (default None) if strict
        """
        # Dict
        if isinstance(obj, dict):
            return {str(k): _json_sanitize(v, strict=strict, nonfinite_as=nonfinite_as)
                    for k, v in obj.items()}

        # Sequence types
        if isinstance(obj, (list, tuple, set)):
            return [_json_sanitize(v, strict=strict, nonfinite_as=nonfinite_as) for v in obj]

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return _json_sanitize(obj.tolist(), strict=strict, nonfinite_as=nonfinite_as)

        # NumPy scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            x = float(obj)
            if strict and not math.isfinite(x):
                return nonfinite_as
            return x
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        # Plain Python numerics
        if isinstance(obj, (int, bool)):
            return obj
        if isinstance(obj, float):
            if strict and not math.isfinite(obj):
                return nonfinite_as
            return obj

        # None/str pass-through
        if obj is None or isinstance(obj, str):
            return obj

        # Fallbacks: try to_dict / __dict__ / str
        if hasattr(obj, "to_dict"):
            return _json_sanitize(obj.to_dict(), strict=strict, nonfinite_as=nonfinite_as)
        if hasattr(obj, "__dict__") and obj.__dict__:
            return _json_sanitize(vars(obj), strict=strict, nonfinite_as=nonfinite_as)

        return str(obj)
