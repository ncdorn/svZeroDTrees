# Structured Tree Performance Overhaul

**Objective:** make structured-tree building and downstream impedance computation fast and memory-efficient while preserving physical fidelity and legacy reproducibility.

---

## Goals

- Eliminate Python/OO overhead during tree construction.
- Keep outputs (block dict, junction topology, impedance) consistent with the legacy path.
- Make post-processing (e.g., WSS by generation/diameter, waveforms) cheap and vectorized.

---

## Core Architectural Changes

### 1) Structure-of-Arrays (SoA) storage
- Introduced `StructuredTreeStorage` with contiguous NumPy arrays:
  - `ids, gen, d, left, right, collapsed, …`
- Replaced recursive `TreeVessel` growth with an **iterative BFS** over indices.
  - No deep Python call stack.
  - No per-node object allocation during build.

### 2) Preallocation + lightweight bookkeeping
- Preallocate or grow arrays in **large chunks**; write indices directly.
- Vessel/junction records are produced from SoA in a single pass rather than driving control flow with Python dicts.

### 3) Removal of premature pruning (fixes missing BCs)
- Deleted the branch that pruned a parent when **both children** were `< d_min>`.
- We now **always create both children**; collapsed children:
  - are **not enqueued**,
  - **do** get a vessel record and outlet BC immediately.

Example of what was removed:
```python
# ❌ removed: caused missing outlet BC on final collapsed left child
if left_d < d_min and right_d < d_min:
    continue
```

### 4) Block-dict generation: vectorized and spec-compliant
- `to_block_dict` computes per-vessel 0D parameters in a vectorized pass:
  - \( R = \frac{8 \mu L}{\pi r^4} \),
  - \( C = \frac{3A}{2(Eh/r)} \cdot L \),
  - \( L = \frac{\rho L}{A} \).
- Emits the exact schema required:
```json
{
  "vessel_id": 1,
  "vessel_length": 10.0,
  "vessel_name": "branch1_seg0",
  "zero_d_element_type": "BloodVessel",
  "zero_d_element_values": {
    "R_poiseuille": 1.0,
    "C": 0.0,
    "L": 0.0,
    "stenosis_coefficient": 0.0
  },
  "boundary_conditions": {
    "inlet": "INFLOW"        // root only
    // or
    // "outlet": "P_d{vessel_id}"  // collapsed leaves
  }
}
```

### 5) JSON-safe serialization
- `to_json` now **sanitizes** `np.float32/64`, `np.int*`, `np.ndarray`, and non-finite values.
- Ensures strict JSON (RFC-compliant) output without crashes on large trees.

---

## Numerical & Impedance Pipeline Improvements

### 6) Vectorized Olufsen impedance
- Replaced per-frequency/per-node Python calls with **batch vector ops across ω**.
- Constructed the Hermitian spectrum explicitly; fixed conjugate-mirror **off-by-one** before `ifft`.

### 7) Robustness guards (no NaNs/Infs)
- Small-epsilon floors for \( r, A, Eh/r \) to avoid division by ~0.
- Stabilized Womersley-dependent branches (guarding `sqrt` inputs) to eliminate
  ```
  RuntimeWarning: invalid value encountered in sqrt
  ```
- Final `np.nan_to_num` to prevent NaNs from entering the FFT.

### 8) Legacy-compatibility switches
- Optional flags (“legacy shift”, etc.) reproduce legacy time-domain impedance for regression while the default path uses the numerically cleaner construction.

---

## Results Handling for Fast Post-Processing

### 9) Centralized results store (`StructuredTreeResults`)
- All solver outputs stored as contiguous arrays **(N_vessels, T)**:
  - `flow_in/out`, `pressure_in/out`, shared `time`.
- Vectorized derived fields and aggregations:
  - **WSS**: \( \tau(t) = \frac{4 \mu Q(t)}{\pi r^3} \) (inlet or outlet flow).
  - Means by **generation** and **diameter bins**.
- Lightweight vessel objects (if any) reference the central store; only small summaries or lazy views live on objects.

---

## Performance Implications

- **Time:** Removing recursion & object creation plus SoA/vector ops → tree build is **linear in nodes** with low Python overhead; impedance assembly is batched across frequencies.
- **Memory:** Few large contiguous arrays (often `float32` where safe) instead of thousands of Python objects/dicts → **much lower RAM** and better cache locality.
- **Stability:** Guards eliminate sporadic NaNs that previously poisoned `ifft` and yielded NaN time-domain traces.

---

## API & Maintenance Benefits

- Clear separation of concerns:
  - `storage.py`: geometry/topology, `to_block_dict`.
  - `results.py`: time-series & analytics (WSS, aggregations).
  - `builder.py` / `impedance.py`: construction & physics kernels.
- Easier unit testing:
  - Topology correctness,
  - R/C/L values,
  - Spectral symmetry & time-domain parity,
  - WSS and generation/diameter aggregations.

---

## Validation

- Overlaid **legacy vs new** time-domain impedance:
  - Differences confined to small regions; attributable to corrected Hermitian indexing and numerical regularization.
  - With legacy switches enabled, curves **match historical output** closely.

---

## Bottom Line

By (i) moving to **SoA** with **iterative** construction, (ii) **vectorizing** physics kernels and impedance assembly, (iii) adding **numerical guards**, and (iv) **centralizing results**, the pipeline shifted from recursion-heavy, object-dense code to a **cache-friendly, vectorized** architecture—delivering large speedups and lower memory usage without sacrificing physical fidelity.
