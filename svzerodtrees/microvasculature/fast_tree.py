from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass(slots=True)
class StructuredTreeStorage:
    # node fields (SoA)
    ids: np.ndarray         # int32
    gen: np.ndarray         # int16
    d: np.ndarray           # float32
    parent: np.ndarray      # int32, -1 for root
    left: np.ndarray        # int32, -1 if none/collapsed
    right: np.ndarray       # int32, -1 if none/collapsed
    collapsed: np.ndarray   # bool
    # tree-level scalars/defaults
    lrr: float
    density: float
    compliance_model: str | int | object  # whatever you use
    name: str

    def n_nodes(self) -> int:
        return self.ids.size

def build_tree_soa(initial_d: float,
                    d_min: float,
                    alpha: float,
                    beta: float,
                    lrr: float,
                    density: float,
                    compliance_model,
                    name: str) -> StructuredTreeStorage:
        # Upper bound on nodes (loose): worst case full binary until collapse.
        # We grow python lists (amortized O(1)) and pack once.

        ids, gen, d = [], [], []
        parent, left, right = [], [], []
        collapsed = []

        # root
        ids.append(0); gen.append(0); d.append(float(initial_d))
        parent.append(-1); left.append(-1); right.append(-1)
        collapsed.append(False)

        q = deque([0])
        next_id = 0

        while q:
            i = q.popleft()
            di = d[i]
            gi = gen[i]

            # If both children guarantee collapse, prune here
            if di * max(alpha, beta) < d_min:
                continue

            # left
            ld = alpha * di
            li = next_id + 1
            ids.append(li); gen.append(gi + 1); d.append(ld)
            parent.append(i); left.append(-1); right.append(-1)
            lc = ld < d_min
            collapsed.append(lc)

            # right
            rd = beta * di
            ri = next_id + 2
            ids.append(ri); gen.append(gi + 1); d.append(rd)
            parent.append(i); left.append(-1); right.append(-1)
            rc = rd < d_min
            collapsed.append(rc)

            # set children's indices on parent row
            left[i]  = li
            right[i] = ri

            if not lc: q.append(li)
            if not rc: q.append(ri)
            next_id += 2

        # pack to compact arrays
        return StructuredTreeStorage(
            ids=np.asarray(ids, dtype=np.int32),
            gen=np.asarray(gen, dtype=np.int16),
            d=np.asarray(d, dtype=np.float32),
            parent=np.asarray(parent, dtype=np.int32),
            left=np.asarray(left, dtype=np.int32),
            right=np.asarray(right, dtype=np.int32),
            collapsed=np.asarray(collapsed, dtype=np.bool_),
            lrr=float(lrr),
            density=float(density),
            compliance_model=compliance_model,
            name=name,
        )


def to_block_dict(store: StructuredTreeStorage) -> dict:
    vessels = []
    for i in range(store.n_nodes()):
        v = {
            "id": int(store.ids[i]),
            "gen": int(store.gen[i]),
            "d": float(store.d[i]),
            "lrr": float(store.lrr),
            "density": float(store.density),
            "compliance_model": store.compliance_model,
        }
        if i == 0:
            v["name"] = store.name
            v["boundary_conditions"] = {"inlet": "INFLOW"}
        if store.collapsed[i]:
            v["collapsed"] = True
        vessels.append(v)

    junctions = []
    j = 0
    for i in range(store.n_nodes()):
        li = int(store.left[i]); ri = int(store.right[i])
        if li >= 0 or ri >= 0:
            junctions.append({
                "junction_name": f"J{j}",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": [int(store.ids[i])],
                "outlet_vessels": [x for x in (li, ri) if x >= 0],
            })
            j += 1

    return {"vessels": vessels, "junctions": junctions}