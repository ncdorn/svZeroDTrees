from .storage import StructuredTreeStorage
import numpy as np
from collections import deque
import warnings


def build_tree_soa(initial_d: float,
                    d_min: float,
                    alpha: float,
                    beta: float,
                    lrr: float,
                    density: float,
                    eta: float,
                    compliance_model,
                    name: str,
                    max_nodes: int = 500_000) -> StructuredTreeStorage:
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
        truncated = False

        while q:
            remaining = max_nodes - len(ids)
            if remaining <= 0:
                truncated = True
                break
            i = q.popleft()
            di = d[i]
            gi = gen[i]

            li = ri = -1

            # left
            if remaining >= 1:
                next_id += 1
                ld = alpha * di
                li = next_id
                ids.append(li); gen.append(gi + 1); d.append(ld)
                parent.append(i); left.append(-1); right.append(-1)
                lc = ld < d_min
                collapsed.append(lc)
                if not lc: q.append(li)
            else:
                truncated = True

            remaining = max_nodes - len(ids)
            # right
            if remaining >= 1:
                next_id += 1
                rd = beta * di
                ri = next_id
                ids.append(ri); gen.append(gi + 1); d.append(rd)
                parent.append(i); left.append(-1); right.append(-1)
                rc = rd < d_min
                collapsed.append(rc)
                if not rc: q.append(ri)
            else:
                truncated = True

            # set children's indices on parent row
            left[i]  = li
            right[i] = ri

        if truncated:
            warnings.warn(
                f"Structured tree '{name}' reached max_nodes={max_nodes}; "
                "remaining branches were truncated."
            )

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
            eta=float(eta),
            compliance_model=compliance_model,
            name=name,
        )
