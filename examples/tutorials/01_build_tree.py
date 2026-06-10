from __future__ import annotations

import json
from pathlib import Path

from svzerodtrees import StructuredTree
from svzerodtrees.io.blocks.simulation_parameters import SimParams
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance


def main() -> None:
    tree = StructuredTree(
        name="demo_tree",
        time=[0.0, 0.5, 1.0],
        simparams=SimParams({}),
        compliance_model=ConstantCompliance(6.6e4),
    )
    tree.build(
        initial_d=0.30,
        d_min=0.01,
        lrr=10.0,
        alpha=0.9,
        beta=0.6,
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "vessel_count": tree.count_vessels(),
        "equivalent_resistance": tree.equivalent_resistance(),
        "tree_metadata": tree.to_dict(),
    }
    output_path = output_dir / "build_tree_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {output_path}")
    print(f"vessel_count={summary['vessel_count']}")
    print(f"equivalent_resistance={summary['equivalent_resistance']:.6f}")


if __name__ == "__main__":
    main()
