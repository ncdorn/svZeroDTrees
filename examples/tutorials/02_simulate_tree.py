from __future__ import annotations

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

    result = tree.simulate(
        Q_in=[80.0, 80.0],
        Pd=12.0 * 1333.2,
        json_path=str(output_dir / "tree_simulation_input.json"),
    )
    result_path = output_dir / "tree_simulation_result.csv"
    result.to_csv(result_path, index=False)

    root_mean_pressure = float(tree.results.pressure_in[0].mean())
    root_mean_flow = float(tree.results.flow_in[0].mean())

    print(f"wrote {result_path}")
    print(f"root_mean_flow={root_mean_flow:.6f}")
    print(f"root_mean_pressure={root_mean_pressure:.6f}")


if __name__ == "__main__":
    main()
