from __future__ import annotations

from pathlib import Path

from svzerodtrees import ConfigHandler, StructuredTree
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "examples" / "construct_tree" / "simple_config.json"
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigHandler.from_json(str(config_path), is_pulmonary=False)

    tree = StructuredTree(
        name="demo_tree_bc",
        time=config.inflows["INFLOW"].t,
        simparams=config.simparams,
        compliance_model=ConstantCompliance(6.6e4),
    )
    tree.build(
        initial_d=0.30,
        d_min=0.01,
        lrr=10.0,
        alpha=0.9,
        beta=0.6,
    )

    config.bcs["BC"] = tree.create_resistance_bc("BC", Pd=0.0)
    config.tree_params[tree.name] = tree.to_dict()

    output_path = output_dir / "simple_config_with_tree_resistance.json"
    config.to_json(str(output_path))

    print(f"wrote {output_path}")
    print(f"updated_outlet_resistance={config.bcs['BC'].values['R']:.6f}")


if __name__ == "__main__":
    main()
