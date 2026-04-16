import argparse
import sys

from .config import load_config, render_schema
from .api import WORKFLOW_MAP


def _run_from_config(path: str) -> int:
    cfg = load_config(path)
    workflow_cls = WORKFLOW_MAP[cfg.workflow]
    workflow = workflow_cls.from_config(cfg)
    workflow.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="svzerodtrees", description="svzerodtrees YAML-first CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("pipeline", "tune-bcs", "construct-trees", "adapt", "postprocess"):
        p = sub.add_parser(name)
        p.add_argument("config", help="Path to YAML config")

    sub.add_parser("schema", help="Print YAML schema template")

    args = parser.parse_args()

    if args.cmd == "schema":
        sys.stdout.write(render_schema())
        return 0

    cmd_map = {
        "pipeline": "pipeline",
        "tune-bcs": "tune_bcs",
        "construct-trees": "construct_trees",
        "adapt": "adapt",
        "postprocess": "postprocess",
    }

    cfg = load_config(args.config)
    expected = cmd_map[args.cmd]
    if cfg.workflow != expected:
        raise ValueError(f"Config workflow '{cfg.workflow}' does not match subcommand '{args.cmd}'")

    workflow_cls = WORKFLOW_MAP[cfg.workflow]
    workflow = workflow_cls.from_config(cfg)
    workflow.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
