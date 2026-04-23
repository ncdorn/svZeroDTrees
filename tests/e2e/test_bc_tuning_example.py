import os
import shutil
from pathlib import Path

import pytest

from svzerodtrees.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "bc-tuning" / "local_pipeline.yml"


def test_bc_tuning_local_example_config_loads():
    cfg = load_config(str(EXAMPLE_CONFIG))

    assert cfg.workflow == "pipeline"
    assert cfg.threed.execution.mode == "local"
    assert cfg.threed.execution.executable == "svmultiphysics"
    assert cfg.pipeline.optimize_bcs is True
    assert cfg.pipeline.run_threed is True
    assert cfg.pipeline.adapt is False


@pytest.mark.external
@pytest.mark.slow
def test_bc_tuning_local_example_external_smoke(tmp_path):
    if os.environ.get("SVZERODTREES_RUN_EXTERNAL_EXAMPLES") != "1":
        pytest.skip("set SVZERODTREES_RUN_EXTERNAL_EXAMPLES=1 to run external examples")
    if shutil.which("svmultiphysics") is None:
        pytest.skip("svmultiphysics is not available on PATH")

    from svzerodtrees.api import run_from_config_file

    case_dir = tmp_path / "bc-tuning"
    shutil.copytree(REPO_ROOT / "examples" / "bc-tuning", case_dir)
    cfg_path = case_dir / "local_pipeline.yml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            "root: examples/bc-tuning",
            f"root: {case_dir}",
        ),
        encoding="utf-8",
    )

    result = run_from_config_file(str(cfg_path))
    assert result["status"] == "ok"
