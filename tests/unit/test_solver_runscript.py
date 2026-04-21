from pathlib import Path

from svzerodtrees.simulation.input_builders.solver_runscript import SolverRunscript


def test_solver_runscript_writes_stage_local_log_paths_and_working_dir(tmp_path: Path):
    runscript_path = tmp_path / "run_solver.sh"
    runscript = SolverRunscript(str(runscript_path))
    solver_path = "/opt/simvascular/bin/svmultiphysics"

    runscript.write(
        nodes=2,
        procs_per_node=8,
        hours=4,
        memory=12,
        svfsiplus_path=solver_path,
        working_dir=str(tmp_path),
    )

    rendered = runscript_path.read_text(encoding="utf-8")
    assert f"#SBATCH --chdir={tmp_path}" in rendered
    assert f"#SBATCH --output={tmp_path / 'svFlowSolver.o%j'}" in rendered
    assert f"#SBATCH --error={tmp_path / 'svFlowSolver.e%j'}" in rendered
    assert f"cd {tmp_path}" in rendered
    assert f"srun {solver_path} svFSIplus.xml" in rendered
