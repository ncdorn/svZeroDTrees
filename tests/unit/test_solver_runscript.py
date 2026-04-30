from pathlib import Path
from types import SimpleNamespace

from svzerodtrees.simulation.simulation_directory import SimulationDirectory
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


def test_solver_runscript_writes_configurable_slurm_options(tmp_path: Path):
    runscript_path = tmp_path / "run_solver.sh"
    runscript = SolverRunscript(str(runscript_path))

    runscript.write(
        nodes=1,
        procs_per_node=2,
        hours=3,
        memory=4,
        svfsiplus_path="/custom/svmultiphysics",
        working_dir=str(tmp_path),
        partition="localpart",
        qos="debug",
    )

    rendered = runscript_path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=localpart" in rendered
    assert "#SBATCH --qos=debug" in rendered
    assert "#SBATCH --nodes=1" in rendered
    assert "#SBATCH --ntasks-per-node=2" in rendered
    assert 'if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ -n "${SLURM_TRES_PER_TASK:-}" ]; then' in rendered
    assert 'unset SLURM_TRES_PER_TASK' in rendered
    assert "srun /custom/svmultiphysics svFSIplus.xml" in rendered


def test_simulation_directory_run_local_invokes_solver(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    sim_dir = SimulationDirectory.__new__(SimulationDirectory)
    sim_dir.path = str(tmp_path)
    sim_dir.svFSIxml = SimpleNamespace(path=str(tmp_path / "svFSIplus.xml"))
    sim_dir.solver_runscript = SimpleNamespace(path=str(tmp_path / "run_solver.sh"))
    monkeypatch.setattr(sim_dir, "check_files", lambda verbose=False: True)
    monkeypatch.setattr("svzerodtrees.simulation.simulation_directory.subprocess.run", fake_run)

    sim_dir.run(
        execution_config={
            "mode": "local",
            "executable": "svmultiphysics",
            "clean_command": None,
        }
    )

    assert calls == [(["svmultiphysics", "svFSIplus.xml"], {"cwd": str(tmp_path), "check": True})]


def test_simulation_directory_run_slurm_submits_batch(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    sim_dir = SimulationDirectory.__new__(SimulationDirectory)
    sim_dir.path = str(tmp_path)
    sim_dir.svFSIxml = SimpleNamespace(path=str(tmp_path / "svFSIplus.xml"))
    sim_dir.solver_runscript = SimpleNamespace(path=str(tmp_path / "run_solver.sh"))
    monkeypatch.setattr(sim_dir, "check_files", lambda verbose=False: True)
    monkeypatch.setattr("svzerodtrees.simulation.simulation_directory.subprocess.run", fake_run)

    sim_dir.run(
        execution_config={
            "mode": "slurm",
            "submit_command": "sbatch",
            "clean_command": None,
        }
    )

    assert calls == [(["sbatch", "run_solver.sh"], {"cwd": str(tmp_path), "check": True})]
