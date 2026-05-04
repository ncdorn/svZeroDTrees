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


def test_write_files_preserves_explicit_threed_coupler(tmp_path: Path):
    coupling_path = tmp_path / "svzerod_3Dcoupling.json"
    coupling_path.write_text("explicit", encoding="utf-8")
    calls = []

    class Coupler:
        path = str(coupling_path)
        simparams = SimpleNamespace(external_step_size=None, cardiac_period=None)
        coupling_blocks = {"OUTLET": SimpleNamespace(name="OUTLET")}

        def regenerate_impedance_bcs_for_coupled_timing(self):
            calls.append("regenerate_impedance")

        def to_json(self, path):
            Path(path).write_text("explicit-updated", encoding="utf-8")

    class ZeroDConfig:
        def generate_threed_coupler(self, *_args, **_kwargs):
            calls.append("generate_threed_coupler")
            coupling_path.write_text("regenerated", encoding="utf-8")
            return Coupler(), []

        def generate_inflow_file(self, simdir, period=None, n_tsteps=None):
            calls.append(("generate_inflow_file", period, n_tsteps))
            Path(simdir, "inflow.flow").write_text("flow", encoding="utf-8")

    sim_dir = SimulationDirectory.__new__(SimulationDirectory)
    sim_dir.path = str(tmp_path)
    sim_dir.zerod_config = ZeroDConfig()
    sim_dir.mesh_complete = SimpleNamespace()
    sim_dir.svzerod_3Dcoupling = Coupler()
    sim_dir.svFSIxml = SimpleNamespace(
        is_written=False,
        write=lambda *_args, **kwargs: calls.append(("write_xml", kwargs)),
    )
    sim_dir.solver_runscript = SimpleNamespace(
        is_written=False,
        write=lambda **kwargs: calls.append(("write_runscript", kwargs)),
    )
    sim_dir.convert_to_cm = False
    sim_dir.mesh_scale_factor = 1.0
    sim_dir.explicit_threed_coupler = True
    sim_dir.check_files = lambda verbose=False: calls.append("check_files")

    sim_dir.write_files(
        user_input=False,
        sim_config={
            "n_tsteps": 10,
            "dt": 0.01,
            "inflow_boundary_condition": "dirichlet",
        },
    )

    assert "generate_threed_coupler" not in calls
    assert coupling_path.read_text(encoding="utf-8") == "explicit-updated"
    assert any(call[0] == "generate_inflow_file" for call in calls if isinstance(call, tuple))
