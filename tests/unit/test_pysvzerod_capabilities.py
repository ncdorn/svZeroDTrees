import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import svzerodtrees._pysvzerod as loader


def _fake_solver(*, module_file: Path, calibrate=True, simulate=True):
    module = SimpleNamespace(
        __file__=str(module_file),
        __version__="2.0",
        __build_identity__={
            "name": "svZeroDSolver",
            "version": "2.0",
            "source_commit": "80eab14a",
        },
    )
    if calibrate:
        module.calibrate = lambda payload: payload
    if simulate:
        module.simulate = lambda payload: payload
    return module


@pytest.fixture(autouse=True)
def clear_solver_cache():
    loader.require_pysvzerod.cache_clear()
    loader.clear_calibration_provenance()
    yield
    loader.require_pysvzerod.cache_clear()
    loader.clear_calibration_provenance()


def test_standard_api_and_digest_provenance_are_machine_readable(
    monkeypatch, tmp_path
):
    module_file = tmp_path / "pysvzerod.so"
    module_bytes = b"pinned solver artifact"
    module_file.write_bytes(module_bytes)
    module = _fake_solver(module_file=module_file)
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    result = loader.require_pysvzerod_api()

    assert result["module_path"] == str(module_file.resolve())
    assert result["module_sha256"] == hashlib.sha256(module_bytes).hexdigest()
    assert result["file_metadata"]["size"] == len(module_bytes)
    assert result["file_metadata"]["readable"] is True
    assert result["version"] == "2.0"
    assert result["build_identity"]["source_commit"] == "80eab14a"


def test_solver_without_capabilities_is_accepted(monkeypatch, tmp_path):
    module_file = tmp_path / "pysvzerod.so"
    module_file.write_bytes(b"solver")
    module = _fake_solver(module_file=module_file)
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    payload = {"vessels": []}
    assert loader.calibrate_pysvzerod(payload) == payload
    assert loader.last_calibration_provenance()["module_sha256"] == hashlib.sha256(
        b"solver"
    ).hexdigest()


def test_missing_standard_callable_reports_resolved_identity(
    monkeypatch, tmp_path
):
    calibrate_called = False

    module_file = tmp_path / "pysvzerod.so"
    module_file.write_bytes(b"solver")

    def calibrate(_payload):
        nonlocal calibrate_called
        calibrate_called = True

    module = _fake_solver(module_file=module_file, simulate=False)
    module.calibrate = calibrate
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="required callable API missing") as exc:
        loader.require_pysvzerod_api()

    message = str(exc.value)
    assert "simulate(config)" in message
    assert str(module_file.resolve()) in message
    assert "module SHA-256" in message
    assert "python3 -m pip install -e ../svZeroDSolver" in message
    assert not calibrate_called


def test_missing_calibrate_reports_resolved_identity(monkeypatch, tmp_path):
    module_file = tmp_path / "pysvzerod.so"
    module_file.write_bytes(b"solver")
    module = _fake_solver(module_file=module_file, calibrate=False)
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="required callable API missing") as exc:
        loader.require_pysvzerod_api()

    message = str(exc.value)
    assert "calibrate(config)" in message
    assert str(module_file.resolve()) in message
    assert "module SHA-256" in message


def test_missing_import_reports_required_api_and_install_hint(monkeypatch):
    def missing_import(_name):
        raise ModuleNotFoundError("No module named 'pysvzerod'", name="pysvzerod")

    monkeypatch.setattr(loader.importlib, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match="Resolved module state") as exc:
        loader.require_pysvzerod_api()

    message = str(exc.value)
    assert "calibrate(config)" in message
    assert "simulate(config)" in message
    assert "python3 -m pip install -e ../svZeroDSolver" in message


def test_provenance_is_not_recorded_when_calibrate_fails(monkeypatch):
    module_file = Path("/tmp/pinned-svZeroDSolver/pysvzerod.so")
    module = _fake_solver(module_file=module_file)
    module.calibrate = lambda _payload: (_ for _ in ()).throw(ValueError("bad input"))
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    with pytest.raises(ValueError, match="bad input"):
        loader.calibrate_pysvzerod({})

    assert loader.last_calibration_provenance() is None
