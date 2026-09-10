from pathlib import Path
from types import SimpleNamespace

import pytest

import svzerodtrees._pysvzerod as loader


def _fake_solver(*, capabilities=None):
    module = SimpleNamespace(
        __file__="/tmp/pinned-svZeroDSolver/pysvzerod.so",
        __version__="2.0",
        __build_identity__={
            "name": "svZeroDSolver",
            "version": "2.0",
            "source_commit": "991fa17cfe4d436395f649a2c6be0bbda987c23e",
        },
        capabilities=lambda: capabilities or {},
        calibrate=lambda payload: payload,
    )
    return module


@pytest.fixture(autouse=True)
def clear_solver_cache():
    loader.require_pysvzerod.cache_clear()
    loader.clear_calibration_provenance()
    yield
    loader.require_pysvzerod.cache_clear()
    loader.clear_calibration_provenance()


def test_capabilities_and_provenance_are_machine_readable(monkeypatch):
    module = _fake_solver(
        capabilities={
            "per_block_parameter_selection": True,
        }
    )
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    result = loader.require_calibration_capabilities()

    assert result["module_path"] == str(Path(module.__file__).resolve())
    assert result["version"] == "2.0"
    assert result["build_identity"]["source_commit"].startswith("991fa17")
    assert result["capabilities"] == {
        "per_block_parameter_selection": True,
    }


def test_stale_solver_is_rejected_before_calibrate(monkeypatch):
    calibrate_called = False

    def calibrate(_payload):
        nonlocal calibrate_called
        calibrate_called = True

    module = _fake_solver()
    module.calibrate = calibrate
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="Incompatible pysvzerod calibrator") as exc:
        loader.calibrate_pysvzerod({})

    message = str(exc.value)
    assert module.__file__ in message
    assert "build identity" in message
    assert "python3 -m pip install -e ../svZeroDSolver" in message
    assert not calibrate_called


def test_calibration_dispatch_records_provenance(monkeypatch):
    module = _fake_solver(
        capabilities={
            "per_block_parameter_selection": True,
        }
    )
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    payload = {"vessels": []}
    assert loader.calibrate_pysvzerod(payload) == payload
    assert loader.last_calibration_provenance() == {
        "module_path": str(Path(module.__file__).resolve()),
        "version": "2.0",
        "build_identity": module.__build_identity__,
        "capabilities": {
            "per_block_parameter_selection": True,
        },
    }


def test_provenance_is_not_recorded_when_calibrate_fails(monkeypatch):
    module = _fake_solver(
        capabilities={
            "per_block_parameter_selection": True,
        }
    )
    module.calibrate = lambda _payload: (_ for _ in ()).throw(ValueError("bad input"))
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)

    with pytest.raises(ValueError, match="bad input"):
        loader.calibrate_pysvzerod({})

    assert loader.last_calibration_provenance() is None
