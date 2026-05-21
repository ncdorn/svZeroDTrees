import importlib


def test_config_import_does_not_trigger_package_cycle():
    module = importlib.import_module("svzerodtrees.config")
    assert hasattr(module, "load_config")


def test_public_package_exports_import_cleanly():
    from svzerodtrees.adaptation import MicrovascularAdaptor, run_structured_tree_adaptation
    from svzerodtrees.microvasculature import StructuredTree, TreeParameters, compliance
    from svzerodtrees.post_processing import compute_pulmonary_resistance_map
    from svzerodtrees.simulation import Simulation, SimulationDirectory
    from svzerodtrees.tune_bcs import ClinicalTargets, ImpedanceTuner, TuneSpace
    from svzerodtrees.tuning import run_impedance_tuning_for_iteration

    assert MicrovascularAdaptor is not None
    assert run_structured_tree_adaptation is not None
    assert StructuredTree is not None
    assert TreeParameters is not None
    assert compliance is not None
    assert compute_pulmonary_resistance_map is not None
    assert Simulation is not None
    assert SimulationDirectory is not None
    assert ClinicalTargets is not None
    assert ImpedanceTuner is not None
    assert TuneSpace is not None
    assert run_impedance_tuning_for_iteration is not None
