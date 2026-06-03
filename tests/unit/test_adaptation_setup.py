from __future__ import annotations

import svzerodtrees.adaptation.setup as adaptation_setup_module
from svzerodtrees.microvasculature.compliance import OlufsenCompliance
from svzerodtrees.microvasculature.treeparams import TreeParameters
from svzerodtrees.tune_bcs.clinical_targets import ClinicalTargets


def test_create_preop_model_preserves_tuned_tree_compliance(monkeypatch):
    class FakeTree:
        def __init__(self, compliance_model):
            self.compliance_model = compliance_model

        def count_vessels(self):
            return 1

    class FakePreopPA:
        def __init__(self):
            self.lpa_tree = None
            self.rpa_tree = None

        def create_steady_trees(self, lpa_params, rpa_params, *, max_nodes=100_000):
            self.max_nodes = max_nodes
            self.lpa_tree = FakeTree(lpa_params.compliance_model)
            self.rpa_tree = FakeTree(rpa_params.compliance_model)

    fake_preop_pa = FakePreopPA()
    monkeypatch.setattr(adaptation_setup_module.ConfigHandler, "from_json", lambda path, is_pulmonary=True: object())
    monkeypatch.setattr(adaptation_setup_module.PAConfig, "from_pa_config", lambda *_args, **_kwargs: fake_preop_pa)

    clinical_targets = ClinicalTargets(
        mpa_p=[30.0, 15.0, 22.0],
        lpa_p=[25.0, 12.0, 18.0],
        rpa_p=[25.0, 12.0, 18.0],
        q=5.0,
        rpa_split=0.55,
        wedge_p=12.0,
    )
    lpa_model = OlufsenCompliance(k1=11.0, k2=-22.0, k3=33.0)
    rpa_model = OlufsenCompliance(k1=44.0, k2=-55.0, k3=66.0)
    lpa_params = TreeParameters(
        name="lpa",
        lrr=2.0,
        diameter=0.5,
        d_min=0.1,
        alpha=0.9,
        beta=0.6,
        compliance_model=lpa_model,
    )
    rpa_params = TreeParameters(
        name="rpa",
        lrr=3.0,
        diameter=0.6,
        d_min=0.2,
        alpha=0.8,
        beta=0.5,
        compliance_model=rpa_model,
    )

    result = adaptation_setup_module.create_preop_model(
        "preop.json",
        clinical_targets,
        lpa_params,
        rpa_params,
        max_nodes=200_000,
    )

    assert result is fake_preop_pa
    assert result.lpa_tree.compliance_model is lpa_model
    assert result.rpa_tree.compliance_model is rpa_model
    assert result.max_nodes == 200_000
    assert result.source_config_path == "preop.json"
