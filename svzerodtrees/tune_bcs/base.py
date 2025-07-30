class BoundaryConditionTuner:
    def __init__(self, config_handler, mesh_surfaces_path, clinical_targets, **kwargs):
        self.config_handler = config_handler
        self.mesh_surfaces_path = mesh_surfaces_path
        self.clinical_targets = clinical_targets
        self.kwargs = kwargs

    def tune(self):
        raise NotImplementedError("Each tuner must implement the tune() method.")