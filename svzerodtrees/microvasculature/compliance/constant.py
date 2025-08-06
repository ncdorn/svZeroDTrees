from .base import ComplianceModel

class ConstantCompliance(ComplianceModel):
    def __init__(self, value: float):
        self.value = value
        self.params['Eh/r'] = value

    def evaluate(self, r: float) -> float:
        return self.value