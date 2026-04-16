from abc import ABC, abstractmethod
from typing import Dict

class ComplianceModel(ABC):

    def __init__(self):
        self.params: Dict[str, float] = {}

    @abstractmethod
    def evaluate(self, r: float) -> float:
        """Return compliance as a function of radius r."""
        pass

    def description(self) -> str:
        return self.__class__.__name__