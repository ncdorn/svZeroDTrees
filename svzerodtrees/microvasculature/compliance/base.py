from abc import ABC, abstractmethod

class ComplianceModel(ABC):
    @abstractmethod
    def evaluate(self, r: float) -> float:
        """Return compliance as a function of radius r."""
        pass

    def description(self) -> str:
        return self.__class__.__name__