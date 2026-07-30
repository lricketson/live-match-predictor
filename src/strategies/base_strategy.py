from abc import ABC, abstractmethod
import numpy as np


class FeatureStrategy(ABC):
    """
    The contract that all matrix feature modifiers must follow.
    Operates on 12x12 intensity matrices (NumPy ndarrays).
    """

    @abstractmethod
    def apply(self, matrix: np.ndarray, ctx: dict) -> np.ndarray:
        pass
