from abc import ABC, abstractmethod
import pandas as pd


class FeatureStrategy(ABC):
    """
    The contract that all matrix feature modifiers must follow.
    """

    @abstractmethod
    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        pass
