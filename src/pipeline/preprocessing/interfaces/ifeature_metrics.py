from abc import ABC
from typing import Optional


class IFeatureMetrics(ABC):
    """
    Interface for a feature metrics object
    """

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dataset_type(self) -> str:
        raise NotImplementedError

    @property
    def ftype(self) -> str:
        raise NotImplementedError

    @property
    def number_of_nulls(self) -> int:
        raise NotImplementedError

    @property
    def mean(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def variance(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def is_important_feature(self) -> bool:
        raise NotImplementedError

    def __eq__(self, other: 'IFeatureMetrics'):
        assert all([
            self.mean == other.mean,
            self.variance == other.variance,
            self.number_of_nulls == other.number_of_nulls,
            self.feature_type == other.feature_type,
            self.name == other.name
        ])