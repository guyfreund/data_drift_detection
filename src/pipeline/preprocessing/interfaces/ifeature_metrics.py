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
        answer = all([
            self.number_of_nulls == other.number_of_nulls,
            self.ftype == other.ftype,
            self.name == other.name
        ])
        if self.mean is not None and other.mean is not None:
            answer = answer and self.mean == other.mean
        if self.variance is not None and other.variance is not None:
            answer = answer and self.variance == other.variance

        return answer
