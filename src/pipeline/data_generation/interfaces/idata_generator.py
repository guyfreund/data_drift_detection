from abc import ABC, abstractmethod
from typing import List, Any

from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.datasets.dataset import Dataset


class IDataGenerator(ABC):  # TODO: Implement DataGenerator per DataDriftType.
    """
    Interface for a data generator object
    """

    def generate_normal_samples(self, n_samples: int) -> Any:
        raise NotImplementedError

    def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Any:
        raise NotImplementedError


    @property
    def raw_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def generated_dataset(self, n_samples: int, do_drift: bool = False, seed: int = 0) -> Dataset:
        raise NotImplementedError

    @property
    def data_drift_type(self) -> DataDriftType:
        raise NotImplementedError

    @property
    def generation_percent(self) -> float:
        """
        Returns:
            (float) the proportional percent of the new generated dataset. For a double size generated dataset, use: 200
        """
        raise NotImplementedError

    @generation_percent.setter
    def generation_percent(self, value: float):
        """ sets the proportional percent of the new generated dataset. For a double size generated dataset, use: 200 """
        raise NotImplementedError

    @property
    def features_to_drift(self) -> List[str]:
        """
        Returns:
            (List[str]) the list of features to perform drift on
        """
        raise NotImplementedError

    @features_to_drift.setter
    def features_to_drift(self, value: List[str]):
        """ sets the list of features to perform drift on """
        raise NotImplementedError
