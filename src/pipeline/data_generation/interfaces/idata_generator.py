from abc import ABC
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
    def origin_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def synthesizer(self) -> None:
        raise NotImplementedError


