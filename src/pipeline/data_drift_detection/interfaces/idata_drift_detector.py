from abc import ABC, abstractmethod

from src.pipeline.data_drift_detection.data_drift import DataDrift


class IDataDriftDetector(ABC):
    """
    Interface for a data drift detectors object
    """

    @abstractmethod
    def detect(self) -> DataDrift:
        raise NotImplementedError
