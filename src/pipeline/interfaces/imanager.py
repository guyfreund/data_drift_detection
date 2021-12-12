from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import pandas as pd

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector


class IManager(ABC):
    """
    Interface for a manager object
    """

    @abstractmethod
    def manage(self) -> Any:
        raise NotImplementedError

