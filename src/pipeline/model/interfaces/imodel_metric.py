from abc import ABC, abstractmethod
from typing import Any

from src.pipeline.model.constants import ModelMetricType


class IModelMetric(ABC):
    """
    Interface for a model metric object
    """

    @property
    def value(self) -> Any:
        raise NotImplementedError

    @property
    def mtype(self) -> ModelMetricType:
        raise NotImplementedError
