from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd

from src.pipeline.preprocessing.interfaces.ifeature_metric import IFeatureMetric
from src.pipeline.datasets.dataset import Dataset

class IPreprocessor(ABC):
  """
  Interface for a preprocessor object
  """

  @abstractmethod
  def preprocess(self, dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetric]]:
    raise NotImplementedError
