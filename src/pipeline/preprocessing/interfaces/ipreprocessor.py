from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd

from src.data_structures.interfaces.ifeature_metric import IFeatureMetric

class IPreprocessor(ABC):
  """
  Interface for a preprocessor object
  """

  @abstractmethod
  def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetric]]:
    raise NotImplementedError

