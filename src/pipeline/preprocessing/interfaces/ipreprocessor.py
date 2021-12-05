from abc import ABC, abstractmethod
from typings import Tuple, List
import pandas as pd

from data_drift_detector.src.data_structures.interfaces import IFeatureMetric

class IPreprocessor(ABC):
  """
  Interface for a preprocessor object
  """

  @abstractmethod
  def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetric]]
    raise NotImplementedError

