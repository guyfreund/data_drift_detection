from abc import ABC, abstractmethod
import pandas as pd

from src.pipeline.datasets.constants import DatasetType

class IDataset(ABC):
  """
  Interface for a dataset object
  """

  @property
  def num_features(self) -> int:
    raise NotImplementedError

  @property
  def num_instances(self) -> int:
    raise NotImplementedError

  @property
  def dtype(self) -> DatasetType:
    raise NotImplementedError

  @property
  def path(self) -> str:
    raise NotImplementedError

  @property
  def df(self) -> pd.DataFrame:
    raise NotImplementedError

  @abstractmethod
  def load(self):
    """
    loads the dataset from memory
    """
    raise NotImplementedError