from abc import ABC

from src.constants import DatasetType

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