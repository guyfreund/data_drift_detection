import pandas as pd
import os

from src.pipeline.datasets.interfaces.idataset import IDataset
from src.pipeline.datasets.constants import DatasetType

class Dataset(IDataset):
  """
  A class that represents a training dataset
  """
  def __init__(self, dtype: DatasetType, path: str):
    assert os.path.exists(path)
    self._path = path
    self._df = self.load(self._path)
    self._num_instances, self._num_features = self._df.shape
    self._dtype = dtype
  
  @property
  def num_features(self) -> int:
    return self._num_features

  @property
  def num_instances(self) -> int:
    return self._num_instances

  @property
  def dtype(self) -> DatasetType:
    return self._dtype

  @property
  def path(self) -> str:
    return self._path

  @property
  def df(self) -> pd.DataFrame:
    return self._df

  def load(self) -> pd.DataFrame:
    return pd.DataFrame({})