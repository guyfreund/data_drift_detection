from abc import ABC

class IFeatureMetric(ABC):
  """
  Interface for a feature metric object
  """

  @property
  def name(self) -> str:
    raise NotImplementedError

  @property
  def dataset_type(self) -> str:
    raise NotImplementedError

  @property
  def feature_type(self) -> str:
    raise NotImplementedError

  @property
  def number_of_nulls(self) -> int:
    raise NotImplementedError

  @property
  def mean(self) -> float:
    raise NotImplementedError

  @property
  def variance(self) -> float:
    raise NotImplementedError

  @property
  def is_important_feature(self) -> bool:
    raise NotImplementedError
