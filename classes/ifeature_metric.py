from abc import ABS, abstractmethod

class IFeatureMetric(ABC):
  """
  Interface for a feature metric object
  """

  @property
  @abstractmethod
  def name(self) -> str
    raise NotImplementedError

  @property
  @abstractmethod
  def dataset(self) -> str
    raise NotImplementedError

  @property
  @abstractmethod
  def feature_type(self) -> str
    raise NotImplementedError

  @property
  @abstractmethod
  def number_of_nulls(self) -> int
    raise NotImplementedError

  @property
  @abstractmethod
  def mean(self) -> float
    raise NotImplementedError

  @property
  @abstractmethod
  def variance(self) -> float
    raise NotImplementedError

  @property
  @abstractmethod
  def is_important_feature(self) -> bool
    raise NotImplementedError
