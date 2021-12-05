from abc import ABS, abstractmethod

class IDataDriftDetector(ABC):
  """
  Interface for a data drift detector object
  """

  @abstractmethod
  def detect(self) -> bool
    raise NotImplementedError

