from abc import ABC, abstractmethod

class IDataDriftDetector(ABC):
  """
  Interface for a data drift detector object
  """

  @abstractmethod
  def detect(self) -> bool:
    raise NotImplementedError

