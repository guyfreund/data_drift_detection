from abc import ABC, abstractmethod

class IDataSlicer(ABC):
  """
  Interface for a data slicer object
  """

  @abstractmethod
  def slice(self):
    raise NotImplementedError

