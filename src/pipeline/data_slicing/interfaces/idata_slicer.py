from abc import ABS, abstractmethod

class IDataSlicer(ABC):
  """
  Interface for a data slicer object
  """

  @abstractmethod
  def slice(self)
    raise NotImplementedError

