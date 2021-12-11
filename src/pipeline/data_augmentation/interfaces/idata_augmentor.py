from abc import ABC, abstractmethod


class IDataAugmentor(ABC):  # TODO: Implement DataAugmentor per DataDriftType.
    """
    Interface for a data augmentor object
    """

    @abstractmethod
    def augment(self):
        raise NotImplementedError
