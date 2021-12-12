from abc import ABC, abstractmethod


class IDataGenerator(ABC):  # TODO: Implement DataGenerator per DataDriftType.
    """
    Interface for a data generator object
    """

    @abstractmethod
    def generate(self):
        raise NotImplementedError
