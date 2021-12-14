from abc import ABC, abstractmethod
from typing import Any


class IManager(ABC):
    """
    Interface for a manager object
    """

    @abstractmethod
    def manage(self) -> Any:
        raise NotImplementedError
