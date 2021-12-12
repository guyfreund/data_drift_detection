from abc import abstractmethod
import pandas as pd
import os

from src.pipeline.datasets.constants import DatasetType


class Dataset:
    """
    A class that represents a dataset
    """

    def __init__(self, dtype: DatasetType, path: str):
        assert os.path.exists(path)
        self._path = path
        self._raw_df = self.load()
        self._num_instances, self._num_features = self._raw_df.shape
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
    def raw_df(self) -> pd.DataFrame:
        return self._raw_df

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """ loads the dataset from memory

        Returns:
            (pd.DataFrame): the raw dataframe

        """
        raise NotImplementedError
