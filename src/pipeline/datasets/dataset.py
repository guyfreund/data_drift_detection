from abc import abstractmethod
import pandas as pd
from typing import List
import os

from src.pipeline.datasets.constants import DatasetType


class Dataset:
    """
    A class that represents a dataset
    """
    def __init__(self, dtype: DatasetType, path: str, numeric_features: List[str], categorical_features: List[str], is_drifted: bool = False):
        assert os.path.exists(path)
        self._path = path
        self._raw_df = self.load()
        self._num_instances, self._num_features = self._raw_df.shape
        self._dtype = dtype
        self._drifted_flag_= is_drifted
        self._numeric_features = numeric_features # TODO implement
        self._categorical_features = categorical_features # TODO implement

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
    def numeric_features(self):
        return self._numeric_features

    @property
    def categorical_features(self):
        return self._categorical_features

    @property
    def raw_df(self) -> pd.DataFrame:
        return self._raw_df

    @abstractmethod  # TODO: might be not static as using self.path to read the file..
    def load(self) -> pd.DataFrame:
        """ loads the dataset from memory

        Returns:
            (pd.DataFrame): the raw dataframe

        """
        raise NotImplementedError

    @raw_df.setter
    def raw_df(self, value):
        self._raw_df = value



