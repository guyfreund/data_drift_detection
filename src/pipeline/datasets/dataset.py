from abc import abstractmethod
from typing import List
import pandas as pd
import os

from src.pipeline.datasets.constants import DatasetType


class Dataset:
    """
    A class that represents a dataset
    """

    def __init__(self, dtype: DatasetType, path: str, label_column_name: str, categorical_feature_names: List[str],
                 numeric_feature_names: List[str]):
        assert os.path.exists(path)
        self._path = path
        self._raw_df = self.load()
        self._num_instances, self._num_features = self._raw_df.shape
        self._dtype = dtype
        self._label_column_name = label_column_name
        self._categorical_feature_names = categorical_feature_names
        self._numeric_feature_names = numeric_feature_names

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

    @raw_df.setter
    def raw_df(self, value: pd.DataFrame):
        self._raw_df = value

    @property
    def label_column_name(self) -> str:
        return self._label_column_name

    @label_column_name.setter
    def label_column_name(self, value: str):
        self._label_column_name = value

    @property
    def numeric_feature_names(self) -> List[str]:
        return self._numeric_feature_names

    @numeric_feature_names.setter
    def numeric_feature_names(self, value: List[str]):
        self._numeric_feature_names = value

    @property
    def categorical_feature_names(self) -> List[str]:
        return self._categorical_feature_names

    @categorical_feature_names.setter
    def categorical_feature_names(self, value: List[str]):
        self._categorical_feature_names = value

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """ loads the dataset from memory

        Returns:
            (pd.DataFrame): the raw dataframe

        """
        raise NotImplementedError
