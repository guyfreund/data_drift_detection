from abc import abstractmethod
from typing import List, Type, Set, Optional
import pandas as pd
import os

from src.pipeline.datasets.constants import DatasetType


class Dataset:
    """
    A class that represents a dataset
    """

    def __init__(self, dtype: DatasetType, path: str, label_column_name: str, categorical_feature_names: List[str],
                 numeric_feature_names: List[str], to_load: bool = True, raw_df: Optional[pd.DataFrame] = None):
        assert os.path.exists(path)
        self._path = path
        self._to_load = to_load
        if self._to_load:
            self._raw_df = self.load()
            print('loading dataset')
        if raw_df is not None:
            self._raw_df = raw_df
            print('using raw_df')
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

    @num_instances.setter
    def num_instances(self, value: int):
        self._num_instances = value

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
    def to_load(self) -> bool:
        return self._to_load

    @to_load.setter
    def to_load(self, value: bool):
        self._to_load = value

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

    @classmethod
    def concatenate(cls, dataset_list: List['Dataset'], path: str) -> 'Dataset':
        dataset_types: Set[DatasetType] = {ds.dtype for ds in dataset_list}
        assert len(dataset_types) == 1

        dataset_labels: Set[str] = {ds.label_column_name for ds in dataset_list}
        assert len(dataset_labels) == 1

        dataset_categorical_feature_names: Set[Set[str]] = {set(ds.categorical_feature_names) for ds in dataset_list}
        categorical_feature_names: Set[str] = set()
        for inner_set in dataset_categorical_feature_names:
            categorical_feature_names |= inner_set
        assert categorical_feature_names == dataset_list[0].categorical_feature_names

        dataset_numeric_feature_names: Set[Set[str]] = {set(ds.numeric_feature_names) for ds in dataset_list}
        numeric_feature_names: Set[str] = set()
        for inner_set in dataset_numeric_feature_names:
            numeric_feature_names |= inner_set
        assert numeric_feature_names == dataset_list[0].numeric_feature_names

        raw_df: pd.DataFrame = pd.concat([ds.raw_df for ds in dataset_list])
        pd.to_pickle(raw_df)  # TODO: pickle the new dataset

        return cls(
            dtype=dataset_types.pop(),
            path=path,
            label_column_name=dataset_labels.pop(),
            categorical_feature_names=list(categorical_feature_names),
            numeric_feature_names=list(numeric_feature_names),
            to_load=False,
            raw_df=raw_df
        )

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """ loads the dataset from memory

        Returns:
            (pd.DataFrame): the raw dataframe

        """
        raise NotImplementedError
