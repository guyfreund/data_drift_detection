import pickle
from abc import abstractmethod
from typing import List, Set, Optional
import pandas as pd
import os

from src.pipeline.config import Config
from src.pipeline.datasets.constants import DatasetType


class Dataset:
    """
    A class that represents a dataset
    """

    def __init__(self, dtype: DatasetType, path: str, label_column_name: str, categorical_feature_names: List[str],
                 numeric_feature_names: List[str], to_load: bool = True, raw_df: Optional[pd.DataFrame] = None,
                 name: str = '', original_label_column_name: str = ''):
        assert os.path.exists(path)
        self._path = path
        self._to_load = to_load
        self._raw_df = raw_df
        self._name = self.__class__.__name__ if not name else name
        if self._to_load:
            self._raw_df = self.load()
            print('loading dataset')
        self._num_instances, self._num_features = self._raw_df.shape
        self._dtype = dtype
        self._label_column_name = label_column_name
        self._categorical_feature_names = categorical_feature_names
        self._numeric_feature_names = numeric_feature_names
        self._original_label_column_name = original_label_column_name
        self._update_types()

    def _update_types(self):
        for col in self._categorical_feature_names:
            self._raw_df[col] = self._raw_df[col].astype('category')

    @property
    def original_label_column_name(self) -> str:
        return self._original_label_column_name

    @original_label_column_name.setter
    def original_label_column_name(self, value: str):
        self._original_label_column_name = value

    @property
    def name(self) -> str:
        return self._name

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

        dataset_labels: Set[str] = {ds.label_column_name for ds in dataset_list}
        assert len(dataset_labels) == 1

        dataset_categorical_feature_names: List[List[str]] = [ds.categorical_feature_names for ds in dataset_list]
        categorical_feature_names: Set[str] = set()
        for inner_list in dataset_categorical_feature_names:
            categorical_feature_names |= set(inner_list)
        assert categorical_feature_names == set(dataset_list[0].categorical_feature_names)

        dataset_numeric_feature_names: List[List[str]] = [ds.numeric_feature_names for ds in dataset_list]
        numeric_feature_names: Set[str] = set()
        for inner_list in dataset_numeric_feature_names:
            numeric_feature_names |= set(inner_list)
        assert numeric_feature_names == set(dataset_list[0].numeric_feature_names)

        raw_df: pd.DataFrame = pd.concat([ds.raw_df for ds in dataset_list])
        with open(path, 'wb') as output:
            pickle.dump(raw_df, output)

        return cls(
            dtype=DatasetType.Retraining,
            path=path,
            name='_'.join(dataset.name for dataset in dataset_list),
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
        if self._raw_df is not None:
            return self._raw_df

        raise NotImplementedError


class SampledDataset(Dataset):
    def __init__(self, path: str, numeric_feature_names: List[str],
                 categorical_feature_names: List[str], label_column_name: str, dtype: DatasetType,
                 sample_size_in_percent: float, original_dataset: Dataset = None, raw_df_paths: List[str] = None,
                 original_label_column_name: str = ''):

        if original_dataset is not None:
            raw_df = original_dataset.raw_df
        else:
            raw_dfs: List[pd.DataFrame] = []
            for raw_df_path in raw_df_paths:
                df = pd.read_pickle(raw_df_path)
                raw_dfs.append(df)
            # used to concat X to y
            raw_df = pd.concat(raw_dfs, axis=1)

        # sample dataframe
        raw_df: pd.DataFrame = raw_df.sample(frac=sample_size_in_percent, replace=False)

        with open(path, 'wb') as output:
            pickle.dump(raw_df, output)

        super().__init__(
            dtype=dtype,
            path=path,
            label_column_name=label_column_name,
            categorical_feature_names=categorical_feature_names,
            numeric_feature_names=numeric_feature_names,
            raw_df=raw_df,
            to_load=False,
            original_label_column_name=original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return super().load()
