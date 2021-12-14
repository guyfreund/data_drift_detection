from src.pipeline.datasets.constants import DatasetType
from src.pipeline.preprocessing.constants import FeatureType
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
import numpy as np


class CategoricalFeatureMetrics(IFeatureMetrics):
    def __init__(self, name: str, dataset_type: DatasetType):
        self._name = name
        self._dataset_type = dataset_type
        self._feature_type = FeatureType.Categorical
        self._number_of_nulls = 0
        self._mean = np.inf
        self._variance = np.inf
        self._is_important_feature = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset_type(self) -> DatasetType:
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, dataset_type):
        self._dataset_type = dataset_type

    @property
    def feature_type(self) -> FeatureType:
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def number_of_nulls(self) -> int:
        return self._number_of_nulls

    @property
    def is_important_feature(self) -> bool:
        return self._is_important_feature

    @name.setter
    def name(self, value: str):
        self._name = value

    @dataset_type.setter
    def dataset_type(self, value: DatasetType):
        self._dataset_type = value

    @number_of_nulls.setter
    def number_of_nulls(self, value: int):
        self._number_of_nulls = value

    @is_important_feature.setter
    def is_important_feature(self, value: bool):
        self._is_important_feature = value

