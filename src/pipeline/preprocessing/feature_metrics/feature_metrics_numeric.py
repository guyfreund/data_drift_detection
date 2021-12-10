from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.constants import FeatureType
from src.pipeline.datasets.constants import DatasetType


class NumericFeatureMetrics(IFeatureMetrics):
    """
    A class that represents a numeric feature and its metrics
    """

    def __init__(self, name: str, dataset_type: DatasetType):
        self._name = name
        self._dataset_type = dataset_type
        self._feature_type = FeatureType.Numeric
        self._number_of_nulls = 0
        self._mean = 0
        self._variance = 0
        self._is_important_feature = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset_type(self) -> DatasetType:
        return self._dataset_type

    @property
    def feature_type(self) -> FeatureType:
        return self._feature_type

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

    @mean.setter
    def mean(self, value: float):
        self._mean = value

    @variance.setter
    def variance(self, value: float):
        self._variance = value

    @is_important_feature.setter
    def is_important_feature(self, value: bool):
        self._is_important_feature = value