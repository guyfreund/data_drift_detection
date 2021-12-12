from src.pipeline.datasets.constants import DatasetType
from src.pipeline.preprocessing.interfaces.constants import FeatureType
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics


class CategoricalFeatureMetrics(IFeatureMetrics):
    def __init__(self, name: str, dataset_type: DatasetType):
        self._name = name
        self._dataset_type = dataset_type
        self._feature_type = FeatureType.Categorical
        self._number_of_nulls = 0
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
    def number_of_nulls(self) -> int:
        return self._number_of_nulls

    @property
    def is_important_feature(self) -> bool:
        return self._is_important_feature

