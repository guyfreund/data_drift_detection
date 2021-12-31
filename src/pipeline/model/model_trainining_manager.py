from typing import List, Tuple, Dict

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset
from src.pipeline import logger

logging = logger.get_logger(__name__)


class ModelTrainingManagerInfo:
    def __init__(self, preprocessor: IPreprocessor, dataset: Dataset, model: IModel, to_train: bool = True):
        self.preprocessor: IPreprocessor = preprocessor
        self.training_dataset: Dataset = dataset
        self.model: IModel = model
        self._to_train = to_train

    @property
    def to_train(self) -> bool:
        return self._to_train

    @to_train.setter
    def to_train(self, value: bool):
        self._to_train = value


class ModelTrainingManager(IManager):
    def __init__(self, info: ModelTrainingManagerInfo):
        self._info = info

    def manage(self) -> Tuple[ModelTrainingManagerInfo, List[IFeatureMetrics], Dict[ModelMetricType, IModelMetric]]:
        assert bool(self._info.to_train) is True
        processed_dataset, processed_dataset_plus, feature_metrics_list = self._info.preprocessor.preprocess(self._info.training_dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = self._info.preprocessor.split(
            processed_df=processed_dataset,
            label_column_name=self._info.training_dataset.label_column_name,
            dataset_class_name=self._info.training_dataset.name
        )

        self._info.model.train(X_train, y_train)
        self._info.model.tune_hyperparameters(X_validation, y_validation)
        model_metrics_dict: Dict[ModelMetricType, IModelMetric] = self._info.model.evaluate(X_test, y_test)

        return self._info, feature_metrics_list, model_metrics_dict

    @property
    def info(self) -> ModelTrainingManagerInfo:
        return self._info

    @info.setter
    def info(self, value: ModelTrainingManagerInfo):
        self._info = value


class MultipleDatasetModelTrainingManager(IManager):
    def __init__(self, info_list: List[ModelTrainingManagerInfo]):
        self._info_list = info_list
        self._model_training_managers: List[ModelTrainingManager] = []

    def manage(self) -> Tuple[List[ModelTrainingManagerInfo], List[List[IFeatureMetrics]], List[Dict[ModelMetricType, IModelMetric]]]:
        self._model_training_managers: List[ModelTrainingManager] = [
            ModelTrainingManager(info) for info in self._info_list if info.to_train
        ]  # critical for retraining where we want to update the info list - don't move to constructor
        model_training_manger_info_list: List[ModelTrainingManagerInfo] = []
        model_training_manager_feature_metrics_lists: List[List[IFeatureMetrics]] = []
        model_training_model_metrics_list: List[Dict[ModelMetricType, IModelMetric]] = []

        for model_training_manager in self._model_training_managers:
            info, feature_metrics_list, model_metrics_dict = model_training_manager.manage()

            model_training_manger_info_list.append(info)
            model_training_manager_feature_metrics_lists.append(feature_metrics_list)
            model_training_model_metrics_list.append(model_metrics_dict)

        return model_training_manger_info_list, model_training_manager_feature_metrics_lists, model_training_model_metrics_list

    @property
    def info_list(self) -> List[ModelTrainingManagerInfo]:
        return self._info_list

    @info_list.setter
    def info_list(self, value: List[ModelTrainingManagerInfo]):
        self._info_list = value
