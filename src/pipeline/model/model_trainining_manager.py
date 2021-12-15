from typing import List, Tuple, Dict

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset


class ModelTrainingManagerInfo:
    def __init__(self, preprocessor: IPreprocessor, dataset: Dataset, model: IModel):
        self.preprocessor: IPreprocessor = preprocessor
        self.training_dataset: Dataset = dataset
        self.model: IModel = model


class ModelTrainingManager(IManager):
    def __init__(self, info: ModelTrainingManagerInfo):
        self._info = info

    def manage(self) -> Tuple[ModelTrainingManagerInfo, List[IFeatureMetrics], Dict[ModelMetricType, IModelMetric]]:
        processed_dataset, processed_dataset_plus, feature_metrics_list = self._info.preprocessor.preprocess(self._info.training_dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = self._info.preprocessor.split(processed_dataset)

        self._info.model.train(X_train, y_train)
        self._info.model.tune_hyperparameters(X_validation, y_validation)
        model_metrics_dict = self._info.model.evaluate(X_test, y_test)

        self._info.model.load(self._info.model.__class__.__name__)
        return self._info, feature_metrics_list, model_metrics_dict

    @property
    def info(self) -> ModelTrainingManagerInfo:
        return self._info

    @info.setter
    def info(self, info):
        self._info = info


class MultipleDatasetModelTrainingManager(IManager):
    def __init__(self, info_list: List[ModelTrainingManagerInfo]):
        self._model_training_managers: List[ModelTrainingManager] = [ModelTrainingManager(info) for info in info_list]

    def manage(self) -> Tuple[List[ModelTrainingManagerInfo], List[List[IFeatureMetrics]], List[Dict[ModelMetricType, IModelMetric]]]:
        model_training_manger_info_list: List[ModelTrainingManagerInfo] = []
        model_training_manager_feature_metrics_lists: List[List[IFeatureMetrics]] = []
        model_training_model_metrics_list: List[Dict[ModelMetricType, IModelMetric]] = []

        for model_training_manager in self._model_training_managers:
            info, feature_metrics_list, model_metrics_dict = model_training_manager.manage()

            model_training_manger_info_list.append(info)
            model_training_manager_feature_metrics_lists.append(feature_metrics_list)
            model_training_model_metrics_list.append(model_metrics_dict)

        return model_training_manger_info_list, model_training_manager_feature_metrics_lists, model_training_model_metrics_list
