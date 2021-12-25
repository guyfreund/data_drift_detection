from typing import List, Dict

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset


class EvaluationManagerInfo:
    def __init__(self, production_model: IModel, retrained_production_model: IModel, preprocessor: IPreprocessor,
                 training_dataset: Dataset, deployment_dataset: Dataset, retraining_dataset: Dataset, to_evaluate: bool = True):
        self.production_model: IModel = production_model
        self.retrained_production_model: IModel = retrained_production_model
        self.preprocessor: IPreprocessor = preprocessor
        self.training_dataset: Dataset = training_dataset
        self.deployment_dataset: Dataset = deployment_dataset
        self.retraining_dataset: Dataset = retraining_dataset
        self._to_evaluate: bool = to_evaluate

    @property
    def to_evaluate(self) -> bool:
        return self._to_evaluate

    @to_evaluate.setter
    def to_evaluate(self, value: bool):
        self._to_evaluate = value


class EvaluationManager(IManager):
    def __init__(self, info: EvaluationManagerInfo):
        self._info = info

    def manage(self):
        # detect degradation of original production model, training dataset vs deployment dataset
        self._info.production_model.load(self._info.production_model.__class__.__name__)

        processed_training_dataframe, _, _ = self._info.preprocessor.preprocess(self._info.training_dataset)
        _, _, X_test_training, _, _, y_test_training = \
            self._info.preprocessor.split(processed_training_dataframe, self._info.training_dataset.label_column_name)
        training_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_training, y_test_training)

        processed_deployment_dataframe, _, _ = self._info.preprocessor.preprocess(self._info.deployment_dataset)
        _, _, X_test_deployment, _, _, y_test_deployment = \
            self._info.preprocessor.split(processed_deployment_dataframe, self._info.deployment_dataset.label_column_name)
        deployment_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_deployment, y_test_deployment)

        for model_metric_type, training_model_metric in training_production_model_metrics_dict.items():
            deployment_model_metric: IModelMetric = deployment_production_model_metrics_dict[model_metric_type]
            if training_model_metric.value != deployment_model_metric.value:
                print(f'A change in model metric {ModelMetricType.name} was detected with original production model '
                      f'{self._info.production_model.__class__.__name__}. '
                      f'Training: {training_model_metric.value} - Deployment: {deployment_model_metric.value}')

        # detect increase in performance of retrained production model vs original production model on the retraining dataset
        self._info.retrained_production_model.load(self._info.retrained_production_model.__class__.__name__)

        processed_retraining_dataframe, _, _ = self._info.preprocessor.preprocess(self._info.retraining_dataset)
        _, _, X_test_retraining, _, _, y_test_retraining = \
            self._info.preprocessor.split(processed_retraining_dataframe, self._info.retraining_dataset.label_column_name)

        original_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_retraining, y_test_retraining)
        retrained_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.retrained_production_model.evaluate(X_test_retraining, y_test_retraining)

        for model_metric_type, original_model_metric in original_production_model_metrics_dict.items():
            retrained_model_metric: IModelMetric = retrained_production_model_metrics_dict[model_metric_type]
            if original_model_metric.value != retrained_model_metric.value:
                print(f'A change in model metric {ModelMetricType.name} was detected on retraining '
                      f'dataset {self._info.retraining_dataset.__class__.__name__}. '
                      f'Original Model: {original_model_metric.value} - Retrained Model: {retrained_model_metric.value}')

    @property
    def info(self) -> EvaluationManagerInfo:
        return self._info

    @info.setter
    def info(self, value: EvaluationManagerInfo):
        self._info = value


class MultipleDatasetEvaluationManager(IManager):
    def __init__(self, info_list: List[EvaluationManagerInfo]):
        self._info_list = info_list
        self._evaluation_managers: List[EvaluationManager] = []

    def manage(self):
        self._evaluation_managers: List[EvaluationManager] = [
            EvaluationManager(info) for info in self._info_list if info.to_evaluate
        ]  # critical for retraining where we want to update the info list - don't move to constructor
        for evaluation_manager in self._evaluation_managers:
            evaluation_manager.manage()

    @property
    def info_list(self) -> List[EvaluationManagerInfo]:
        return self._info_list

    @info_list.setter
    def info_list(self, value: List[EvaluationManagerInfo]):
        self._info_list = value