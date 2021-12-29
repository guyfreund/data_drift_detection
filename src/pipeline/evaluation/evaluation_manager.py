from typing import List, Dict
import pandas as pd

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset
from src.pipeline import logger

logging = logger.get_logger(__name__)


class EvaluationManagerInfo:

    def __init__(self, production_model: IModel, retrained_production_model: IModel, preprocessor: IPreprocessor,
                 training_X_test_path: str, training_y_test_path: str, deployment_dataset: Dataset,
                 retraining_X_test_path: str, retraining_y_test_path: str, to_evaluate: bool = True):
        self.production_model: IModel = production_model
        self.retrained_production_model: IModel = retrained_production_model
        self.preprocessor: IPreprocessor = preprocessor
        self.training_X_test_path: str = training_X_test_path
        self.training_y_test_path: str = training_y_test_path
        self.deployment_dataset: Dataset = deployment_dataset
        self.retraining_X_test_path: str = retraining_X_test_path
        self.retraining_y_test_path: str = retraining_y_test_path
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
        logging.info(f'Original Production Model Evaluation {self._info.production_model.__class__.__name__}: '
                     f'training dataset vs deployment dataset:')

        self._info.production_model.load(self._info.production_model.__class__.__name__)
        X_test_training = pd.read_pickle(self._info.training_X_test_path)
        y_test_training = pd.read_pickle(self._info.training_y_test_path)
        training_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_training, y_test_training)

        processed_deployment_dataframe, _, _ = self._info.preprocessor.preprocess(self._info.deployment_dataset)
        _, _, X_test_deployment, _, _, y_test_deployment = self._info.preprocessor.split(
            processed_deployment_dataframe, self._info.deployment_dataset.label_column_name, dump=False
        )
        deployment_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_deployment, y_test_deployment)

        for model_metric_type, training_model_metric in training_production_model_metrics_dict.items():
            deployment_model_metric: IModelMetric = deployment_production_model_metrics_dict[model_metric_type]
            if training_model_metric.value != deployment_model_metric.value:
                logging.info(f'A change in model metric {model_metric_type.name} was detected with original production model '
                      f'{self._info.production_model.__class__.__name__}. '
                      f'Training: {training_model_metric.value} - Deployment: {deployment_model_metric.value}')

        # detect increase in performance of retrained production model vs original production model on the retraining dataset
        logging.info(f'Original Production Model vs Retrained Model Evaluation {self._info.production_model.__class__.__name__}: '
                     f'on the sampled concatenated dataset:')
        retrained_production_model_name: str = self._info.retrained_production_model.__class__.__name__
        self._info.retrained_production_model.load(retrained_production_model_name)

        X_test_retraining = pd.read_pickle(self._info.retraining_X_test_path)
        y_test_retraining = pd.read_pickle(self._info.retraining_y_test_path)
        original_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_retraining, y_test_retraining)
        retrained_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.retrained_production_model.evaluate(X_test_retraining, y_test_retraining)

        for model_metric_type, original_model_metric in original_production_model_metrics_dict.items():
            retrained_model_metric: IModelMetric = retrained_production_model_metrics_dict[model_metric_type]
            if original_model_metric.value != retrained_model_metric.value:
                logging.info(f'A change in model metric {model_metric_type.name} was detected on retraining '
                      f'dataset {retrained_production_model_name}. '
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
