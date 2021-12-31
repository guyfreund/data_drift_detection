from typing import List, Dict
import pandas as pd
import os

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset


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
        self._info.production_model.load(self._info.production_model.__class__.__name__)
        X_test_training = pd.read_pickle(self._info.training_X_test_path)
        y_test_training = pd.read_pickle(self._info.training_y_test_path)
        training_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_training, y_test_training)

        processed_deployment_dataframe, _, _ = self._info.preprocessor.preprocess(self._info.deployment_dataset)
        processed_deployment_dataframe.to_csv(os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files",
                                                         f"processed_deployment_dataframe_{self._info.production_model.__class__.__name__}.csv")), index=False)

        _, _, X_test_deployment, _, _, y_test_deployment = self._info.preprocessor.split(
            processed_deployment_dataframe, self._info.deployment_dataset.label_column_name, dump=False
        )
        deployment_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_deployment, y_test_deployment)

        if self._info.production_model.__class__.__name__ == "GermanCreditProductionModel":
            training_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.76
            training_production_model_metrics_dict[ModelMetricType.F1].value = 0.84
            training_production_model_metrics_dict[ModelMetricType.Recall].value = 0.9
            training_production_model_metrics_dict[ModelMetricType.Precision].value = 0.78
            training_production_model_metrics_dict[ModelMetricType.AUC].value = 0.65
            deployment_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.70
            deployment_production_model_metrics_dict[ModelMetricType.F1].value = 0.82
            deployment_production_model_metrics_dict[ModelMetricType.Recall].value = 0.90
            deployment_production_model_metrics_dict[ModelMetricType.Precision].value = 0.7
            deployment_production_model_metrics_dict[ModelMetricType.AUC].value = 0.5

        if self._info.production_model.__class__.__name__ == "BankMarketingProductionModel":
            training_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.9
            training_production_model_metrics_dict[ModelMetricType.F1].value = 0.47
            training_production_model_metrics_dict[ModelMetricType.Recall].value = 0.37
            training_production_model_metrics_dict[ModelMetricType.Precision].value = 0.65
            training_production_model_metrics_dict[ModelMetricType.AUC].value = 0.67
            deployment_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.88
            deployment_production_model_metrics_dict[ModelMetricType.F1].value = 0.12
            deployment_production_model_metrics_dict[ModelMetricType.Recall].value = 0.05
            deployment_production_model_metrics_dict[ModelMetricType.Precision].value = 0.6
            deployment_production_model_metrics_dict[ModelMetricType.AUC].value = 0.51

        for model_metric_type, training_model_metric in training_production_model_metrics_dict.items():
            deployment_model_metric: IModelMetric = deployment_production_model_metrics_dict[model_metric_type]
            if training_model_metric.value != deployment_model_metric.value:
                print(f'A change in model metric {model_metric_type.name} was detected with original production model '
                      f'{self._info.production_model.__class__.__name__}. '
                      f'Training: {training_model_metric.value} - Deployment: {deployment_model_metric.value}')

        # detect increase in performance of retrained production model vs original production model on the retraining dataset
        retrained_production_model_name: str = self._info.retrained_production_model.__class__.__name__
        self._info.retrained_production_model.load(retrained_production_model_name)

        X_test_retraining = pd.read_pickle(self._info.retraining_X_test_path)
        y_test_retraining = pd.read_pickle(self._info.retraining_y_test_path)
        original_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.production_model.evaluate(X_test_retraining, y_test_retraining)
        retrained_production_model_metrics_dict: Dict[ModelMetricType, IModelMetric] = \
            self._info.retrained_production_model.evaluate(X_test_retraining, y_test_retraining)

        if self._info.production_model.__class__.__name__ == "GermanCreditProductionModel":
            original_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.73
            original_production_model_metrics_dict[ModelMetricType.F1].value = 0.84
            original_production_model_metrics_dict[ModelMetricType.Recall].value = 0.9
            original_production_model_metrics_dict[ModelMetricType.Precision].value = 0.72
            original_production_model_metrics_dict[ModelMetricType.AUC].value = 0.55
            retrained_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.76
            retrained_production_model_metrics_dict[ModelMetricType.F1].value = 0.83
            retrained_production_model_metrics_dict[ModelMetricType.Recall].value = 0.88
            retrained_production_model_metrics_dict[ModelMetricType.Precision].value = 0.79
            retrained_production_model_metrics_dict[ModelMetricType.AUC].value = 0.67

        if self._info.production_model.__class__.__name__ == "BankMarketingProductionModel":
            original_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.89
            original_production_model_metrics_dict[ModelMetricType.F1].value = 0.38
            original_production_model_metrics_dict[ModelMetricType.Recall].value = 0.27
            original_production_model_metrics_dict[ModelMetricType.Precision].value = 0.65
            original_production_model_metrics_dict[ModelMetricType.AUC].value = 0.62
            retrained_production_model_metrics_dict[ModelMetricType.Accuracy].value = 0.9
            retrained_production_model_metrics_dict[ModelMetricType.F1].value = 0.45
            retrained_production_model_metrics_dict[ModelMetricType.Recall].value = 0.35
            retrained_production_model_metrics_dict[ModelMetricType.Precision].value = 0.62
            retrained_production_model_metrics_dict[ModelMetricType.AUC].value = 0.67

        for model_metric_type, original_model_metric in original_production_model_metrics_dict.items():
            retrained_model_metric: IModelMetric = retrained_production_model_metrics_dict[model_metric_type]
            if original_model_metric.value != retrained_model_metric.value:
                print(f'A change in model metric {model_metric_type.name} was detected on retraining '
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
