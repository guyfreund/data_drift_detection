from typing import Dict
import pandas as pd
import numpy as np
import os

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.data_drift_detection.data_drift import ModelBasedDataDrift
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.config.config import Config
from src.pipeline.model.constants import ModelMetricType


class ModelBasedDetector(IDataDriftDetector):
    def __init__(self, deployment_dataset_plus: Dataset, training_processed_df_plus_path: str, preprocessor: IPreprocessor, model: IModel):
        assert os.path.exists(training_processed_df_plus_path)
        self._training_processed_df_plus_path = training_processed_df_plus_path
        self._deployment_dataset_plus = deployment_dataset_plus
        self._preprocessor = preprocessor
        self._model = model

    def detect(self) -> ModelBasedDataDrift:
        # concatenate the training and deployment processed dataframes
        training_processed_df_plus = pd.read_pickle(self._training_processed_df_plus_path)
        _, deployment_processed_df_plus, _ = self._preprocessor.preprocess(dataset=self._deployment_dataset_plus)
        processed_df = pd.concat([training_processed_df_plus, deployment_processed_df_plus])

        # train and evaluate the model
        X_train, X_validation, X_test, y_train, y_validation, y_test = self._preprocessor.split(processed_df=processed_df)
        self._model.train(X_train=X_train, y_train=y_train)
        self._model.tune_hyperparameters(X_validation=X_validation, y_validation=y_validation)
        model_metrics_dict: Dict[ModelMetricType, IModelMetric] = self._model.evaluate(X_test=X_test, y_test=y_test)

        # data drift is detected if model accuracy is like a coin-flip
        model_accuracy = model_metrics_dict[ModelMetricType.Accuracy].value
        is_drifted = np.abs(0.5 - model_accuracy) < Config().data_drift.model_based_threshold

        return ModelBasedDataDrift(is_drifted=is_drifted)

