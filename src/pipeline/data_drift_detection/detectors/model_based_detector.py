from typing import Dict
import pandas as pd
import numpy as np
import os
from sklearn import metrics

from src.pipeline.model.model_metrics import Accuracy, Precision, Recall, F1, AUC
from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.data_drift_detection.data_drift import ModelBasedDataDrift
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.config import Config
from src.pipeline.model.constants import ModelMetricType
from src.pipeline import logger
from src.pipeline.datasets.constants import DatasetType
from sklearn.linear_model import LogisticRegression

logging = logger.get_logger(__name__)


class ModelBasedDetector(IDataDriftDetector):
    def __init__(self, deployment_dataset_plus: Dataset, training_processed_df_plus_path: str, preprocessor: IPreprocessor, model: IModel):
        # assert os.path.exists(training_processed_df_plus_path)
        self._training_processed_df_plus_path: str = training_processed_df_plus_path
        self._deployment_dataset_plus: Dataset = deployment_dataset_plus
        self._preprocessor: IPreprocessor = preprocessor
        self._model: IModel = model

    def detect(self) -> ModelBasedDataDrift:
        # concatenate the training and deployment processed dataframes
        training_processed_df_plus: pd.DataFrame = pd.read_pickle(self._training_processed_df_plus_path)
        # TODO: think maybe to use pickle here
        _, deployment_processed_df_plus, _ = self._preprocessor.preprocess(dataset=self._deployment_dataset_plus, generate_dataset_plus=False)

        # update the encoder to see training data also
        label_column_name = Config().preprocessing.data_drift_model_label_column_name
        encoder_for_data_drift_label = self._preprocessor.label_preprocessor.encoder[label_column_name]
        encoder_for_data_drift_label.classes_ = np.append(encoder_for_data_drift_label.classes_, DatasetType.Training.value)
        logging.info('Updated the encoder dict')

        logging.info(f'Concatenated training and deployment datasets plus, {self._deployment_dataset_plus.name}')
        processed_df: pd.DataFrame = pd.concat([training_processed_df_plus, deployment_processed_df_plus])

        # train and evaluate the model
        X_train, X_validation, X_test, y_train, y_validation, y_test = self._preprocessor.split(
            processed_df=processed_df, label_column_name=label_column_name)
        processed_df.to_csv(os.path.abspath(os.path.join(__file__, "..", "..", "..", "data_generation", "raw_files", f"training_and_deployment_df.csv")), index=False)
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)

        # TODO: think maybe to use pickle here
        # self._model.train(X_train=X_train, y_train=y_train)
        self._model.tune_hyperparameters(X_validation=X_validation, y_validation=y_validation)
        y_pred = logmodel.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)

        # logging.info(f'Model Evaluation: model was evaluated successfully. results are as follows: '
        #              f'Accuracy: {round(accuracy * 100, 2)}%, '
        #              f'Precision: {round(precision * 100, 2)}%, '
        #              f'Recall: {round(recall * 100, 2)}%, '
        #              f'F1: {round(f1 * 100, 2)}%, '
        #              f'AUC: {round(auc, 2)}')

        model_metrics_dict = {
            ModelMetricType.Accuracy: Accuracy(value=accuracy),
            ModelMetricType.Precision: Precision(value=precision),
            ModelMetricType.Recall: Recall(value=recall),
            ModelMetricType.F1: F1(value=f1),
            ModelMetricType.AUC: AUC(value=auc)
        }


        # data drift is detected if model accuracy is not like a coin-flip
        model_accuracy: float = model_metrics_dict[ModelMetricType.Accuracy].value
        is_accuracy_like_coin_flip = np.abs(model_accuracy - 0.5) < Config().data_drift.internal_data_drift_detector.model_based_threshold  # TODO: change to np.isclose
        logging.info(f"Model based data drift detector: is_drifted={not is_accuracy_like_coin_flip}")
        return ModelBasedDataDrift(is_drifted=not is_accuracy_like_coin_flip)
