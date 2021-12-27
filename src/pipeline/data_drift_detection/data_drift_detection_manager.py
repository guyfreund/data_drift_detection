from typing import List
import numpy as np

from src.pipeline.config import Config
from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.detectors.data_drift_detector import DataDriftDetector
from src.pipeline.data_drift_detection.detectors.model_based_detector import ModelBasedDetector
from src.pipeline.data_drift_detection.detectors.scikit_multiflow_data_drift_detector import \
    ScikitMultiflowDataDriftDetector
from src.pipeline.data_drift_detection.detectors.statistical_based_detector import StatisticalBasedDetector
from src.pipeline.data_drift_detection.detectors.tensorflow_data_drift_detector import \
    TensorflowDataValidationDataDriftDetector
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline import logger

logging = logger.get_logger(__name__)


class DataDriftDetectionManagerInfo:
    def __init__(self, deployment_dataset_plus: Dataset, training_processed_df_plus_path: str,
                 preprocessor: IPreprocessor, model: IModel, deployment_dataset: Dataset,
                 training_feature_metrics_list_path: str, training_processed_df_path: str):
        self.deployment_dataset: Dataset = deployment_dataset
        self.deployment_dataset_plus: Dataset = deployment_dataset_plus
        self.training_processed_df_path: str = training_processed_df_path
        self.training_processed_df_plus_path: str = training_processed_df_plus_path
        self.training_feature_metrics_list_path: str = training_feature_metrics_list_path
        self.preprocessor: IPreprocessor = preprocessor
        self.model: IModel = model


class DataDriftDetectionManager(IManager):
    def __init__(self, info: DataDriftDetectionManagerInfo):
        self._internal_data_drift_detector = DataDriftDetector(detectors=[
            StatisticalBasedDetector(
                deployment_dataset=info.deployment_dataset,
                training_feature_metrics_list_path=info.training_feature_metrics_list_path,
                preprocessor=info.preprocessor
            ),
            ModelBasedDetector(
                deployment_dataset_plus=info.deployment_dataset_plus,
                training_processed_df_plus_path=info.training_processed_df_plus_path,
                preprocessor=info.preprocessor,
                model=info.model
            )
        ])
        self._tensorflow_data_drift_detector = TensorflowDataValidationDataDriftDetector()
        self._scikit_multiflow_data_drift_detector = ScikitMultiflowDataDriftDetector(
            deployment_dataset=info.deployment_dataset,
            training_processed_df_path=info.training_processed_df_path,
            preprocessor=info.preprocessor
        )
        self._internal_data_drift = None
        self._tensorflow_data_drift = None
        self._scikit_multiflow_data_drift = None

    def manage(self) -> DataDrift:
        self._internal_data_drift = self._internal_data_drift_detector.detect().is_drifted
        self._tensorflow_data_drift = self._tensorflow_data_drift_detector.detect().is_drifted
        self._scikit_multiflow_data_drift = self._scikit_multiflow_data_drift_detector.detect().is_drifted

        is_drifted = np.dot(
            np.array([self._internal_data_drift, self._tensorflow_data_drift, self._scikit_multiflow_data_drift]),
            np.array([
                Config().data_drift.internal_data_drift_detector.weight,
                Config().data_drift.tensorflow_data_validation.weight,
                Config().data_drift.scikit_multiflow.weight
            ])
        ) >= Config().data_drift.threshold

        logging.info(f"Internal data drift: is_drifted={self._internal_data_drift}")
        # logging.info(f"Tensorflow data drift: is_drifted={self._tensorflow_data_drift}")
        # logging.info(f"Scikit multiflow data drift: is_drifted={self._scikit_multiflow_data_drift}")

        return DataDrift(is_drifted=is_drifted)


class MultipleDatasetDataDriftDetectionManager(IManager):
    def __init__(self, info_list: List[DataDriftDetectionManagerInfo]):
        self.data_drift_detection_managers: List[DataDriftDetectionManager] = [DataDriftDetectionManager(info) for info in info_list]

    def manage(self) -> List[DataDrift]:
        data_drifts: List[DataDrift] = [manager.manage() for manager in self.data_drift_detection_managers]
        logging.info(f"Managers data drift detections: {[data_drift.is_drifted for data_drift in data_drifts]}")
        return data_drifts
