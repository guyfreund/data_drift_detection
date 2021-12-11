import numpy as np

from src.pipeline.config import Config
from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.detector.data_drift_detector import DataDriftDetector
from src.pipeline.data_drift_detection.detector.model_based_detector import ModelBasedDetector
from src.pipeline.data_drift_detection.detector.scikit_multiflow_data_drift_detector import \
    ScikitMultiflowDataDriftDetector
from src.pipeline.data_drift_detection.detector.statistical_based_detector import StatisticalBasedDetector
from src.pipeline.data_drift_detection.detector.tensorflow_data_drift_detector import \
    TensorflowDataValidationDataDriftDetector
from src.pipeline.interfaces.imanager import IManager


class DataDriftDetectionManager(IManager):
    def __init__(self):
        self._internal_statistical_based_data_drift_detector = self._initiate_internal_statistical_based_data_drift_detector()
        self._internal_model_based_data_drift_detector = self._initiate_internal_model_based_data_drift_detector()
        detectors = [self._internal_statistical_based_data_drift_detector, self._internal_model_based_data_drift_detector]
        self._internal_data_drift_detector = DataDriftDetector(detectors=detectors)
        self._tensorflow_data_drift_detector = TensorflowDataValidationDataDriftDetector()
        self._scikit_multiflow_data_drift_detector = ScikitMultiflowDataDriftDetector()
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

        return DataDrift(is_drifted=is_drifted)

    def _initiate_internal_model_based_data_drift_detector(self) -> ModelBasedDetector:
        pass

    def _initiate_internal_statistical_based_data_drift_detector(self) -> StatisticalBasedDetector:
        pass
