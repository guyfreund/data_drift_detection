from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.data_drift_detection.detector.model_based_detector import ModelBasedDetector
from src.pipeline.data_drift_detection.detector.statistical_based_detector import StatisticalBasedDetector


class DataDriftDetector(IDataDriftDetector):
    def __init__(self):
        self._statistical_based_detector = StatisticalBasedDetector()
        self._model_based_detector = ModelBasedDetector()
        self._detectors = [self._statistical_based_detector, self._model_based_detector]

    def detect(self) -> bool:
        return any(detector.detect() for detector in self._detectors)
