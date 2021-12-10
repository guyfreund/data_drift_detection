from typing import List

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.data_drift_detection.data_drift import DataDrift


class DataDriftDetector(IDataDriftDetector):
    def __init__(self, detectors: List[IDataDriftDetector]):
        self._detectors = detectors

    def detect(self) -> DataDrift:
        is_drifted = any(detector.detect().is_drifted for detector in self._detectors)
        return DataDrift(is_drifted=is_drifted)
