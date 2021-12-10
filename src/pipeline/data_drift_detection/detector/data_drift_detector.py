from typing import List

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector


class DataDriftDetector(IDataDriftDetector):
    def __init__(self, detectors: List[IDataDriftDetector]):
        self._detectors = detectors

    def detect(self) -> bool:
        return any(detector.detect().is_drifted for detector in self._detectors)
