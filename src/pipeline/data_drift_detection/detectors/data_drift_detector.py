from typing import List

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.data_drift_detection.data_drift import DataDrift


class DataDriftDetector(IDataDriftDetector):
    def __init__(self, detectors: List[IDataDriftDetector]):
        self._detectors = detectors
        self._data_drifts: List[DataDrift] = []

    def detect(self) -> DataDrift:
        self._data_drifts = [detector.detect() for detector in self._detectors]
        return DataDrift(is_drifted=any([dd.is_drifted for dd in self._data_drifts]))
