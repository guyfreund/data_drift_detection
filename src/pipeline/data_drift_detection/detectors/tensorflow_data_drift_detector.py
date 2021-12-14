from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector


class TensorflowDataValidationDataDriftDetector(IDataDriftDetector):
    def __init__(self):
        pass

    def detect(self) -> DataDrift:
        return DataDrift(is_drifted=False)
