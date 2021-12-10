from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector


class StatisticalBasedDetector(IDataDriftDetector):
    def __init__(self):
        self.wow = None

    def detect(self) -> bool:
        return True
