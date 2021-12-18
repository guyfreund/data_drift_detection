from src.pipeline.data_drift_detection.data_drift import DataDrift, MeanDataDrift, VarianceDataDrift


def test_data_drift():
    assert DataDrift(True) == DataDrift(True)
    assert MeanDataDrift(True) != VarianceDataDrift(True)
    assert MeanDataDrift(True) != MeanDataDrift(False)
    assert MeanDataDrift(True) == MeanDataDrift(True)
    assert MeanDataDrift(True) != DataDrift(True)
