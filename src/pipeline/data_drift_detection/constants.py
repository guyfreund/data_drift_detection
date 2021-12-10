from enum import Enum


from src.pipeline.data_drift_detection.data_drift import DataDrift, ModelBasedDataDrift, StatisticalBasedDataDrift, \
    MeanDataDrift, VarianceDataDrift, NumNullsDataDrift


class DataDriftType(Enum):
    Model = 0
    Statistical = 1
    Mean = 2
    Variance = 3
    NumNulls = 4


DATA_DRIFT_TYPE_TO_OBJ = {
    DataDriftType.Model: ModelBasedDataDrift,
    DataDriftType.Statistical: StatisticalBasedDataDrift,
    DataDriftType.Mean: MeanDataDrift,
    DataDriftType.Variance: VarianceDataDrift,
    DataDriftType.NumNulls: NumNullsDataDrift
}