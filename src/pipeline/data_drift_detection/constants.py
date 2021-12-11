from enum import Enum


class DataDriftType(Enum):
    Model = 0
    Statistical = 1
    Mean = 2
    Variance = 3
    NumNulls = 4
