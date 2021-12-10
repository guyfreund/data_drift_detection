from enum import Enum


class DataDriftType(Enum):
    Model = 0
    Mean = 1
    Variance = 2
    NumNulls = 3

