from enum import Enum


# TODO: figure out
class DataDriftType(Enum):
    Model = 0
    Mean = 1
    Variance = 2
    NumNulls = 3

