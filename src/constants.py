from enum import Enum

class DatasetType(Enum):
    Training = 0
    Test = 1
    Validation = 2
    Deployment = 3


class FeatureType(Enum):
    Numeric = 0
    Categorical = 1
