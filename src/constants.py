from enum import Enum

class DatasetType(Enum):
    Training = 0
    Test = 1
    Deployment = 2


class FeatureType(Enum):
    Numeric = 0
    Categorical = 1
