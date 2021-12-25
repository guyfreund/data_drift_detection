from enum import Enum


class DatasetType(Enum):
    Training = 0
    Deployment = 1
    Retraining = 2
    TrainingSampled = 3
    DeploymentSampled = 4
