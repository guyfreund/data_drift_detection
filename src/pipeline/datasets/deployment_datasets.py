import pandas as pd

from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DEPLOYMENT_DATASET_PATH, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH


class BankMarketingDeploymentDataset(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class BankMarketingDeploymentDatasetPlus(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class GermanCreditDeploymentDataset(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class GermanCreditDeploymentDatasetPlus(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)