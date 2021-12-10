import pandas as pd

from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DATASET_PATH


class BankMarketingDataset(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=BANK_MARKETING_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)
