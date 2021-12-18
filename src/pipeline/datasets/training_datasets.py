from typing import List

import pandas as pd

from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DATASET_PATH, GERMAN_CREDIT_DATASET_PATH


class BankMarketingDataset(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=BANK_MARKETING_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class GermanCreditDataset(Dataset):
    def __init__(self):
        super().__init__(dtype=DatasetType.Training, path=GERMAN_CREDIT_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
                 'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
                 'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
                 'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

        return pd.read_csv(self._path, names=names, delimiter=' ')
