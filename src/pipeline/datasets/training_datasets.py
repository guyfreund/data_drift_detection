import pandas as pd

from src.pipeline.config import Config
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DATASET_PATH, GERMAN_CREDIT_DATASET_PATH


class BankMarketingDataset(Dataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class GermanCreditDataset(Dataset):
    NAMES = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, names=GermanCreditDataset.NAMES, delimiter=' ')
