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
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, names=Config().preprocessing.german_credit.names, delimiter=' ')


class BankMarketingDatasetPlus(Dataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features + ['y'],
            label_column_name=Config().preprocessing.data_drift_model_label_column_name
        )

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, delimiter=';')
        df[Config().preprocessing.data_drift_model_label_column_name] = DatasetType.Training.value
        return df


class GermanCreditDatasetPlus(Dataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features + ['y'],
            label_column_name=Config().preprocessing.data_drift_model_label_column_name
        )

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, names=Config().preprocessing.german_credit.names, delimiter=' ')
        df[Config().preprocessing.data_drift_model_label_column_name] = DatasetType.Training.value
        return df
