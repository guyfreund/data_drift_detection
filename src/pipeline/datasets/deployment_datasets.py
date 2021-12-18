import pandas as pd

from src.pipeline.config import Config
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DEPLOYMENT_DATASET_PATH, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH


class BankMarketingDeploymentDataset(Dataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class BankMarketingDeploymentDatasetPlus(Dataset):
    def __init__(self):
        data_drift_model_label_column_name = Config().preprocessing.data_drift_model_label_column_name
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features + [data_drift_model_label_column_name],
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=data_drift_model_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class GermanCreditDeploymentDataset(Dataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        raise NotImplementedError  # TODO: implement


class GermanCreditDeploymentDatasetPlus(Dataset):
    def __init__(self):
        data_drift_model_label_column_name = Config().preprocessing.data_drift_model_label_column_name
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features + [data_drift_model_label_column_name],
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=data_drift_model_label_column_name
        )

    def load(self) -> pd.DataFrame:
        raise NotImplementedError  # TODO: implement
