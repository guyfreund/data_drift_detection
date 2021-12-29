import pandas as pd

from src.pipeline.config import Config
from src.pipeline.datasets.dataset import Dataset, SampledDataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DEPLOYMENT_DATASET_PATH, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_SAMPLED_DEPLOYMENT_DATASET, \
    BANK_MARKETING_SAMPLED_DEPLOYMENT_DATASET


class BankMarketingDeploymentDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Deployment,
            path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class BankMarketingDeploymentDatasetPlus(Dataset):
    def __init__(self, to_load: bool = True):
        data_drift_model_label_column_name = Config().preprocessing.data_drift_model_label_column_name
        super().__init__(
            dtype=DatasetType.Deployment,
            path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=data_drift_model_label_column_name,
            original_label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class GermanCreditDeploymentDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Deployment,
            path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class GermanCreditDeploymentDatasetPlus(Dataset):
    def __init__(self, to_load: bool = True):
        data_drift_model_label_column_name = Config().preprocessing.data_drift_model_label_column_name
        super().__init__(
            dtype=DatasetType.Deployment,
            path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=data_drift_model_label_column_name,
            original_label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class BankMarketingSampledDeploymentDataset(SampledDataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.DeploymentSampled,
            original_dataset=BankMarketingDeploymentDataset(),
            path=BANK_MARKETING_SAMPLED_DEPLOYMENT_DATASET,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            sample_size_in_percent=Config().retraining.deployment_sample_size_in_percent,
            original_label_column_name=Config().preprocessing.bank_marketing.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class GermanCreditSampledDeploymentDataset(SampledDataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.DeploymentSampled,
            original_dataset=GermanCreditDeploymentDataset(),
            path=GERMAN_CREDIT_SAMPLED_DEPLOYMENT_DATASET,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            sample_size_in_percent=Config().retraining.deployment_sample_size_in_percent,
            original_label_column_name=Config().preprocessing.german_credit.original_label_column_name
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)
