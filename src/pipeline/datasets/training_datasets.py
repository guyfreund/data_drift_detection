import pandas as pd
import pickle
from src.pipeline.config import Config
from src.pipeline.datasets.dataset import Dataset, SampledDataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import BANK_MARKETING_DATASET_PATH, GERMAN_CREDIT_DATASET_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_SAMPLED_DATASET_PATH, \
    GERMAN_CREDIT_SAMPLED_DATASET_PATH


class BankMarketingDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class GermanCreditDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, names=Config().preprocessing.german_credit.names, delimiter=' ')


class BankMarketingProcessedDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pickle:
        return pd.read_pickle(self._path)


class GermanCreditProcessedDataset(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            to_load=to_load
        )

    def load(self) -> pickle:
        return pd.read_pickle(self._path)


class BankMarketingDatasetPlus(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=BANK_MARKETING_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features + ['y'],
            label_column_name=Config().preprocessing.data_drift_model_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, delimiter=';')
        df[Config().preprocessing.data_drift_model_label_column_name] = DatasetType.Training.value
        return df


class GermanCreditDatasetPlus(Dataset):
    def __init__(self, to_load: bool = True):
        super().__init__(
            dtype=DatasetType.Training,
            path=GERMAN_CREDIT_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features + ['y'],
            label_column_name=Config().preprocessing.data_drift_model_label_column_name,
            to_load=to_load
        )

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, names=Config().preprocessing.german_credit.names, delimiter=' ')
        df[Config().preprocessing.data_drift_model_label_column_name] = DatasetType.Training.value
        return df


class BankMarketingSampledTrainingDataset(SampledDataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.TrainingSampled,
            original_dataset=BankMarketingDataset(),
            path=BANK_MARKETING_SAMPLED_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.bank_marketing.numeric_features,
            categorical_feature_names=Config().preprocessing.bank_marketing.categorical_features,
            label_column_name=Config().preprocessing.bank_marketing.original_label_column_name,
            sample_size_in_percent=Config().training_retraining.sample_size_in_percent
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, delimiter=';')


class GermanCreditSampledTrainingDataset(SampledDataset):
    def __init__(self):
        super().__init__(
            dtype=DatasetType.Training,
            original_dataset=GermanCreditDataset(),
            path=GERMAN_CREDIT_SAMPLED_DATASET_PATH,
            numeric_feature_names=Config().preprocessing.german_credit.numeric_features,
            categorical_feature_names=Config().preprocessing.german_credit.categorical_features,
            label_column_name=Config().preprocessing.german_credit.original_label_column_name,
            sample_size_in_percent=Config().retraining.training_sample_size_in_percent
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, names=Config().preprocessing.german_credit.names, delimiter=' ')
