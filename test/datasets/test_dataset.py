import numpy as np

from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.paths import BANK_MARKETING_RETRAINING_DF
from src.pipeline.datasets.training_datasets import BankMarketingDataset


def test_data_set_concatenation():
    bank_marketing_dataset = BankMarketingDataset()
    dataset: Dataset = Dataset.concatenate(
        dataset_list=[
            bank_marketing_dataset,
            bank_marketing_dataset
        ],
        path=BANK_MARKETING_RETRAINING_DF
    )

    assert all([
        dataset.num_instances == 2 * bank_marketing_dataset.num_instances,
        dataset.numeric_feature_names == bank_marketing_dataset.numeric_feature_names,
        dataset.categorical_feature_names == bank_marketing_dataset.categorical_feature_names,
        dataset.label_column_name == bank_marketing_dataset.label_column_name,
        np.all(dataset.load().to_numpy() == dataset.raw_df.to_numpy()),
        dataset.dtype == DatasetType.Retraining
    ]) is True
