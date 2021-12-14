import pytest

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.data_drift_detection_manager import DataDriftDetectionManagerInfo, \
    MultipleDatasetDataDriftDetectionManager
from src.pipeline.datasets.deployment_datasets import BankMarketingDeploymentDatasetPlus, BankMarketingDeploymentDataset
from src.pipeline.datasets.paths import BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
from src.pipeline.datasets.training_datasets import BankMarketingDataset
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.production_models import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class TestDataDriftDetectionManager:
    def test_no_data_drift(self):
        bank_marketing_info = DataDriftDetectionManagerInfo(
            deployment_dataset_plus=BankMarketingDataset(),
            training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
            preprocessor=Preprocessor(),  # TODO: fix
            model=BankMarketingProductionModel(),  # TODO: fix
            deployment_dataset=BankMarketingDataset(),
            training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
        )
        data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=[bank_marketing_info])
        data_drift: DataDrift = data_drift_detection_manager.manage()[0]
        assert not data_drift.is_drifted
