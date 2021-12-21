import logging

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.data_drift_detection_manager import DataDriftDetectionManagerInfo, \
    MultipleDatasetDataDriftDetectionManager
from src.pipeline.datasets.deployment_datasets import BankMarketingDeploymentDatasetPlus, BankMarketingDeploymentDataset
from src.pipeline.datasets.paths import BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
from src.pipeline.datasets.training_datasets import BankMarketingDataset, BankMarketingDatasetPlus
from src.pipeline.model.data_drift_models import BankMarketingDataDriftModel
from src.pipeline.preprocessing.preprocessor import Preprocessor

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level='INFO',
    datefmt='%d-%b-%y %H:%M:%S'
)


class TestDataDriftDetectionManager:
    def test_no_data_drift(self):
        bank_marketing_info = DataDriftDetectionManagerInfo(
            deployment_dataset=BankMarketingDataset(),
            deployment_dataset_plus=BankMarketingDatasetPlus(),
            training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH,
            training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
            training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
            preprocessor=Preprocessor(),
            model=BankMarketingDataDriftModel(),
        )
        data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=[bank_marketing_info])
        data_drift: DataDrift = data_drift_detection_manager.manage()[0]
        assert not data_drift.is_drifted
