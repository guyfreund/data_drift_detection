import os
from typing import List

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generation_manager import MultipleDatasetGenerationManager
from src.pipeline.data_drift_detection.data_drift_detection_manager import MultipleDatasetDataDriftDetectionManager, \
    DataDriftDetectionManagerInfo
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager
from src.pipeline.constants import PipelineMode
from src.pipeline.datasets.paths import GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH, GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH, \
    BANK_MARKETING_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH
from src.pipeline.datasets.deployment_datasets import BankMarketingDeploymentDataset, \
    BankMarketingDeploymentDatasetPlus, GermanCreditDeploymentDataset, GermanCreditDeploymentDatasetPlus
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor


class PipelineManager(IManager):
    def __init__(self, pipeline_mode: PipelineMode, data_drift_info_list: List[DataDriftDetectionManagerInfo]):
        self._mode = pipeline_mode
        self._data_generation_manager = MultipleDatasetGenerationManager()
        self._data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=data_drift_info_list)
        self._model_training_manager = MultipleDatasetModelTrainingManager()
        self._data_drifts: List[DataDrift] = []

    def manage(self):
        if self._mode == PipelineMode.Training:
            self._model_training_manager.manage()
        elif self._mode == PipelineMode.Monitoring:
            self._data_generation_manager.manage()
            self._data_drifts = self._data_drift_detection_manager.manage()
            # TODO: retrain
        else:
            raise NotImplementedError

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @mode.setter
    def mode(self, value: PipelineMode):
        self._mode = value


def prepare_data_drift_config() -> List[DataDriftDetectionManagerInfo]:
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH)

    german_credit_info = DataDriftDetectionManagerInfo(
        deployment_dataset_plus=GermanCreditDeploymentDatasetPlus(),
        training_processed_df_plus_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH,
        preprocessor=IPreprocessor(),  # TODO: fix
        model=IModel(),  # TODO: fix
        deployment_dataset=GermanCreditDeploymentDataset(),
        training_feature_metrics_list_path=GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH,
        training_processed_df_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH
    )

    bank_marketing_info = DataDriftDetectionManagerInfo(
        deployment_dataset_plus=BankMarketingDeploymentDatasetPlus(),
        training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
        preprocessor=IPreprocessor(),  # TODO: fix
        model=IModel(),  # TODO: fix
        deployment_dataset=BankMarketingDeploymentDataset(),
        training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
        training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
    )

    return [german_credit_info, bank_marketing_info]


def main():
    # training
    pipeline_manager = PipelineManager(
        pipeline_mode=PipelineMode.Training,
        data_drift_info_list=prepare_data_drift_config()
    )
    pipeline_manager.manage()

    # monitoring
    pipeline_manager.mode = PipelineMode.Monitoring
    for _ in range(10):
        pipeline_manager.manage()


if __name__ == '__main__':
    main()
