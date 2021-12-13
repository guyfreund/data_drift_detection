import argparse
import os
from typing import List

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generation_manager import DataGenerationManager
from src.pipeline.data_drift_detection.data_drift_detection_manager import MultipleDatasetDataDriftDetectionManager, \
    DataDriftDetectionManagerInfo
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.model_trainining_manager import ModelTrainingManager
from src.pipeline.constants import PipelineMode
from src.pipeline.datasets.paths import GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH, GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PATH, BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH, \
    BANK_MARKETING_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor


class PipelineManager(IManager):
    def __init__(self, pipeline_mode: PipelineMode, data_drift_info_list: List[DataDriftDetectionManagerInfo]):
        self._mode = pipeline_mode
        self._data_generation_manager = DataGenerationManager()
        self._multiple_dataset_data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=data_drift_info_list)
        self._model_training_manager = ModelTrainingManager()
        self._data_drifts: List[DataDrift] = []

    def manage(self):
        if self._mode == PipelineMode.Training:
            self._model_training_manager.manage()
        elif self._mode == PipelineMode.Monitoring:
            self._data_generation_manager.manage()
            self._data_drifts = self._multiple_dataset_data_drift_detection_manager.manage()
            # TODO: retrain
        else:
            raise NotImplementedError


def args_handler():
    parser = argparse.ArgumentParser(description='Running the pipeline manager')
    parser.add_argument('-m', '--mode', default=PipelineMode.Training, type=int, help='Pipeline mode: 0=Training, 1=Monitoring')
    return parser.parse_args()


def prepare_data_drift_config() -> List[DataDriftDetectionManagerInfo]:
    assert os.path.exists(GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH)
    assert os.path.exists(GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH)
    assert os.path.exists(BANK_MARKETING_DEPLOYMENT_DATASET_PATH)
    assert os.path.exists(BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH)

    german_credit_info = DataDriftDetectionManagerInfo(
        deployment_dataset_plus=Dataset(dtype=DatasetType.Deployment, path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH),  # TODO: move to specific dataset
        training_processed_df_plus_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH,
        preprocessor=IPreprocessor(),  # TODO: fix
        model=IModel(),  # TODO: fix
        deployment_dataset=Dataset(dtype=DatasetType.Deployment, path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH),  # TODO: move to specific dataset
        training_feature_metrics_list_path=GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH,
        training_processed_df_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH
    )

    bank_marketing_info = DataDriftDetectionManagerInfo(
        deployment_dataset_plus=Dataset(dtype=DatasetType.Deployment, path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH),  # TODO: move to specific dataset
        training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
        preprocessor=IPreprocessor(),  # TODO: fix
        model=IModel(),  # TODO: fix
        deployment_dataset=Dataset(dtype=DatasetType.Deployment, path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH),  # TODO: move to specific dataset
        training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
        training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
    )

    return [german_credit_info, bank_marketing_info]


def main():
    args = args_handler()
    mode = args.mode

    pipeline_manager = PipelineManager(
        pipeline_mode=mode,
        data_drift_info_list=prepare_data_drift_config()
    )
    pipeline_manager.manage()


if __name__ == '__main__':
    main()
