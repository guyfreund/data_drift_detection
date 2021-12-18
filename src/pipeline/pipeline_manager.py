import os
from typing import List
import logging

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.training_datasets import BankMarketingDataset, GermanCreditDataset
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generation_manager import MultipleDatasetGenerationManager, \
    DataGenerationManagerInfo
from src.pipeline.data_drift_detection.data_drift_detection_manager import MultipleDatasetDataDriftDetectionManager, \
    DataDriftDetectionManagerInfo
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager, ModelTrainingManagerInfo
from src.pipeline.constants import PipelineMode
from src.pipeline.datasets.paths import GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH, GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH, \
    BANK_MARKETING_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH, BANK_MARKETING_CONCATENATED_DF, GERMAN_CREDIT_CONCATENATED_DF
from src.pipeline.datasets.deployment_datasets import BankMarketingDeploymentDataset, \
    BankMarketingDeploymentDatasetPlus, GermanCreditDeploymentDataset, GermanCreditDeploymentDatasetPlus
from src.pipeline.model.production_models import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class PipelineManager(IManager):
    def __init__(self, pipeline_mode: PipelineMode, data_drift_info_list: List[DataDriftDetectionManagerInfo],
                 training_info_list: List[ModelTrainingManagerInfo],
                 retraining_info_list: List[ModelTrainingManagerInfo],
                 data_generation_info_list: List[DataGenerationManagerInfo]
                 ):
        self._mode = pipeline_mode
        self._data_generation_manager = MultipleDatasetGenerationManager()  # TODO: init with info_list=data_generation_info_list
        self._data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=data_drift_info_list)
        self._model_training_manager = MultipleDatasetModelTrainingManager(info_list=training_info_list)
        self._retraining_info_list: List[ModelTrainingManagerInfo] = retraining_info_list
        self._model_retraining_manager = MultipleDatasetModelTrainingManager(info_list=self._retraining_info_list)
        self._data_drifts: List[DataDrift] = []

    def manage(self):
        if self._mode == PipelineMode.Training:
            # training all models
            self._model_training_manager.manage()

        elif self._mode == PipelineMode.Monitoring:
            # generating deployment datasets
            self._data_generation_manager.manage()

            # detecting data drifts in each of the deployment datasets
            self._data_drifts = self._data_drift_detection_manager.manage()

            # training only if a data drift was detected
            for idx, data_drift in enumerate(self._data_drifts):
                self._retraining_info_list[idx].to_train = data_drift.is_drifted
                self._model_retraining_manager.info_list = self._retraining_info_list  # no need, but more readable

            # retraining all models that a data drift has detected for their corresponding deployment dataset
            self._model_retraining_manager.manage()  # TODO: train by the result od data drifts

        else:
            # pipeline running mode is not supported
            raise NotImplementedError

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @mode.setter
    def mode(self, value: PipelineMode):
        self._mode = value


def prepare_model_training_info() -> List[ModelTrainingManagerInfo]:
    return [
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=BankMarketingDataset(),
            model=BankMarketingProductionModel()
        ),
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=GermanCreditDataset(),
            model=IModel()  # TODO: fix
        )
    ]


def prepare_data_generation_info() -> List[DataGenerationManagerInfo]:
    return [
        DataGenerationManagerInfo(),  # Bank Marketing
        DataGenerationManagerInfo()   # German Credit
    ]


def prepare_data_drift_info() -> List[DataDriftDetectionManagerInfo]:
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH)
    assert os.path.exists(BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH)

    return [
        DataDriftDetectionManagerInfo(
            deployment_dataset_plus=GermanCreditDeploymentDatasetPlus(),
            training_processed_df_plus_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH,
            preprocessor=Preprocessor(),
            model=IModel(),  # TODO: fix
            deployment_dataset=GermanCreditDeploymentDataset(),
            training_feature_metrics_list_path=GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH
        ),
        DataDriftDetectionManagerInfo(
            deployment_dataset_plus=BankMarketingDeploymentDatasetPlus(),
            training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
            preprocessor=Preprocessor(),
            model=BankMarketingProductionModel(),
            deployment_dataset=BankMarketingDeploymentDataset(),
            training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
        )
    ]


def prepare_model_retraining_info() -> List[ModelTrainingManagerInfo]:
    return [
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=Dataset.concatenate(
                dataset_list=[BankMarketingDataset(), BankMarketingDeploymentDataset()],
                path=BANK_MARKETING_CONCATENATED_DF
            ),
            model=BankMarketingProductionModel()
        ),
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=Dataset.concatenate(
                dataset_list=[GermanCreditDataset(), GermanCreditDeploymentDataset()],
                path=GERMAN_CREDIT_CONCATENATED_DF
            ),
            model=IModel()  # TODO: fix
        )
    ]


def main():
    # training
    pipeline_manager = PipelineManager(
        pipeline_mode=PipelineMode.Training,
        training_info_list=prepare_model_training_info(),
        data_generation_info_list=prepare_data_generation_info(),
        data_drift_info_list=prepare_data_drift_info(),
        retraining_info_list=prepare_model_retraining_info()
    )
    pipeline_manager.manage()

    # monitoring
    pipeline_manager.mode = PipelineMode.Monitoring
    for _ in range(10):
        pipeline_manager.manage()


if __name__ == '__main__':
    main()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.warning('This will get logged to a file')
