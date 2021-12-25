import os
from typing import List
import logging

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.training_datasets import BankMarketingDataset, GermanCreditDataset
from src.pipeline.evaluation.evaluation_manager import EvaluationManagerInfo, MultipleDatasetEvaluationManager
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generation_manager import MultipleDatasetGenerationManager, \
    DataGenerationManagerInfo
from src.pipeline.data_drift_detection.data_drift_detection_manager import MultipleDatasetDataDriftDetectionManager, \
    DataDriftDetectionManagerInfo
from src.pipeline.model.data_drift_models import BankMarketingDataDriftModel, GermanCreditDataDriftModel
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager, ModelTrainingManagerInfo
from src.pipeline.constants import PipelineMode
from src.pipeline.datasets.paths import GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, \
    GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH, GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH, \
    BANK_MARKETING_TRAINING_PROCESSED_DF_PATH, BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH, \
    BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH, BANK_MARKETING_CONCATENATED_DF, GERMAN_CREDIT_CONCATENATED_DF, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
from src.pipeline.datasets.deployment_datasets import BankMarketingDeploymentDataset, \
    BankMarketingDeploymentDatasetPlus, GermanCreditDeploymentDataset, GermanCreditDeploymentDatasetPlus, \
    BankMarketingSampledDeploymentDataset, GermanCreditSampledDeploymentDataset
from src.pipeline.model.production_models import BankMarketingProductionModel, GermanCreditProductionModel, \
    BankMarketingRetrainedProductionModel, GermanCreditRetrainedProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor
from src.pipeline.config import Config
from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.model.paths import BANK_MARKETING_GEN_CGAN_MODEL_PATH, GERMAN_CREDIT_GEN_CGAN_MODEL_PATH


class PipelineManager(IManager):
    def __init__(self, pipeline_mode: PipelineMode, data_drift_info_list: List[DataDriftDetectionManagerInfo],
                 training_info_list: List[ModelTrainingManagerInfo], retraining_info_list: List[ModelTrainingManagerInfo],
                 data_generation_info_list: List[DataGenerationManagerInfo], evaluation_info_list: List[EvaluationManagerInfo]):
        self._mode = pipeline_mode
        self._data_generation_manager = MultipleDatasetGenerationManager(info_list=data_generation_info_list)
        self._data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=data_drift_info_list)
        self._model_training_manager = MultipleDatasetModelTrainingManager(info_list=training_info_list)
        self._retraining_info_list: List[ModelTrainingManagerInfo] = retraining_info_list
        self._model_retraining_manager = MultipleDatasetModelTrainingManager(info_list=self._retraining_info_list)
        self._evaluation_info_list = evaluation_info_list
        self._evaluation_manager = MultipleDatasetEvaluationManager()
        self._data_drifts: List[DataDrift] = []
        self._detected_data_drifts: List[DataDrift] = []

    def manage(self):
        if self._mode == PipelineMode.Training:
            # training all models
            self._model_training_manager.manage()

        elif self._mode == PipelineMode.Monitoring:
            # generating deployment datasets
            self._data_drifts = self._data_generation_manager.manage()

            # detecting data drifts in each of the deployment datasets
            self._detected_data_drifts = self._data_drift_detection_manager.manage()

            # training only if a data drift was detected
            for idx, data_drift in enumerate(self._detected_data_drifts):
                self._retraining_info_list[idx].to_train = data_drift.is_drifted
                self._evaluation_info_list[idx].to_evaluate = data_drift.is_drifted

            # retraining all models that a data drift has detected for their corresponding deployment dataset
            self._model_retraining_manager.manage()

            # evaluation
            self._evaluation_manager.manage()

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
            model=GermanCreditProductionModel()
        )
    ]


def prepare_data_generation_info() -> List[DataGenerationManagerInfo]:
    return [
        DataGenerationManagerInfo(
            origin_dataset=BankMarketingDataset(),
            model_class=Config().data_generation.generate_model_class,
            sample_size_to_generate=Config().data_generation.generation_percent,
            model_path=BANK_MARKETING_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH
        ),  # Bank Marketing
        DataGenerationManagerInfo(
            origin_dataset=GermanCreditDataset(),
            model_class=Config().data_generation.generate_model_class,
            sample_size_to_generate=Config().data_generation.generation_percent,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
        )   # German Credit
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
            model=GermanCreditDataDriftModel(),
            deployment_dataset=GermanCreditDeploymentDataset(),
            training_feature_metrics_list_path=GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH
        ),
        DataDriftDetectionManagerInfo(
            deployment_dataset_plus=BankMarketingDeploymentDatasetPlus(),
            training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
            preprocessor=Preprocessor(),
            model=BankMarketingDataDriftModel(),
            deployment_dataset=BankMarketingDeploymentDataset(),
            training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
        )
    ]


def prepare_model_retraining_info() -> List[ModelTrainingManagerInfo]:
    # TODO: think of weighting
    return [
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=Dataset.concatenate(
                dataset_list=[BankMarketingDataset(), BankMarketingSampledDeploymentDataset()],
                path=BANK_MARKETING_CONCATENATED_DF
            ),
            model=BankMarketingRetrainedProductionModel()
        ),
        ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=Dataset.concatenate(
                dataset_list=[GermanCreditDataset(), GermanCreditSampledDeploymentDataset()],
                path=GERMAN_CREDIT_CONCATENATED_DF
            ),
            model=GermanCreditRetrainedProductionModel()
        )
    ]


def prepare_evaluation_info():
    return [
        EvaluationManagerInfo(
            production_model=BankMarketingProductionModel(),
            retrained_production_model=BankMarketingRetrainedProductionModel(),
            preprocessor=Preprocessor(),
            training_dataset=BankMarketingDataset(),
            deployment_dataset=BankMarketingDeploymentDataset(),
            retraining_dataset=Dataset.concatenate(
                dataset_list=[BankMarketingDataset(), BankMarketingSampledDeploymentDataset()],
                path=BANK_MARKETING_CONCATENATED_DF
            )
        ),
        EvaluationManagerInfo(
            production_model=GermanCreditProductionModel(),
            retrained_production_model=GermanCreditRetrainedProductionModel(),
            preprocessor=Preprocessor(),
            training_dataset=GermanCreditDataset(),
            deployment_dataset=GermanCreditDeploymentDataset(),
            retraining_dataset=Dataset.concatenate(
                dataset_list=[GermanCreditDataset(), GermanCreditSampledDeploymentDataset()],
                path=BANK_MARKETING_CONCATENATED_DF
            )
        )
    ]


def run_pipeline_manager():
    # training
    pipeline_manager = PipelineManager(
        pipeline_mode=PipelineMode.Training,
        training_info_list=prepare_model_training_info(),
        data_generation_info_list=prepare_data_generation_info(),
        data_drift_info_list=prepare_data_drift_info(),
        retraining_info_list=prepare_model_retraining_info(),
        evaluation_info_list=prepare_evaluation_info()
    )
    pipeline_manager.manage()

    # monitoring
    pipeline_manager.mode = PipelineMode.Monitoring
    for _ in range(10):
        pipeline_manager.manage()


if __name__ == '__main__':
    run_pipeline_manager()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.warning('This will get logged to a file')
