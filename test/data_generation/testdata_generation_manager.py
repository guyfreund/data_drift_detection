import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from typing import List
from src.pipeline.config import Config
from src.pipeline.model.paths import GERMAN_CREDIT_GEN_CGAN_MODEL_PATH
from src.pipeline.datasets.training_datasets import GermanCreditDataset, BankMarketingDataset
from src.pipeline.datasets.paths import GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, \
    GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH, BANK_MARKETING_DEPLOYMENT_DATASET_PATH, BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH
from src.pipeline.data_generation.data_generation_manager import DataGenerationManagerInfo, \
    MultipleDatasetGenerationManager, DataGenerationManager
from src.pipeline.preprocessing.paths import BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT, GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT
from src.pipeline.preprocessing.label_preprocessor import LabelProcessor
from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.data_generation.data_generation_manager import MultipleDatasetGenerationManager
# class _TestDatagenerationManager:

def prepare_data_generation_info() -> List[DataGenerationManagerInfo]:
    use_gan: bool = Config().data_generation.use_gan
    use_smote: bool = Config().data_generation.use_smote
    assert use_gan != use_smote
    bank_marketing_dataset = BankMarketingDataset()
    german_credit_dataset = GermanCreditDataset()
    return [
        DataGenerationManagerInfo(
            origin_dataset=bank_marketing_dataset,
            model_class=None,  # for GAN
            sample_size_to_generate=Config().data_generation.generation_percent,
            model_path=None,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=BANK_MARKETING_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH,
            processor=LabelProcessor(bank_marketing_dataset, BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT)
        ),  # Bank Marketing
        DataGenerationManagerInfo(
            origin_dataset=german_credit_dataset,
            model_class=None,
            sample_size_to_generate=Config().data_generation.generation_percent,
            model_path=None, # for GAN
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH,
            processor=LabelProcessor(german_credit_dataset, GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT)
        )   # German Credit
    ]


info_lst = prepare_data_generation_info()
manager = MultipleDatasetGenerationManager(info_lst)