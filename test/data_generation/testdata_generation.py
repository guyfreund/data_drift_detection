import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from src.pipeline.model.paths import GERMAN_CREDIT_GEN_CGAN_MODEL_PATH
from src.pipeline.datasets.training_datasets import GermanCreditDataset
from src.pipeline.datasets.paths import SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,\
    SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
from src.pipeline.data_generation.data_generation_manager import DataGenerationManagerInfo, \
    MultipleDatasetGenerationManager, DataGenerationManager
from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.preprocessing.label_preprocessor import LabelProcessor
from src.pipeline.preprocessing.paths import BANK_MARKETING_LABEL_ENCODER_PATH, GERMAN_CREDIT_LABEL_ENCODER_PATH


class TestGANDatageneration:

    def __init__(self):
        # self._german_credit_origin_data = GermanCreditDataset(),
        dataset = GermanCreditDataset()
        self._german_credit_info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=CGAN,
            sample_size_to_generate=100,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH,
            processor=LabelProcessor(dataset, GERMAN_CREDIT_LABEL_ENCODER_PATH)
        )

    def _test_data_normal_generation(self):
        data_generation_manager = DataGenerationManager(self._bank_marketing_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        # data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        data_generation_manager = DataGenerationManager(self._bank_marketing_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        return data_generation_manager, generated_data


class TestSMOTENCDatageneration:

    def __init__(self):
        dataset = GermanCreditDataset()
        self._german_credit_info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=None,
            sample_size_to_generate=100,
            model_path=None,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH,
            processor=LabelProcessor(dataset, GERMAN_CREDIT_LABEL_ENCODER_PATH)
        )

    def _test_data_normal_generation(self):
        data_generation_manager = DataGenerationManager(self._german_credit_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        # data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        data_generation_manager = DataGenerationManager(self._german_credit_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        return data_generation_manager, generated_data

# SMOTE
test_manager = TestSMOTENCDatageneration()
data_generation_manager, generated_data = test_manager._test_data_normal_generation()
print('Succeed generate normal data')
data_generation_manager_drift, generated_data_drift = test_manager._test_data_drift_generation()
print('Succeed generate drifted data')

# GAN
# test_manager = TestGANDatageneration()
# data_generation_manager, generated_data = test_manager._test_data_normal_generation()
# print('Succeed generate normal data')
# data_generation_manager, generated_data = test_manager._test_data_drift_generation()
# print('Succeed generate drifted data')
#

