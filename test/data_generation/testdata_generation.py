import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
# models
from src.pipeline.model.paths import GERMAN_CREDIT_GEN_CGAN_MODEL_PATH, BANK_MARKETING_GEN_CGAN_MODEL_PATH
# data
from src.pipeline.datasets.training_datasets import GermanCreditDataset, BankMarketingDataset
from src.pipeline.datasets.paths import *

from src.pipeline.data_generation.data_generation_manager import DataGenerationManagerInfo, \
    MultipleDatasetGenerationManager, DataGenerationManager
from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.preprocessing.label_preprocessor import LabelProcessor
from src.pipeline.preprocessing.paths import BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT, GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT


class GermanCreditTestGANDatageneration:

    def __init__(self):
        # self._german_credit_origin_data = GermanCreditDataset(),
        dataset = GermanCreditDataset()
        self._info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=CGAN,
            sample_size_to_generate=100,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=None,
            save_data_plus_path=None,
            processor=LabelProcessor(dataset, GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT)
        )

    def _test_data_normal_generation(self):
        self._info.save_data_path = GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_NORMAL
        self._info.save_data_plus_path = GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        self._info.save_data_path = GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_DRIFT
        self._info.save_data_plus_path = GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data


class GermanCreditTestSMOTENCDatageneration:

    def __init__(self):
        dataset = GermanCreditDataset()
        self._info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=None,
            sample_size_to_generate=100,
            model_path=None,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=None,
            save_data_plus_path=None,
            processor=LabelProcessor(dataset, GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT)
        )

    def _test_data_normal_generation(self):
        self._info.save_data_path = SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_NORMAL
        self._info.save_data_plus_path = SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        self._info.save_data_path = SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_DRIFT
        self._info.save_data_plus_path = SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data


class BankMarketingTestGANDatageneration:

    def __init__(self):
        # self._german_credit_origin_data = GermanCreditDataset(),
        dataset = BankMarketingDataset()
        self._info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=CGAN,
            sample_size_to_generate=100,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=None,
            save_data_plus_path=None,
            processor=LabelProcessor(dataset, BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT)
        )

    def _test_data_normal_generation(self):
        self._info.save_data_path = GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_NORMAL
        self._info.save_data_plus_path = GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        self._info.save_data_path = GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_DRIFT
        self._info.save_data_plus_path = GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data


class BankMarketingTestSMOTENCDatageneration:

    def __init__(self):
        dataset = BankMarketingDataset()
        self._info = DataGenerationManagerInfo(
            origin_dataset=dataset,
            model_class=None,
            sample_size_to_generate=100,
            model_path=None,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=None,
            save_data_plus_path=None,
            processor=LabelProcessor(dataset, BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT)
        )

    def _test_data_normal_generation(self):
        self._info.save_data_path = SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_NORMAL
        self._info.save_data_plus_path = SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        self._info.save_data_path = SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_DRIFT
        self._info.save_data_plus_path = SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT
        data_generation_manager = DataGenerationManager(self._info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data



# SMOTE
test_manager = BankMarketingTestSMOTENCDatageneration()
data_generation_manager, generated_data = test_manager._test_data_normal_generation()
print('Bank Marketing: Succeed generate normal dataset using SMOTENC')
data_generation_manager_drift, generated_data_drift = test_manager._test_data_drift_generation()
print('Bank Marketing: Succeed generate drifted dataset using SMOTENC')

test_manager = GermanCreditTestSMOTENCDatageneration()
data_generation_manager, generated_data = test_manager._test_data_normal_generation()
print('German Credit: Succeed generate normal data')
data_generation_manager_drift, generated_data_drift = test_manager._test_data_drift_generation()
print('German Credit: Succeed generate drifted data')


# GAN
# test_manager = TestGANDatageneration()
# data_generation_manager, generated_data = test_manager._test_data_normal_generation()
# print('Succeed generate normal data')
# data_generation_manager, generated_data = test_manager._test_data_drift_generation()
# print('Succeed generate drifted data')
#

