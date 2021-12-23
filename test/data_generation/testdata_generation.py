import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from src.pipeline.model.paths import GERMAN_CREDIT_GEN_CGAN_MODEL_PATH
from src.pipeline.datasets.training_datasets import GermanCreditDataset
from src.pipeline.datasets.paths import GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
from src.pipeline.data_generation.data_generation_manager import DataGenerationManagerInfo, \
    MultipleDatasetGenerationManager, DataGenerationManager
from src.pipeline.data_drift_detection.constants import DataDriftType


class _TestDatageneration:

    def __init__(self):
        # self._german_credit_origin_data = GermanCreditDataset(),
        self._bank_marketing_info = DataGenerationManagerInfo(
            origin_dataset=GermanCreditDataset(),
            model_class=CGAN,
            sample_size_to_generate=100,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[DataDriftType.Statistical, DataDriftType.NumNulls],
            save_data_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
        )

    def _test_data_normal_generation(self):
        data_generation_manager = DataGenerationManager(self._bank_marketing_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=False)
        data_generation_manager._save_data_as_pickle(generated_data)
        return data_generation_manager, generated_data

    def _test_data_drift_generation(self):
        data_generation_manager = DataGenerationManager(self._bank_marketing_info)
        generated_data = data_generation_manager._get_generated_dataset(is_drifted=True)
        return data_generation_manager, generated_data


test_manager = _TestDatageneration()
data_generation_manager, generated_data = test_manager._test_data_drift_generation()
print('Succeed generate normal data')
data_generation_manager, generated_data = test_manager._test_data_drift_generation()
print('Succeed generate drifted data')
