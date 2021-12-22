import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from src.pipeline.model.paths import GERMAN_CREDIT_GEN_CGAN_MODEL_PATH
from src.pipeline.datasets.training_datasets import GermanCreditDataset
from src.pipeline.datasets.paths import GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH, GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
from src.pipeline.data_generation.data_generation_manager import DataGenerationManagerInfo, \
    MultipleDatasetGenerationManager, DataGenerationManager


class _TestDatagenerationManager:

    def __init__(self):
        # self._german_credit_origin_data = GermanCreditDataset(),
        self._bank_marketing_info = DataGenerationManagerInfo(
            origin_dataset=GermanCreditDataset(),
            model_class=CGAN,
            sample_size_to_generate=100,
            model_path=GERMAN_CREDIT_GEN_CGAN_MODEL_PATH,
            data_drift_types=[],
            save_data_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH,
            save_data_plus_path=GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH
        )
        # self._german_credit_origin_data = pd.read_pickle(GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH)
        # self._bank_marketing_info = DataGenerationManagerInfo(
        #     origin_dataset=self._origin_data,
        #     model_class=CGAN,
        #     sample_size_to_generate=100,
        #     model_path=self._model_path,
        #     data_drift_types=[],
        #     save_data_path=self._save_data_path,
        #     save_data_plus_path=self._save_data_plus_path
        # )


    def _test_multiple_data_generation_manager(self):
        data_generation_managers = MultipleDatasetGenerationManager(info_list=[self._bank_marketing_info])
        res = data_generation_managers.manage()


    def _test_data_generation_manager(self):
        data_generation_managers = DataGenerationManager(self._bank_marketing_info)
        res = data_generation_managers.manage()



test_manager = _TestDatagenerationManager()
test_manager._test_data_generation_manager()
test_manager._test_multiple_data_generation_manager()
