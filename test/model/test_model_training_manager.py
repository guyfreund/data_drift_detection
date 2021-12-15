from unittest import TestCase

from src.pipeline.datasets.training_datasets import BankMarketingDataset
from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager, ModelTrainingManagerInfo, \
    ModelTrainingManager
from src.pipeline.model.production_models import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class TestMultipleDatasetModelTrainingManager(TestCase):
    def setUp(self) -> None:
        self.bank_marketing_info = ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=BankMarketingDataset(),
            model=BankMarketingProductionModel(),
        )

    def test_manage(self):
        info_list, feature_metrics_list_of_lists, model_metrics_list = MultipleDatasetModelTrainingManager([self.bank_marketing_info]).manage()
        for info, feature_metrics_list, model_metrics_dict in zip(info_list, feature_metrics_list_of_lists, model_metrics_list):
            self.assertEqual(model_metrics_dict, info.model.model_metrics)
