from typing import List
from unittest import TestCase

from src.pipeline.datasets.training_datasets import BankMarketingDataset
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager, ModelTrainingManagerInfo
from src.pipeline.model.production_models import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class TestMultipleDatasetModelReTrainingManager(TestCase):
    def setUp(self) -> None:
        self.info_list: List[ModelTrainingManagerInfo] = [
            ModelTrainingManagerInfo(
                preprocessor=Preprocessor(),
                dataset=BankMarketingDataset(),
                model=BankMarketingProductionModel()
            ),
            ModelTrainingManagerInfo(
                preprocessor=Preprocessor(),
                dataset=BankMarketingDataset(),
                model=BankMarketingProductionModel(),
                to_train=False
            )
        ]

    def test_manage(self):
        info_list, feature_metrics_list_of_lists, model_metrics_list = MultipleDatasetModelTrainingManager(self.info_list).manage()
        assert len(info_list) == 1
        assert len(feature_metrics_list_of_lists) == 1
        assert len(model_metrics_list) == 1
        for info, feature_metrics_list, model_metrics_dict in zip(info_list, feature_metrics_list_of_lists, model_metrics_list):
            self.assertEqual(model_metrics_dict, info.model.model_metrics)
