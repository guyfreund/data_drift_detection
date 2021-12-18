from unittest import TestCase
import logging

from src.pipeline.datasets.training_datasets import BankMarketingDataset, GermanCreditDataset
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager, ModelTrainingManagerInfo
from src.pipeline.model.production_models import BankMarketingProductionModel, GermanCreditProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class TestBankMarketingDatasetModelTrainingManager(TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level='INFO',
            datefmt='%d-%b-%y %H:%M:%S'
        )

        self.bank_marketing_info = ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=BankMarketingDataset(),
            model=BankMarketingProductionModel(),
        )

    def test_manage(self):
        info_list, feature_metrics_list_of_lists, model_metrics_list = MultipleDatasetModelTrainingManager([self.bank_marketing_info]).manage()
        for info, feature_metrics_list, model_metrics_dict in zip(info_list, feature_metrics_list_of_lists, model_metrics_list):
            self.assertEqual(model_metrics_dict, info.model.model_metrics)


class TestGermanCreditDatasetModelTrainingManager(TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level='INFO',
            datefmt='%d-%b-%y %H:%M:%S'
        )

        self.german_credit_info = ModelTrainingManagerInfo(
            preprocessor=Preprocessor(),
            dataset=GermanCreditDataset(),
            model=GermanCreditProductionModel(),
        )

    def test_manage(self):
        info_list, feature_metrics_list_of_lists, model_metrics_list = MultipleDatasetModelTrainingManager([self.german_credit_info]).manage()
        for info, feature_metrics_list, model_metrics_dict in zip(info_list, feature_metrics_list_of_lists, model_metrics_list):
            self.assertEqual(model_metrics_dict, info.model.model_metrics)
