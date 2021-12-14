from unittest import TestCase

from src.pipeline.datasets.training_datasets import BankMarketingDataset
from src.pipeline.model.model_trainining_manager import MultipleDatasetModelTrainingManager
from src.pipeline.model.production_models import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor import Preprocessor


class TestMultipleDatasetModelTrainingManager(TestCase):
    def setUp(self) -> None:
        self.manager = MultipleDatasetModelTrainingManager(Preprocessor(),
                                                           BankMarketingDataset(),
                                                           BankMarketingProductionModel())

    def test_manage(self):
        self.manager.manage()
