from unittest import TestCase

from src.pipeline.datasets.training_datasets import BankMarketingDataset
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
        ModelTrainingManager(self.bank_marketing_info).manage()
