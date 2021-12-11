from unittest import TestCase

from src.pipeline.datasets.dataset_training_bank_marketing import BankMarketingDataset
from src.pipeline.preprocessing.preprocessor.preprocessor import Preprocessor


class TestPreprocessor(TestCase):
    def setUp(self) -> None:
        pass

    def test_preprocess(self):
        preprocessor = Preprocessor()
        dataset = BankMarketingDataset()
        raw_df, raw_df, feature_metrics = preprocessor.preprocess(dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = preprocessor.split()
        print("finished")