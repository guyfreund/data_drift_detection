from src.pipeline.datasets.dataset_training_bank_marketing import BankMarketingDataset
from src.pipeline.preprocessing.preprocessor.preprocessor import Preprocessor


class TestPreprocessor:
    def setUp(self) -> None:
        pass

    def test_preprocess(self):
        preprocessor = Preprocessor()
        dataset = BankMarketingDataset()
        processed_dataset, processed_dataset_plus, feature_metrics_list = preprocessor.preprocess(dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = preprocessor.split(processed_dataset)
        print("finished")
