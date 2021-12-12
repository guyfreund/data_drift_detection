from src.pipeline.datasets.dataset_training_bank_marketing import BankMarketingDataset
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.model.production_model.production_model_bank_marketing import BankMarketingProductionModel
from src.pipeline.preprocessing.preprocessor.preprocessor import Preprocessor


class ModelTrainingManager(IManager):
    def manage(self):
        preprocessor = Preprocessor()
        dataset = BankMarketingDataset()
        processed_dataset, processed_dataset_plus, feature_metrics_list = preprocessor.preprocess(dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = preprocessor.split(processed_dataset)
        model = BankMarketingProductionModel()
        model.train(X_train, y_train)
        model.tune_hyperparameters(X_validation, y_validation)
        model.evaluate(X_test, y_test)
        model.load(model.__class__.__name__)
