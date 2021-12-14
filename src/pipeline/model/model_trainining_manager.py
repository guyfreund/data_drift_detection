from src.pipeline.datasets.training_datasets import BankMarketingDataset
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.preprocessing.preprocessor import Preprocessor
from src.pipeline.datasets.dataset import Dataset


class ModelTrainingManagerInfo:
    pass


class ModelTrainingManager(IManager):
    pass


class MultipleDatasetModelTrainingManager(IManager):
    def __init__(self, preprocessor: IPreprocessor, dataset: Dataset, model: IModel):

        self.preprocessor = preprocessor
        self.dataset = dataset
        self.model = model

    def manage(self):
        processed_dataset, processed_dataset_plus, feature_metrics_list = self.preprocessor.preprocess(self.dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = self.preprocessor.split(processed_dataset)
        self.model.train(X_train, y_train)
        self.model.tune_hyperparameters(X_validation, y_validation)
        self.model.evaluate(X_test, y_test)
        self.model.load(self.model.__class__.__name__)
