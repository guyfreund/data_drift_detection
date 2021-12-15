from typing import List

from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.datasets.dataset import Dataset


class ModelTrainingManagerInfo:
    def __init__(self, preprocessor: IPreprocessor, dataset: Dataset, model: IModel):
        self.preprocessor: IPreprocessor = preprocessor
        self.training_dataset: Dataset = dataset
        self.model: IModel = model


class ModelTrainingManager(IManager):
    def __init__(self, info: ModelTrainingManagerInfo):
        self.info = info

    def manage(self):
        processed_dataset, processed_dataset_plus, feature_metrics_list = self.info.preprocessor.preprocess(self.info.training_dataset)
        X_train, X_validation, X_test, y_train, y_validation, y_test = self.info.preprocessor.split(processed_dataset)
        self.info.model.train(X_train, y_train)
        self.info.model.tune_hyperparameters(X_validation, y_validation)
        self.info.model.evaluate(X_test, y_test)
        self.info.model.load(self.info.model.__class__.__name__)
        return self.info.model


class MultipleDatasetModelTrainingManager(IManager):
    def __init__(self, info_list: List[ModelTrainingManagerInfo]):
        self.model_training_managers: List[ModelTrainingManager] = [ModelTrainingManager(info) for info in info_list]

    def manage(self) -> List[IModel]:
        models: List[IModel] = [manager.manage() for manager in self.model_training_managers]
        return models
