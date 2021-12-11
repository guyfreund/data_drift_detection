from typing import Dict, Any
import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics
import pickle
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
import matplotlib as plt

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.model.paths import DEFAULT_SAVED_MODEL_PATH


class BankMarketingProductionModel(IModel):
    def __init__(self):
        self._model = XGBClassifier()

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self._model.fit(X_train, y_train)
        with open(DEFAULT_SAVED_MODEL_PATH, 'wb') as handle:
            pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("... save model ...")

    def tune_hyperparameters(self, X_validation: pd.DataFrame, y_validation: pd.DataFrame):
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[ModelMetricType, IModelMetric]:
        y_pred = self._model.predict(X_test)

        print(f'test accuracy score is: {round(metrics.accuracy_score(y_test, y_pred), 2)}%')

        model_metric = IModelMetric()
        metric_type = ModelMetricType(0)

        return {metric_type: model_metric}

    def load(self, path: str):
        with open(path, 'rb') as handle:
            self._model = pickle.load(handle)
        print("... load model ...")

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_metrics(self) -> Dict[ModelMetricType, IModelMetric]:
        return self._model_metrics

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_tuned(self) -> bool:
        return self._is_tuned


