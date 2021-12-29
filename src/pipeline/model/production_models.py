from xgboost import XGBClassifier
from typing import Dict, Any
from sklearn import metrics
import pandas as pd
import pickle
import os

from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.model.model_metrics import Accuracy, Precision, Recall, F1, AUC
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import logger

logging = logger.get_logger(__name__)


class BankMarketingProductionModel(IModel):
    def __init__(self):
        self._model = XGBClassifier()
        self._is_trained = False
        self._is_tuned = False
        self._model_metrics: Dict[ModelMetricType, IModelMetric] = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self._model.fit(X_train, y_train)
        self._is_trained = True

        logging.info(f'Train Model: model was trained successfully')

        self._save_model_as_pickle(self.__class__.__name__)

    def _save_model_as_pickle(self, model_class_name: str):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{model_class_name}.sav"))
        with open(path, 'wb') as handle:
            pickle.dump(self._model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f'Save Data: {model_class_name}.save model has been saved')

    def tune_hyperparameters(self, X_validation: pd.DataFrame, y_validation: pd.DataFrame):
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[ModelMetricType, IModelMetric]:
        y_pred = self._model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)

        logging.info(f'Model Evaluation: model was evaluated successfully. ')
        logging.info(f'Evaluation Info: accuracy: {round(accuracy * 100, 2)}% | '
                      f'precision: {round(precision * 100, 2)}% | '
                      f'recall: {round(recall * 100, 2)}% | '
                      f'f1: {round(f1 * 100, 2)}% | '
                      f'auc: {round(auc, 2)}')

        self._model_metrics = {
            ModelMetricType.Accuracy: Accuracy(value=accuracy),
            ModelMetricType.Precision: Precision(value=precision),
            ModelMetricType.Recall: Recall(value=recall),
            ModelMetricType.F1: F1(value=f1),
            ModelMetricType.AUC: AUC(value=auc)
        }

        return self._model_metrics

    def load(self, model_class_name: str):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{model_class_name}.sav"))
        with open(path, 'rb') as handle:
            self._model = pickle.load(handle)

        logging.info(f'Load Data: {model_class_name}.sav model has been loaded')

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


class GermanCreditProductionModel(IModel):
    def __init__(self):
        self._model = XGBClassifier(**{'n_estimators': 3000,
                                       'objective': 'binary:logistic',
                                       'learning_rate': 0.005,
                                       'subsample': 0.555,
                                       'colsample_bytree': 0.7,
                                       'min_child_weight': 3,
                                       'max_depth': 8,
                                       'n_jobs': -1})
        self._is_trained = False
        self._is_tuned = False
        self._model_metrics: Dict[ModelMetricType, IModelMetric] = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self._model.fit(X_train, y_train, eval_metric='auc')
        self._model.set_params(**{'n_estimators': self._model.best_ntree_limit})
        self._model.fit(X_train, y_train)
        self._is_trained = True

        logging.info(f'Train Model: model was trained successfully')

        self._save_model_as_pickle(self.__class__.__name__)

    def _save_model_as_pickle(self, model_class_name: str):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{model_class_name}.sav"))
        with open(path, 'wb') as handle:
            pickle.dump(self._model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f'Save Data: {model_class_name}.save model has been saved')

    def tune_hyperparameters(self, X_validation: pd.DataFrame, y_validation: pd.DataFrame):
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[ModelMetricType, IModelMetric]:
        y_pred = self._model.predict(X_test, ntree_limit=self._model.best_ntree_limit)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)

        logging.info(f'Model Evaluation: model was evaluated successfully. results are as follows: '
                      f'Accuracy: {round(accuracy * 100, 2)}%, '
                      f'Precision: {round(precision * 100, 2)}%, '
                      f'Recall: {round(recall * 100, 2)}%, '
                      f'F1: {round(f1 * 100, 2)}%, '
                      f'AUC: {round(auc, 2)}')

        self._model_metrics = {
            ModelMetricType.Accuracy: Accuracy(value=accuracy),
            ModelMetricType.Precision: Precision(value=precision),
            ModelMetricType.Recall: Recall(value=recall),
            ModelMetricType.F1: F1(value=f1),
            ModelMetricType.AUC: AUC(value=auc)
        }

        return self._model_metrics

    def load(self, model_class_name: str):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{model_class_name}.sav"))
        with open(path, 'rb') as handle:
            self._model = pickle.load(handle)

        logging.info(f'Load Data: {model_class_name}.sav model has been loaded')

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


class BankMarketingRetrainedProductionModel(BankMarketingProductionModel):
    def __init__(self):
        super().__init__()


class GermanCreditRetrainedProductionModel(GermanCreditProductionModel):
    def __init__(self):
        super().__init__()
