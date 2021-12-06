from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, List, Dict
from pipeline.model.interfaces.imodel_metric import IModelMetric
from src.pipeline.model.constants import ModelMetricType

class IModel(ABC):
  """
  Interface for a model object
  """

  @abstractmethod
  def train(self, X_train: pd.DataFrame):
    """ trains the model
        keep track on model training (loss, etc...)
        saves the trained model as a pickle
        saves the trained model in self._model

    Args:
        X_train (pd.DataFrame): the training dataset

    """
    raise NotImplementedError

  @abstractmethod
  def tune_hyperparameters(self, X_validation: pd.DataFrame):
    """ tunes the parameters of the model
        keep track on model training (loss, etc...)
        saves the tuned model as a pickle
        saves the tuned model in self._model

    Args:
        X_validation (pd.DataFrame): the validation set

    """
    raise NotImplementedError

  @abstractmethod
  def evaluate(self, X_test: pd.DataFrame) -> List[ModelMetricType]:
    """ evaluates the model
        prints the results
        saves the list of model metric to self._model_metrics as a Dict[ModelMetricType, IModelMetric]

    Args:
        X_test (pd.DataFrame): the test dataset

    Returns:
        List[ModelMetricType]: List of metrics per the trained model
    """
    raise NotImplementedError

  @property
  def model(self) -> Any:
    raise NotImplementedError

  @property
  def model_metrics(self) -> Dict[ModelMetricType, IModelMetric]:
    raise NotImplementedError