from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd

from src.pipeline.preprocessing.interfaces.ifeature_metric import IFeatureMetric
from src.pipeline.datasets.dataset import Dataset

class IPreprocessor(ABC):
  """
  Interface for a preprocessor object
  """

  @abstractmethod
  def preprocess(self, dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetric]]:
    """ preprocesses the raw dataset
        saves the processed data frame in self._processed_df
        saves the processes data as a pickle

    Args:
        dataset (Dataset): The raw dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetric]]:
        X (pd.DataFrame): the processesed data frame
        X+ (pd.DataFrame): the processed data frame with the addition of the DatasetType column for all instances
        feature_metrics_list (List[IFeatureMetric]]): a list of IFeatureMetric objects per feature
    """
    raise NotImplementedError

  @abstractmethod
  def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ split the processed dataset into X_train, X_validation, X_test
        saves the sets in self._X_train, self._X_validation, self._X_test
        save X_train, X_validation, X_test as a pickle

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train (pd.DataFrame): the training set
        X_validation (pd.DataFrame): the validation set
        X_test (pd.DataFrame): the test set
    """
    raise NotImplementedError

  @property
  def processed_df(self) -> pd.DataFrame:
    raise NotImplementedError

  @property
  def X_train(self) -> pd.DataFrame:
    raise NotImplementedError

  @property
  def X_validation(self) -> pd.DataFrame:
    raise NotImplementedError

  @property
  def X_test(self) -> pd.DataFrame:
    raise NotImplementedError