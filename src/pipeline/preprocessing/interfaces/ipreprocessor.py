from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd

from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.datasets.dataset import Dataset


class IPreprocessor(ABC):
    """
    Interface for a preprocessor object
    """

    @abstractmethod
    def preprocess(self, dataset: Dataset, generate_dataset_plus: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:
        """ preprocesses the raw dataset
        saves the processed data frame in self._processed_df
        saves the processed dataset as a pickle
        saves the processed dataset plus as a pickle
        saves feature_metrics_list as a pickle

        Args:
            dataset (Dataset): The raw dataset
            generate_dataset_plus (bool): Whether to generate dataset plus or not.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:
            processed_dataset (pd.DataFrame): the processed data frame
            processed_dataset_plus (pd.DataFrame): the processed data frame with the addition of the DatasetType column for all instances
            feature_metrics_list (List[IFeatureMetrics]): a list of IFeatureMetrics objects per feature
        """
        raise NotImplementedError

    def postprocess(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """ perform post process on processed df
        need to be able to process X, y or X+y

        Args:
            processed_df:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def split(self, processed_df: pd.DataFrame, label_column_name: str, dataset_class_name: str = '', dump: bool = True) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ split the processed dataset into X_train, X_validation, X_test
        saves the sets in self._X_train, self._X_validation, self._X_test
        save X_train, X_validation, X_test as a pickle
        saves the sets in self._y_train, self._y_validation, self._y_test
        save y_train, y_validation, y_test as a pickle

        Args:
            processed_df (pd.DataFrame): The processed dataframe
            label_column_name (str): The classification label
            dataset_class_name (str): The dataset class name being splitted
            dump (bool): Whether to dump the split data or not.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            X_train (pd.DataFrame): the training set
            X_validation (pd.DataFrame): the validation set
            X_test (pd.DataFrame): the test set
            y_train (pd.DataFrame): the training set labels
            y_validation (pd.DataFrame): the validation set labels
            y_test (pd.DataFrame): the test set labels
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

    @property
    def y_train(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def y_validation(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def y_test(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def feature_metrics_list(self) -> List[IFeatureMetrics]:
        raise NotImplementedError

    @property
    def label_preprocessor(self):
        raise NotImplementedError