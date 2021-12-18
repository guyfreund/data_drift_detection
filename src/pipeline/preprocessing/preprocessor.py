import pickle
import os
from collections import defaultdict
from typing import Tuple, List, Type
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler

from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.feature_metrics.feature_metrics_categorical import CategoricalFeatureMetrics
from src.pipeline.preprocessing.feature_metrics.feature_metrics_numeric import NumericFeatureMetrics
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor


class Preprocessor(IPreprocessor):

    def __init__(self):
        self._processed_df = None
        self._processed_df_plus = None
        self._X_train = None
        self._X_validation = None
        self._X_test = None
        self._y_train = None
        self._y_validation = None
        self._y_test = None
        self._feature_metrics_list = []
        self._y_col = None

    def preprocess(self, dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:

        feature_metrics_list: List[IFeatureMetrics] = self.build_feature_metrics_list(dataset)
        self._feature_metrics_list = feature_metrics_list
        dataset_columns = dataset.raw_df.columns

        if 'y' in list(dataset_columns):
            self.y_col = 'y'
            dataset.raw_df[self.y_col].replace(('yes', 'no'), (1, 0), inplace=True)

        if 'classification' in list(dataset_columns):
            self.y_col = 'classification'
            dataset.raw_df[self.y_col].replace([1, 2], [1, 0], inplace=True)
            numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
                       'existingcredits', 'peopleliable']
            pd.DataFrame(StandardScaler().fit_transform(dataset.raw_df[numvars]))
            catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
             'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
             'telephone', 'foreignworker']
            d = defaultdict(LabelEncoder)
            dataset.raw_df[catvars].apply(lambda x: d[x.name].fit_transform(x))

        categorical_columns = dataset.raw_df.select_dtypes(include=['bool', 'object']).columns
        self._processed_df = pd.get_dummies(dataset.raw_df, columns=categorical_columns)

        self._processed_df_plus = self._processed_df.copy()
        self._processed_df_plus['datasetType'] = dataset.dtype

        self._save_datasets_as_pickle(dataset.__class__.__name__)

        return self._processed_df, self._processed_df_plus, self._feature_metrics_list

    def build_feature_metrics_list(self, dataset):
        feature_metrics_list = []

        categorical_cols = dataset.raw_df.select_dtypes(include=['bool', 'object']).columns
        for col in categorical_cols:
            feature_metric = CategoricalFeatureMetrics(col, dataset.dtype)
            feature_metric.number_of_nulls = int(dataset.raw_df[col].isna().sum())
            feature_metrics_list.append(feature_metric)

        numerical_cols = dataset.raw_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            feature_metric = NumericFeatureMetrics(col, dataset.dtype)
            feature_metric.mean = dataset.raw_df[col].mean()
            feature_metric.variance = dataset.raw_df[col].var()
            feature_metric.number_of_nulls = int(dataset.raw_df[col].isna().sum())
            feature_metrics_list.append(feature_metric)

        return feature_metrics_list

    def _save_datasets_as_pickle(self, dataset_class_name: str):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}.pickle"))
        with open(path, 'wb') as output:
            pickle.dump(self._processed_df, output)

        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}Plus.pickle"))
        with open(path, 'wb') as output:
            pickle.dump(self._processed_df_plus, output)

    def split(self, processed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_y = processed_df[self.y_col]
        data_X = processed_df.drop(self.y_col, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=1)

        self._X_train = X_train
        self._X_validation = X_validation
        self._X_test = X_test
        self._y_train = y_train
        self._y_validation = y_validation
        self._y_test = y_test

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    @property
    def processed_df(self) -> pd.DataFrame:
        return self._processed_df

    @property
    def processed_df_plus(self) -> pd.DataFrame:
        return self._processed_df_plus

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train

    @X_train.setter
    def X_train(self, X_train):
        self._X_train = X_train

    @property
    def X_validation(self) -> pd.DataFrame:
        return self._X_validation

    @X_validation.setter
    def X_validation(self, X_validation):
        self._X_validation = X_validation

    @property
    def X_test(self) -> pd.DataFrame:
        return self._X_test

    @X_test.setter
    def X_test(self, X_test):
        self._X_test = X_test

    @property
    def y_train(self) -> pd.DataFrame:
        return self._y_train

    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train

    @property
    def y_validation(self) -> pd.DataFrame:
        return self._y_validation

    @y_validation.setter
    def y_validation(self, y_validation):
        self._y_validation = y_validation

    @property
    def y_test(self) -> pd.DataFrame:
        return self._y_test

    @y_test.setter
    def y_test(self, y_test):
        self._y_test = y_test

    @property
    def y_col(self) -> str:
        return self._y_col

    @y_col.setter
    def y_col(self, y_col):
        self._y_col = y_col

    @property
    def feature_metrics_list(self) -> List[IFeatureMetrics]:
        return self._feature_metrics_list
