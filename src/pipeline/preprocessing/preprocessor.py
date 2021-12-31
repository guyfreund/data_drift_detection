import pickle
import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.pipeline.config import Config
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.feature_metrics.feature_metrics_categorical import CategoricalFeatureMetrics
from src.pipeline.preprocessing.feature_metrics.feature_metrics_numeric import NumericFeatureMetrics
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.preprocessing.label_preprocessor import LabelProcessor
from src.pipeline import logger
from src.pipeline.preprocessing.paths import BANK_MARKETING_LABEL_ENCODER_PATH_TRAINING, GERMAN_CREDIT_LABEL_ENCODER_PATH_TRAINING

logging = logger.get_logger(__name__)

class Preprocessor(IPreprocessor):

    def __init__(self):
        self._processed_df: pd.DataFrame = pd.DataFrame()
        self._processed_df_plus: pd.DataFrame = pd.DataFrame()

        self._X_train: pd.DataFrame = pd.DataFrame()
        self._X_validation: pd.DataFrame = pd.DataFrame()
        self._X_test: pd.DataFrame = pd.DataFrame()
        self._y_train: pd.DataFrame = pd.DataFrame()
        self._y_validation: pd.DataFrame = pd.DataFrame()
        self._y_test: pd.DataFrame = pd.DataFrame()

        self._X_train_raw = None
        self._X_validation_raw = None
        self._X_test_raw = None
        self._y_train_raw = None
        self._y_validation_raw = None
        self._y_test_raw = None

        self._feature_metrics_list: List[IFeatureMetrics] = []
        self._label_preprocessor = None

    # def preprocess(self, dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:
    #
    #     logging.info(f'Dataset Info: num of total features: {len(dataset.categorical_feature_names) + len(dataset.numeric_feature_names)} | '
    #                  f'num of categorical features: {len(dataset.categorical_feature_names)} | '
    #                  f'num of numerical features: {len(dataset.numeric_feature_names)}')
    #
    #     feature_metrics_list: List[IFeatureMetrics] = self._build_feature_metrics_list(dataset)
    #     self._feature_metrics_list = feature_metrics_list
    #
    #     self._processed_df = dataset.raw_df.copy()
    #
    #     pd.DataFrame(StandardScaler().fit_transform(self._processed_df[dataset.numeric_feature_names]))
    #     d = defaultdict(LabelEncoder)
    #     self._processed_df[dataset.categorical_feature_names].apply(lambda x: d[x.name].fit_transform(x))
    #     self._processed_df = pd.get_dummies(self._processed_df, columns=dataset.categorical_feature_names)
    #
    #     self._processed_df[dataset.label_column_name] = LabelEncoder().fit_transform(self._processed_df[dataset.label_column_name])
    #     if dataset.original_label_column_name:
    #         # process original label column name in case it is needed
    #         self._processed_df[dataset.original_label_column_name] = LabelEncoder().fit_transform(self._processed_df[dataset.original_label_column_name])
    #
    #     logging.info(f"Preprocessing: data was preprocessed successfully.")
    #     logging.info(f"Preprocessing Info: num of categorical features: {len(self._processed_df.select_dtypes(include=['bool', 'object']).columns)} | "
    #                  f"num of numerical features: {len(self._processed_df.select_dtypes(exclude=['bool', 'object']).columns)}")
    #
    #     self._processed_df_plus = self._processed_df.copy()
    #     self._processed_df_plus[Config().preprocessing.data_drift_model_label_column_name] = dataset.dtype.value
    #
    #     self._save_data_as_pickle(dataset.name)
    #
    #     return self._processed_df, self._processed_df_plus, self._feature_metrics_list

    def preprocess(self, dataset: Dataset, generate_dataset_plus: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:

        logging.info(f'Dataset {dataset.name} Info: num of total features: {len(dataset.categorical_feature_names) + len(dataset.numeric_feature_names)} | '
                     f'num of categorical features: {len(dataset.categorical_feature_names)} | '
                     f'num of numerical features: {len(dataset.numeric_feature_names)}')

        feature_metrics_list: List[IFeatureMetrics] = self._build_feature_metrics_list(dataset)
        self._feature_metrics_list = feature_metrics_list

        self._processed_df = dataset.raw_df.copy()

        self._label_preprocessor = LabelProcessor(dataset)
        self._processed_df = self._label_preprocessor.preprocessed_data(self._processed_df, dump_encoder=False)
        logging.info(f'Dataset {dataset.name} BEFORE cleaning Nulls: num of total rows: {dataset.raw_df.shape[0]}')
        # Remove NULLS
        self._processed_df = self._processed_df.dropna()
        logging.info(f'Dataset {dataset.name} AFTER cleaning Nulls: num of total rows: {self._processed_df.shape[0]}')

    # pd.DataFrame(StandardScaler().fit_transform(self._processed_df[dataset.numeric_feature_names]))
        # d = defaultdict(LabelEncoder)
        # self._processed_df[dataset.categorical_feature_names].apply(lambda x: d[x.name].fit_transform(x))
        # self._processed_df = pd.get_dummies(self._processed_df, columns=dataset.categorical_feature_names)
        #
        # self._processed_df[dataset.label_column_name] = LabelEncoder().fit_transform(self._processed_df[dataset.label_column_name])

        logging.info(f"Preprocessing: data was preprocessed successfully.")
        logging.info(f"Preprocessing Info: num of categorical features: {len(self._processed_df.select_dtypes(include=['bool', 'object']).columns)} | "
                     f"num of numerical features: {len(self._processed_df.select_dtypes(exclude=['bool', 'object']).columns)}")

        if generate_dataset_plus:
            self._processed_df_plus = self._processed_df.copy()
            self._processed_df_plus[Config().preprocessing.data_drift_model_label_column_name] = dataset.dtype.value

        self._save_data_as_pickle(dataset.name, generate_dataset_plus)

        return self._processed_df, self._processed_df_plus, self._feature_metrics_list

    def postprocess(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        return self._label_preprocessor.preprocessed_data(processed_df=processed_df)

    @staticmethod    
    def _build_feature_metrics_list(dataset: Dataset) -> List[IFeatureMetrics]:
        feature_metrics_list: List[IFeatureMetrics] = []

        categorical_cols = dataset.categorical_feature_names
        for col in categorical_cols:
            feature_metric = CategoricalFeatureMetrics(col, dataset.dtype)
            feature_metric.number_of_nulls = int(dataset.raw_df[col].isna().sum())
            feature_metrics_list.append(feature_metric)

        numerical_cols = dataset.numeric_feature_names
        for col in numerical_cols:
            feature_metric = NumericFeatureMetrics(col, dataset.dtype)
            feature_metric.mean = dataset.raw_df[col].mean()
            feature_metric.variance = dataset.raw_df[col].var()
            feature_metric.number_of_nulls = int(dataset.raw_df[col].isna().sum())
            feature_metrics_list.append(feature_metric)

        return feature_metrics_list

    def _save_data_as_pickle(self, dataset_class_name: str, generate_dataset_plus: bool):
        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}.pickle"))
        self._processed_df.to_pickle(path)

        if generate_dataset_plus:
            path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}Plus.pickle"))
            self._processed_df_plus.to_pickle(path)

        path = os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_FeatureMetricsList.pickle"))
        with open(path, 'wb') as output:
            pickle.dump(self._feature_metrics_list, output)

        logging.info(f'Save Data: {dataset_class_name}.pickle, {dataset_class_name}Plus.pickle, '
                     f'{dataset_class_name}_FeatureMetricsList.pickle files has been saved')

    def _save_split_data_as_pickle(self, dataset_class_name: str):
        paths = [
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_train.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_validation.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_test.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_train.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_validation.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_test.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_train_raw.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_validation_raw.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_X_test_raw.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_train_raw.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_validation_raw.pickle")),
            os.path.abspath(os.path.join(__file__, "..", "raw_files", f"{dataset_class_name}_y_test_raw.pickle"))
        ]
        dataframes = [
            self._X_train,
            self._X_validation,
            self._X_test,
            self._y_train,
            self._y_validation,
            self._y_test,
            self._X_train_raw,
            self._X_validation_raw,
            self._X_test_raw,
            self._y_train_raw,
            self._y_validation_raw,
            self._y_test_raw,
        ]

        for path, dataframe in zip(paths, dataframes):
            dataframe.to_pickle(path)

        logging.info(f'Save Data: {paths} files has been saved')

    def split(self, processed_df: pd.DataFrame, label_column_name: str, dataset_class_name: str = '', dump: bool = True) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        processed_df_with_idx_column: pd.DataFrame = processed_df.copy()
        processed_df_with_idx_column['idx'] = np.arange(processed_df.shape[0])
        data_y = processed_df_with_idx_column[label_column_name]
        data_X = processed_df_with_idx_column.drop(label_column_name, axis=1)
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(data_X, data_y, test_size=Config().preprocessing.split.train_test_split_size)
        X_test_idx, X_validation_idx, y_test, y_validation = train_test_split(X_test_idx, y_test, test_size=Config().preprocessing.split.validation_test_split_size)

        self._X_train = X_train_idx.drop('idx', axis=1)
        self._X_validation = X_validation_idx.drop('idx', axis=1)
        self._X_test = X_test_idx.drop('idx', axis=1)
        self._y_train = y_train
        self._y_validation = y_validation
        self._y_test = y_test

        if dump:
            raw_df: pd.DataFrame = self._label_preprocessor.postprocess_data(processed_df=processed_df)
            assert len(raw_df) == len(processed_df)
            X_raw: pd.DataFrame = raw_df.drop(label_column_name, axis=1)
            y_raw: pd.DataFrame = raw_df[label_column_name]

            self._X_train_raw = X_raw.iloc[X_train_idx['idx']]
            self._X_validation_raw = X_raw.iloc[X_validation_idx['idx']]
            self._X_test_raw = X_raw.iloc[X_test_idx['idx']]
            self._y_train_raw = y_raw.iloc[X_train_idx['idx']]
            self._y_validation_raw = y_raw.iloc[X_validation_idx['idx']]
            self._y_test_raw = y_raw.iloc[X_test_idx['idx']]

            self._save_split_data_as_pickle(dataset_class_name=dataset_class_name)

        logging.info(f"Split Data: processed_df has been splitted by the '{label_column_name}' column")
        logging.info(f"Split Info: train size: {round(len(self._X_train)/len(processed_df), 2)*100}% | "
                     f"validation size: {round(len(self._X_validation)/len(processed_df), 2)*100}% | "
                     f"test size: {round(len(self._X_test)/len(processed_df), 2)*100}%")

        return self._X_train, self._X_validation, self._X_test, self._y_train, self._y_validation, self._y_test

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
    def feature_metrics_list(self) -> List[IFeatureMetrics]:
        return self._feature_metrics_list

    @property
    def label_preprocessor(self):
        return self._label_preprocessor
