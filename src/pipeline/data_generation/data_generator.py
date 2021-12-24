from typing import Any, List, Union
import logging
import pandas as pd
import tensorflow as tf
import time
import os
import numpy as np
# from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import SMOTENC
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.config import Config
from src.pipeline.data_drift_detection.constants import DataDriftType


class DataGenerator(IDataGenerator):
    """ this class loads a GAN model trained on the dataset """

    def __init__(self, dataset: Dataset):
        self._origin_dataset = dataset
        self._dataset_name = dataset.__class__.__name__
        self._labels = [0, 1]  # dataset.raw_df[label_col].unique() #TODO: for now need to see why we get [1,2]


    # TODO: n_samples from config
    def generate_normal_samples(self, n_samples: int) -> Union[np.ndarray, pd.DataFrame]:
        z = tf.random.normal((n_samples, self._synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self._labels), maxval=max(self._labels) + 1,
                                    dtype=tf.dtypes.int32)
        generated_data = self._synthesizer.generator([z, label_z])
        generated_data = tf.make_ndarray(tf.make_tensor_proto(generated_data))
        # To Data Frame
        generated_dataset = self._inverse_preprocessor(generated_data) if self._inverse_preprocessor else generated_data
        # TODO remove later  [1,2,3] just for now
        generated_dataset = pd.DataFrame(columns=list(self._origin_dataset.raw_df.columns)+['1','2','3'], data=generated_dataset)
        return generated_dataset

    def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Union[
        np.ndarray, pd.DataFrame]:
        # first, generate normal data
        generated_data = self.generate_normal_samples(n_samples)
        # Do Drifting
        # get parameters for the drift
        num_features, percentage_features, num_drift_features = self._drifted_configs(n_samples)
        return self._add_data_drift(generated_data, num_drift_features, drift_types_list)

    def _drifted_configs(self, n_samples):
        numeric_feature_names, categorical_feature_names = self._origin_dataset.numeric_feature_names, self._origin_dataset.categorical_feature_names
        num_features = len(numeric_feature_names) + len(categorical_feature_names)
        percentage_features = max(Config().data_drift.internal_data_drift_detector.mean.percent_of_features,
                                  Config().data_drift.internal_data_drift_detector.variance.percent_of_features,
                                  Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_of_features)
        num_of_drift_features = np.random.uniform(percentage_features, 1.) * num_features
        num_of_drift_features = int(min(num_of_drift_features, n_samples))
        return num_features, percentage_features, num_of_drift_features

    def _add_data_drift(self, dataset: pd.DataFrame, num_drift_features: int,
                        drift_types_list: List[DataDriftType]
                        ) -> pd.DataFrame:
        """
        from source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

        Suppose we start with {x_i} with mean m_1 and non-zero std s_1:
        We do the following transformation:
            y_i = m_2 + (x_i - m_1) * s_2/s_1
        Then, we get a new mean m_2 with std s_2
        """
        numeric_feature_names, categorical_feature_names = self._origin_dataset.numeric_feature_names, self._origin_dataset.categorical_feature_names

        # TODO add random sample for the drift percentages
        drifted_features_all_types = np.random.choice(numeric_feature_names + categorical_feature_names,
                                                      num_drift_features, replace=False)
        percentage_drift_mean = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.mean.percent_threshold, 1.)
        percentage_drift_std = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.variance.percent_threshold, 1.)
        percentage_drift_nulls = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_threshold, 1.)

        logging.info(f'Features to drift are: {drifted_features_all_types}. '
                      f'number of features: {num_drift_features}. The drift types are: {drift_types_list}.'
                      f'\npercentage_drift_mean: {percentage_drift_mean}, '
                      f'percentage_drift_std: {percentage_drift_std}, '
                      f'percentage_drift_nulls: {percentage_drift_nulls}.')

        # df = dataset.raw_df
        df = dataset.copy()
        for drift_type in drift_types_list:
            if drift_type == DataDriftType.Statistical:
                drifted_features_numeric_only = np.random.choice(numeric_feature_names,
                                                                 min(num_drift_features, len(numeric_feature_names)),
                                                                 replace=False)
                logging.info(f'numeric features to drift are: {drifted_features_numeric_only}')
                for feature in drifted_features_numeric_only:
                    before_drift_data = df[feature]

                    before_drift_mean = before_drift_data.mean()
                    before_drift_std = before_drift_data.std()
                    new_drift_mean = before_drift_mean * percentage_drift_mean
                    new_drift_std = before_drift_std * percentage_drift_std
                    drifted_data = new_drift_mean + (before_drift_data - before_drift_mean) * (
                            new_drift_std / before_drift_std)

                    if pd.api.types.is_integer_dtype(
                            df[feature]):  # We check if variable is distinct so we round the values
                        drifted_data = drifted_data.astype(int)
                    df[feature] = drifted_data

            if drift_type == DataDriftType.NumNulls:  # TODO note if we do together with stat drift we might mask with nulls and change the desired drift..**
                for feature in drifted_features_all_types:
                    df.loc[df[feature].sample(frac=percentage_drift_nulls).index, feature] = np.nan

            # TODO optional: add drift of new unseen values of categorical feature
        # dataset.raw_df = df
        return df


    @property
    def origin_dataset(self) -> Dataset:
        return self._origin_dataset


class GANDataGenerator(DataGenerator):
    """ this class loads a GAN model trained on the dataset """

    def __init__(self, dataset: Dataset, model_class: BaseModel, trained_model_path: str,
                 inverse_preprocesser: Any = None):

        super().__init__(dataset)
        self._synthesizer = model_class.load(trained_model_path)  # for now we use CGAN class only
        self._inverse_preprocessor = inverse_preprocesser

    # TODO: n_samples from config
    def generate_normal_samples(self, n_samples: int) -> Union[np.ndarray, pd.DataFrame]:
        z = tf.random.normal((n_samples, self._synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self._labels), maxval=max(self._labels) + 1,
                                    dtype=tf.dtypes.int32)
        generated_data = self._synthesizer.generator([z, label_z])
        generated_data = tf.make_ndarray(tf.make_tensor_proto(generated_data))
        # To Data Frame
        generated_dataset = self._inverse_preprocessor(generated_data) if self._inverse_preprocessor else generated_data
        # TODO remove later  [1,2,3] just for now
        generated_dataset = pd.DataFrame(columns=list(self._origin_dataset.raw_df.columns)+['1','2','3'], data=generated_dataset)
        return generated_dataset

    # def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Union[
    #     np.ndarray, pd.DataFrame]:
    #     numeric_feature_names = self._origin_dataset.numeric_feature_names
    #     categorical_feature_names = self._origin_dataset.categorical_feature_names
    #     # first, generate normal data
    #     generated_data = self.generate_normal_samples(n_samples)
    #     # Do Drifting
    #     num_features, percentage_features, num_drift_features = drifted_configs(n_samples, numeric_feature_names, categorical_feature_names)
    #     return add_data_drift(generated_data, num_drift_features, drift_types_list,
    #                           numeric_feature_names, categorical_feature_names)

    @property
    def synthesizer(self):
        return self._synthesizer
    #
    # @property
    # def origin_dataset(self) -> Dataset:
    #     return self._origin_dataset


class SMOTENCDataGenerator(DataGenerator):
    """ this class generate synthetic data using SMOTENC method """
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        col_label = self._origin_dataset.label_column_name
        df = self._origin_dataset.raw_df
        self.X = df.drop(col_label, axis=1)
        self.y = df[col_label]
        df = self._origin_dataset.raw_df
        cat_cols = [col for col in df.columns if any(cat_col for cat_col in self._origin_dataset.categorical_feature_names if cat_col + '_' in col)]
        cat_cols_idx = [self.X.columns.get_loc(col) for col in df.columns if col in cat_cols]
        self._model = SMOTENC(random_state=42, categorical_features=cat_cols_idx)

    def generate_normal_samples(self, n_samples: int) -> Union[np.ndarray, pd.DataFrame]:
        synthetic_X, synthetic_y = self._model.fit_resample(self.X, self.y)
        synthetic_X[self._origin_dataset.label_column_name] = synthetic_y
        return synthetic_X

    # def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Union[np.ndarray, pd.DataFrame]:
    #     generated_samples = self.generate_normal_samples(n_samples)
    #     numeric_feature_names = self._origin_dataset.numeric_feature_names
    #     categorical_feature_names = self._origin_dataset.categorical_feature_names
    #     # first, generate normal data
    #     generated_data = self.generate_normal_samples(n_samples)
    #     # Do Drifting
    #     num_features, percentage_features, num_drift_features = drifted_configs(n_samples, numeric_feature_names,
    #                                                                             categorical_feature_names)
    #     return add_data_drift(generated_data, num_drift_features, drift_types_list,
    #                           numeric_feature_names, categorical_feature_names)
    #
    # def save_generated_dataset(self, dataset, path):
    #     pass
    @property
    def model(self):
        return self._model

