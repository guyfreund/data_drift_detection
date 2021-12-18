from typing import Any, List, Union
import logging
import pandas as pd
import tensorflow as tf
import time
import os
import numpy as np
# from imblearn.over_sampling import SMOTE, ADASYN
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.config import Config
from src.pipeline.data_drift_detection.constants import DataDriftType


class GANDataGenerator(IDataGenerator):
    ''''this class loads a GAN model trained on the dataset'''

    def __init__(self, dataset: Dataset, label_col: str, model_class: BaseModel, trained_model_path: str,
                 inverse_preprocesser: Any = None):

        self._synthesizer = model_class.load(trained_model_path)  # for now we use CGAN class only
        self._origin_dataset = dataset
        self._dataset_name = dataset.__class__.__name__
        self._labels = dataset.raw_df[label_col].unique()
        self._inverse_preprocessor = inverse_preprocesser

    # TODO: n_samples from config
    def generate_normal_samples(self, n_samples: int) -> Union[np.ndarray, pd.DataFrame]:
        z = tf.random.normal((n_samples, self._synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self._labels), maxval=max(self._labels) + 1,
                                    dtype=tf.dtypes.int32)
        generated_data = self._synthesizer.generator([z, label_z])
        return self._inverse_preprocessor(generated_data) if self._inverse_preprocessor else generated_data

    def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Union[
        np.ndarray, pd.DataFrame]:
        generated_data = self.generate_normal_samples(n_samples)
        num_features = self._origin_dataset.numeric_feature_names + self._origin_dataset.categorical_feature_names
        precentage_features = max(Config().data_generation.internal_data_drift_detector.mean.percent_of_features,
                                  Config().data_generation.internal_data_drift_detector.variance.percent_of_features,
                                  Config().data_generation.internal_data_drift_detector.number_of_nulls.percent_of_features)
        num_of_drift_features = np.random.uniform(precentage_features, 1.) * num_features
        num_drift_features = min(num_of_drift_features, n_samples)
        # Do Drifting
        return self._add_data_drift(generated_data, num_drift_features, drift_types_list)

    # TODO: OPTIONAL add the statistics and features to drift somwhere and not only printing them.
    @staticmethod
    def _add_data_drift(dataset: pd.DataFrame, num_drift_features: int,
                        drift_types_list: List[DataDriftType]) -> pd.DataFrame:
        """
        from source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

        Suppose we start with {x_i} with mean m_1 and non-zero std s_1:
        We do the following transformation:
            y_i = m_2 + (x_i - m_1) * s_2/s_1
        Then, we get a new mean m_2 with std s_2
        """
        # TODO add random sample for the drift percentages
        drifted_features_all_types = np.random.choice(dataset.numeric_feature_names + dataset.categorical_feature_names,
                                                      num_drift_features, replace=False)
        percentage_drift_mean = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.mean.percent_threshold, 1.)
        percentage_drift_std = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.variance.percent_threshold, 1.)
        percentage_drift_nulls = np.random.uniform(
            Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_threshold, 1.)

        logging.debug(f'Features to drift are: {drifted_features_all_types}. '
                      f'number of features: {num_drift_features}. The drift types are: {drift_types_list}.'
                      f'\npercentage_drift_mean: {percentage_drift_mean}, '
                      f'percentage_drift_std: {percentage_drift_std}, '
                      f'percentage_drift_nulls: {percentage_drift_nulls}.')

        # df = dataset.raw_df
        df = dataset.copy()
        for drift_type in drift_types_list:
            if drift_type == DataDriftType.Statistical:
                drifted_features_numeric_only = np.random.choice(dataset.numeric_feature_names,
                                                                 min(num_drift_features, len(dataset.numeric_feature_names)),
                                                                 replace=False)
                logging.debug(f'numeric features to drift are: {drifted_features_numeric_only}')
                for feature in drifted_features_numeric_only:
                    before_drift_data = df[feature]
                    before_drift_mean = before_drift_data.mean()
                    before_drift_std = before_drift_data.std()
                    new_drift_mean = before_drift_mean * percentage_drift_mean
                    new_drift_std = before_drift_std * percentage_drift_std
                    drifted_data = new_drift_mean + (before_drift_data - before_drift_mean) * (
                            new_drift_std / before_drift_std)

                    if pd.api.types.is_integer_dtype(df[feature]): # We check if variable is distinct so we round the values
                        drifted_data = drifted_data.astype(int)
                    df[feature] = drifted_data

            if drift_type == DataDriftType.NumNulls:  # TODO note if we do together with stat drift we might mask with nulls and change the desired drift..**
                for feature in drifted_features_all_types:
                    df.loc[df[feature].sample(frac=percentage_drift_nulls).index, feature] = np.nan

            # TODO optional: add drift of new unseen values of categorical feature
        # dataset.raw_df = df
        return df

    def save_generated_dataset(self, dataset: Dataset, path: str, file_name: str = 'generated_data'):
        file_name = file_name + self._dataset_name + time.strftime("%Y%m%d-%H%M%S") + '.csv'
        dataset.raw_df.to_csv(os.path.join(path, file_name))


# TODO: OPTIONAL
class BASICDataGenerator(IDataGenerator):
    ''''this class loads a GAN model trained on the dataset'''

    def __init__(self, dataset: Dataset, inverse_preprocesser: Any, model_class: Any, trained_model_path: str):
        # assert (model_class and trained_model_path), 'need to specify model class and model path'
        pass

    def generate_normal_samples(self, n_samples: int) -> Union[np.ndarray, pd.DataFrame]:
        pass

    def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]) -> Union[
        np.ndarray, pd.DataFrame]:
        pass

    def save_generated_dataset(self, dataset, path):
        pass
