from abc import ABC
from typing import Any, List
import tensorflow as tf
import time
import os
import numpy as np
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.datasets.dataset import Dataset
# from imblearn.over_sampling import SMOTE, ADASYN

from src.pipeline.config import Config
from src.pipeline.data_drift_detection.constants import DataDriftType


class GANDataGenerator(IDataGenerator, ABC):
    ''''this class loads a GAN model trained on the dataset'''
    def __init__(self, dataset: Dataset, label_col: str, model_class: BaseModel, trained_model_path: str, inverse_preprocesser: Any = None):

        self._synthesizer = model_class.load(trained_model_path)  # for now we use CGAN class only
        self._origin_dataset = dataset
        self._dataset_name = type(dataset).__name__
        self._generated_dataset = None
        self._labels = dataset.raw_df[label_col].unique()
        self._inverse_preprocessor = inverse_preprocesser

    def generate_normal_samples(self, n_samples):
        z = tf.random.normal((n_samples, self._synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self._labels), maxval=max(self._labels) + 1, dtype=tf.dtypes.int32)
        generated_data = self._synthesizer.generator([z, label_z])
        return self._inverse_preprocessor(generated_data) if self._inverse_preprocessor else generated_data

    def generate_drifted_samples(self, n_samples: int, drift_types_list: List[DataDriftType]):
        generated_data = self.generate_normal_samples(n_samples)
        num_of_drift_features = Config().data_drift.internal_data_drift_detector.mean.percent_of_features * len(self._origin_dataset.raw_df)
        num_drift_features = min(num_of_drift_features, n_samples)

        # Do Drifting
        self._add_data_drift(generated_data, num_drift_features, drift_types_list)

    def _add_data_drift(self, dataset: Dataset, num_drift_features: int, drift_types_list: List[DataDriftType]):
        """
        from source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

        Suppose we start with {x_i} with mean m_1 and non-zero std s_1:
        We do the following transformation:
            y_i = m_2 + (x_i - m_1) * s_2/s_1
        Then, we get a new mean m_2 with std s_2

        """
        # TODO add random sample for the drift percentages
        percentage_drift_mean = Config().data_drift.internal_data_drift_detector.mean.percent_threshold
        percentage_drift_std = Config().data_drift.internal_data_drift_detector.variance.percent_threshold
        percentage_drift_nulls = Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_threshold
        drifted_features_all_types = np.random.choice(dataset.raw_df.columns, num_drift_features, replace=False)
        df = dataset.raw_df
        for drift_type in drift_types_list:
            if drift_type == DataDriftType.Statistical:
                drifted_features_numeric_only = np.random.choice(dataset.numeric_features,
                                                                 num_drift_features,
                                                                 replace=False)
                for feature in drifted_features_numeric_only:
                    before_drift_data = df[feature]
                    before_drift_mean = before_drift_data.mean()
                    before_drift_std = before_drift_data.std()
                    new_drift_mean = before_drift_mean * percentage_drift_mean
                    new_drift_std = before_drift_std * percentage_drift_std
                    drifted_data = new_drift_mean + (before_drift_data - before_drift_mean) * (new_drift_std/before_drift_std)
                    # TODO add check if variable is distinct to round the values
                    df[feature] = drifted_data

            if drift_type == DataDriftType.NumNulls:   # TODO note if we do together with stat drift we might mask with nulls and change the desired drift..**
                for feature in drifted_features_all_types:
                    df.loc[df[feature].sample(frac=percentage_drift_nulls).index, feature] = np.nan

            # TODO optional: add drift of new unseen values of categorical feature

        dataset.raw_df = df
        return dataset

    def save_generated_dataset(self, dataset: Dataset, path: str, file_name: str = 'generated_data'):
        file_name = file_name + self._dataset_name + time.strftime("%Y%m%d-%H%M%S") + '.csv'
        dataset.raw_df.to_csv(os.path.join(path, file_name))


class BASICDataGenerator(IDataGenerator, ABC):
    ''''this class loads a GAN model trained on the dataset'''
    def __init__(self, dataset: Dataset, inverse_preprocesser: Any, model_class: Any, trained_model_path: str):
        # assert (model_class and trained_model_path), 'need to specify model class and model path'
        pass

    def generate_normal_samples(self, n_samples):
        pass


    def generate_drifted_samples(self, n_samples):
        # generated_data = self.generate_normal_samples(n_samples)
        # Do Drifting
        pass


    def add_data_drift(self, dataset):
        pass


    def save_generated_dataset(self, dataset, path):
        pass