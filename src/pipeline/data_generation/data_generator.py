from abc import ABC
from typing import Any
import tensorflow as tf
import numpy as np
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.datasets.dataset import Dataset

# from imblearn.over_sampling import SMOTE, ADASYN
from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Import transformation function


class GANDataGenerator(IDataGenerator, ABC):
    ''''this class loads a GAN model trained on the dataset'''
    def __init__(self, dataset: Dataset, label_col: str, model_class: BaseModel, trained_model_path: str, inverse_preprocesser: Any = None):

        self._synthesizer = model_class.load(trained_model_path)  # for now we use CGAN class only
        self._origin_dataset = dataset
        self._generated_dataset = None
        self._labels = dataset.raw_df[label_col].unique()
        self._inverse_preprocesser = inverse_preprocesser

    def generate_normal_samples(self, n_samples):
        z = tf.random.normal((n_samples, self._synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self._labels), maxval=max(self._labels) + 1, dtype=tf.dtypes.int32)
        generated_data = self._synthesizer.generator([z, label_z])
        return self._inverse_preprocesser(generated_data) if self._inverse_preprocesser else generated_data

    def generate_drifted_samples(self, n_samples):
        generated_data = self.generate_normal_samples(n_samples)
        # Do Drifting


    def add_data_drift(self, dataset):
        pass


    def drift(self, dataset: Dataset, precentage_drift_mean=20.0, precentage_drift_std=0.0, time_drift=None):
        """
        Introduces data drift

        Args:
        pct_drift_mean (float): Percentage of mean drift.
        pct_drift_spread (float): Percentage change in the spread of the data.
        time_drift (str): The time at which drift starts in the format 'YYYY-MM-DD hh:mm:ss'.
        Will be converted and processed internally as a numpy.datetime64 object.

        Returns:

        from source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
        suppose we start with {x_i} with mean m_1 and non-zero std s_1:
        We do the following transformation:
        y_i = m_2 + (x_i - m_1) * s_2/s_1
        to get a new mean m_2 with std s_2
        """
        drifted_features = []
        for feature in drifted_features:
            before_drift_data = dataset.raw_df[feature]
            before_drift_mean = before_drift_data.mean()
            before_drift_std = before_drift_data.std()
            new_drift_mean = before_drift_mean * precentage_drift_mean
            new_drift_std = before_drift_std * precentage_drift_std
            drifted_data = new_drift_mean + (before_drift_data - before_drift_mean) * (new_drift_std/before_drift_std)


        self._drifted_flag_ = True

        if return_df:
            df = pd.DataFrame(
                {"time": self.time_arr, "drifted_data": self.drifted_data}
            )
            return df
        else:
            return self.drifted_data


    def save_datset(self, dataset, path):
        dataset.raw_df.to_csv(path + 'generated_data') # TODO add configs and names suit what has been saved


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


    def save_datset(self, dataset, path):
        dataset.raw_df.to_csv(path + 'generated_data') # TODO add configs and names suit what has been saved