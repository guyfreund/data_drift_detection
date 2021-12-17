from abc import ABC
from typing import Any
import tensorflow as tf
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.datasets.dataset import Dataset

# from imblearn.over_sampling import SMOTE, ADASYN
from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Import transformation function
from ydata_synthetic.preprocessing.regular.credit_fraud import *


class GANDataGenerator(IDataGenerator, ABC):
    ''''this class loads a GAN model trained on the dataset'''
    def __init__(self, dataset: Dataset, label_col: str, inverse_preprocesser: Any, model_class: BaseModel, trained_model_path: str):

        self.synthesizer = model_class.load(trained_model_path)  # for now we use CGAN class only
        self.origin_dataset = dataset
        self.labels = dataset[label_col].unique()

    def generate_normal_samples(self, n_samples):
        z = tf.random.normal((n_samples, self.synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(self.labels), maxval=max(self.labels) + 1, dtype=tf.dtypes.int32)
        return self.synthesizer.generator([z, label_z])


    def generate_drifted_samples(self, n_samples):
        generated_data = self.generate_normal_samples(n_samples)
        # Do Drifting


    def generated_dataset(self, n_samples: int, do_drift: bool = False, seed: int = 0) -> pd.DataFrame:
        np.random.seed(seed)
        generator = self.gen_model.generator
        vector_dim = self.origin_dataset.raw_df.shape[1]
        # Generate features' values based on random noise
        z = np.random.normal(size=(n_samples, vector_dim))
        g_z = generator.predict(z)
        assert len(g_z) == len(z)
        # return self.synthesizer.sample(n_samples)
        if do_drift:
            self.add_data_drift(g_z)

        self.save_dataset(g_z)
        return g_z

    def add_data_drift(self, dataset):
        pass


    def save_datset(self, dataset):
        dataset.raw_df.to_csv(self.saving_path+'generated_data') # TODO add configs and names suit what has been saved


class BASICDataGenerator(IDataGenerator, ABC):
    ''''this class loads a GAN model trained on the dataset'''
    def __init__(self, dataset: Dataset, inverse_preprocesser: Any, model_class: Any, trained_model_path: str):
        # assert (model_class and trained_model_path), 'need to specify model class and model path'
        if (model_class and trained_model_path):
            self.gan_model: model_class.load(trained_model_path)  # for now we use CGAN class only
        self.origin_dataset = dataset



    def generate_samples(self, n_samples, labels):
        z = tf.random.normal((n_samples, synthesizer.noise_dim))
        label_z = tf.random.uniform((n_samples,), minval=min(labels), maxval=max(labels) + 1, dtype=tf.dtypes.int32)
        return synthesizer.generator([z, label_z])

    def generated_dataset(self, n_samples: int, do_drift: bool = False, seed: int = 0) -> pd.DataFrame:
        np.random.seed(seed)
        generator = self.gen_model.generator
        vector_dim = self.origin_dataset.raw_df.shape[1]
        # Generate features' values based on random noise
        z = np.random.normal(size=(n_samples, vector_dim))
        g_z = generator.predict(z)
        assert len(g_z) == len(z)
        # return self.synthesizer.sample(n_samples)
        if do_drift:
            self.add_data_drift(g_z)

        self.save_dataset(g_z)
        return g_z

    def add_data_drift(self, dataset):
        pass


    def save_datset(self, dataset):
        dataset.raw_df.to_csv(self.saving_path+'generated_data') # TODO add configs and names suit what has been saved
