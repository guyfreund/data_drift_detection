from typing import Any

import pandas as pd

from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from ydata_synthetic.synthesizers.gan import BaseModel
# from imblearn.over_sampling import SMOTE, ADASYN
from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Import transformation function
from ydata_synthetic.preprocessing.regular.credit_fraud import *


class DataGenerator(IDataGenerator):
    def __init__(self, synthesizer_model: Any, model_params, training_params):
        self.model_params = ModelParameters(model_params)
        self.training_params = TrainParameters(training_params)
        self.synthesizer = synthesizer_model

    def train(self, dataset):
        # Before training the GAN we apply data transformation
        # PowerTransformation - make data distribution more Gaussian-like.
        dataset = transformations(dataset)
        self.synthesizer.train(dataset, self.training_params)
        print("Dataset info: Number of records - {} Number of variables - {}".format(dataset.shape[0],
                                                                                     dataset.shape[1]))

    def generate(self, n_samples, vector_dim, path=None, seed=0):

        np.random.seed(seed)

        generator_model = self.synthesizer.generator

        # Generate features' values based on random noise
        z = np.random.normal(size=(n_samples, vector_dim))
        g_z = generator_model.predict(z)
        assert len(g_z) == len(n_samples)

        # return self.synthesizer.sample(n_samples)
        return g_z

    def generate_data_drift(self, dataset: pd.DataFrame, drifted_feature):
        pass


