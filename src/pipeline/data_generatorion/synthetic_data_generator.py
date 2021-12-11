from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters


class SyntheticDataGenerator:
    def __init__(self, synthesizer_model, model_params, training_params):
        self.model_params = ModelParameters(model_params)
        self.training_params = TrainParameters(training_params)
        self.synthesizer = synthesizer_model

    def train(self, dataset):
        self.synthesizer.train(dataset, self.training_params)

    def synthesize_data(self, n_samples, path=None):
        return self.synthesizer.sample(n_samples)


    def augment(self, dataset):
        pass


