from abc import ABC
from typing import Any
from src.pipeline.data_generation.interfaces.idata_generator import IDataGenerator
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.datasets.dataset import Dataset
from ydata_synthetic.synthesizers.gan import BaseModel
# from imblearn.over_sampling import SMOTE, ADASYN
from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Import transformation function
from ydata_synthetic.preprocessing.regular.credit_fraud import *


class DataGenerator(IDataGenerator, ABC):
    def __init__(self, dataset: Dataset, preprocessor: IPreprocessor, gen_model: Any, model_params: dict, path: str):
        self.origin_dataset = preprocessor(dataset)
        self.model_params = ModelParameters(model_params)
        self.gen_model = gen_model
        self.train_params = None
        self.saving_path = path

    def train(self, dataset: pd.DataFrame, train_params):
        # Before training the GAN we apply data transformation
        # PowerTransformation - make data distribution more Gaussian-like.
        self.train_params = TrainParameters(train_params)
        print('Training generative model.')
        self.gen_model.train(self.origin_dataset, self.train_params)
        print('Done training.')


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