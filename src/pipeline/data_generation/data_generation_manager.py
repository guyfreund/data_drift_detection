from typing import Any, List
import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generator import GANDataGenerator
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel

# Import transformation function


class DataGenerationManagerInfo:

    def __init__(self, origin_dataset: Dataset,
                 label_col: str,
                 model_path: str
                 # training_feature_metrics_list_path: str,
                 # training_processed_df_path: str
                 ):
        self.origin_dataset: Dataset = origin_dataset  #TODO ?
        self.label_col: str = label_col
        # self.gan_model: CGAN.load(model_path)
        # self.training_feature_metrics_list_path: str = training_feature_metrics_list_path
        # self.training_processed_df_path: str = training_processed_df_path


class DataGenerationManager(IManager):
    def __init__(self, info: DataGenerationManagerInfo):
        self._origin_dataset = info.origin_dataset
        self._generated_dataset = None
        self._label_col
        self._data_generator = GANDataGenerator(dataset=self._origin_dataset,
                                                label_col=info.label_col,
             model_class: BaseModel,
             trained_model_path: str,
             inverse_preprocesser: Optional[Any] = None) -> None)
        self._gen_model_args = info.gen_model_args
        self._train_args = train_args



    def manage(self, sample_size_to_generate: int) -> None:
        # Training the GAN model
        self._data_generator.train(self._origin_dataset)
        self.__generated_dataset = self._data_generator.generate(n_samples=sample_size_to_generate, vector_dim=dataset.shape[1])


    def get_generated_dataset(self):
        return self.__generated_dataset


class MultipleDatasetGenerationManager(IManager):
    pass



if __name__ == '__main__':
    # Define the GAN and training parameters
    ##### FOR NOW ######
    noise_dim = 32
    dim = 128
    batch_size = 128

    log_step = 20
    epochs = 60 + 1
    learning_rate = 5e-4
    beta_1 = 0.5
    beta_2 = 0.9

    n_samples = 100
    # dataset = get_data()
    gen_model = WGAN_GP

    gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, dataset.shape[1], dim]
    train_args = ['', epochs, log_step]

    data_gen_manager = DataGenerationManager()
    data_gen_manager.manage(sample_size_to_generate=n_samples)
    gen_samples = data_gen_manager.get_generated_dataset()

