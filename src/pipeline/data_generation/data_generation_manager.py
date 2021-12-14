from typing import Any, List
import pandas as pd
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generator import DataGenerator
from ydata_synthetic.synthesizers.regular import WGAN_GP
# Import transformation function



def test_data():
    from ydata_synthetic.preprocessing.regular.credit_fraud import *
    # Read the original data and have it preprocessed
    data = pd.read_csv('./data/creditcard.csv', index_col=[0])
    # Extract list of columns
    data_cols = list(data.columns)
    print('Dataset columns: {}'.format(data_cols))

    # We will only synthesize the minority class (=1, aka fraud)
    # train_data contains 492 entries which had 'Class' value as 1 (which were very few)
    train_data = data.loc[data['Class'] == 1].copy()

    # Before training the GAN we apply data transformation
    # PowerTransformation - make data distribution more Gaussian-like.
    data = transformations(train_data)
    print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0],
                                                                                 train_data.shape[1]))

    return data


class DataGenerationManagerInfo:
    pass


class DataGenerationManager(IManager):
    def __init__(self, dataset: pd.DataFrame, gen_model: Any, gen_model_args: List[Any], train_args: List[Any]):
        self.__origin_dataset = dataset
        self.__generated_dataset = None
        self.__gen_model = gen_model
        self.__gen_model_args = gen_model_args
        self.__train_args = train_args



    def manage(self, sample_size_to_generate: int) -> None:
        # Training the GAN model
        data_gen = DataGenerator(synthesizer_model=self.__gen_model, model_params=self.__gen_model_args, training_params=self.__train_args)
        data_gen.train(self.__origin_dataset)
        self.__generated_dataset = data_gen.generate(n_samples=sample_size_to_generate, vector_dim=dataset.shape[1])


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

