from typing import Any, List
import pandas as pd
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generator import DataGenerator
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel
from ydata_synthetic.synthesizers.regular import WGAN_GP
# Import transformation function

'''------Utilities------'''

def test_data():
    import ydata_synthetic.preprocessing.regular.credit_fraud
    # Read the original data and have it preprocessed
    data = ydata_synthetic.preprocessing.regular.credit_fraud.pd.read_csv('./data/creditcard.csv', index_col=[0])
    # Extract list of columns
    data_cols = list(data.columns)
    print('Dataset columns: {}'.format(data_cols))

    # We will only synthesize the minority class (=1, aka fraud)
    # train_data contains 492 entries which had 'Class' value as 1 (which were very few)
    train_data = data.loc[data['Class'] == 1].copy()

    # Before training the GAN we apply data transformation
    # PowerTransformation - make data distribution more Gaussian-like.
    data = ydata_synthetic.preprocessing.regular.credit_fraud.transformations(train_data)
    print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0],
                                                                                 train_data.shape[1]))

    return data


class DataGenerationManagerInfo:

    def __init__(self, origin_dataset: Dataset,
                 preprocessor: IPreprocessor,
                 gen_model: IModel,
                 gen_model_args: List[Any],
                 train_args: List[Any]
                 # training_feature_metrics_list_path: str,
                 # training_processed_df_path: str
                 ):
        self.origin_dataset: Dataset = origin_dataset
        self.preprocessor: IPreprocessor = preprocessor
        self.gen_model: IModel = gen_model
        self.gen_model_args: List[Any] = gen_model_args
        self.train_args: List[Any] = train_args

        # self.training_feature_metrics_list_path: str = training_feature_metrics_list_path
        # self.training_processed_df_path: str = training_processed_df_path


class DataGenerationManager(IManager):
    def __init__(self, info: DataGenerationManagerInfo):
        self._origin_dataset = info.origin_dataset
        self._generated_dataset = None
        self._data_generator = DataGenerator(synthesizer_model=info.gen_model,
                                             model_params=info.gen_model_args,
                                             training_params=info.train_args)
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

