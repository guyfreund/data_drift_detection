from typing import Any
import pandas as pd
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generator import DataGenerator
from ydata_synthetic.synthesizers.regular import WGAN_GP

class DataGenerationManager(IManager):
    pass


    def manage(self, dataset: pd.DataFrame) -> Any:


        # Define the GAN and training parameters
        noise_dim = 32
        dim = 128
        batch_size = 128

        log_step = 20
        epochs = 60+1
        learning_rate = 5e-4
        beta_1 = 0.5
        beta_2 = 0.9

        n_samples = 100

        gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, dataset.shape[1], dim]
        train_args = ['', epochs, log_step]

        # Training the GAN model
        model = WGAN_GP

        data_gen = DataGenerator(synthesizer_model=model, model_params=gan_args, training_params=train_args)
        data_gen.train(dataset)
        data_gen.generate(n_samples=n_samples, vector_dim=dataset.shape[1])




