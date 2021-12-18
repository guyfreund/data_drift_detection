from typing import Any, List
import pandas as pd
from ydata_synthetic.synthesizers.regular import CGAN
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_generation.data_generator import GANDataGenerator
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.model.interfaces.imodel import IModel

# Import transformation function


class DataGenerationManagerInfo:

    def __init__(self, origin_dataset: Dataset,
                 label_col: str,
                 model_path: str,

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
        self._label_col = self._origin_dataset.label_column_name
        self._data_generator = GANDataGenerator(dataset=self._origin_dataset,
                                                label_col=self._label_col,
                                                model_class=CGAN,
                                                trained_model_path=str,
             inverse_preprocesser: Optional[Any] = None) -> None)
        self._gen_model_args = info.gen_model_args
        self._train_args = train_args



    def manage(self, sample_size_to_generate: int) -> DataDrift:
        # Training the GAN model
        self._data_generator.train(self._origin_dataset)
        self.__generated_dataset = self._data_generator.generate(n_samples=sample_size_to_generate, vector_dim=dataset.shape[1])
        return


    def get_generated_dataset(self):
        return self.__generated_dataset


class MultipleDatasetGenerationManager(IManager):
    def __init__(self, info_list: List[DataGenerationManagerInfo]):
        self.data_generation_managers: List[DataGenerationManager] = [DataGenerationManager(info) for info in info_list]

    def manage(self) -> List[DataDrift]:
        data_drifts: List[DataDrift] = [manager.manage() for manager in self.data_generation_managers]
        return data_drifts



