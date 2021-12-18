from typing import Any, List
import pandas as pd
import numpy as np
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
                                                inverse_preprocesser=None)
        self._data_drift_types = info.data_drift_types


    def manage(self, sample_size_to_generate: int) -> DataDrift:
        is_drifted = np.random.choice([False, True])
        self._get_generated_dataset(sample_size_to_generate, is_drifted)
        return DataDrift(is_drifted=is_drifted)


    def _get_generated_dataset(self, sample_size_to_generate: int, is_drifted: bool) -> None:
        if is_drifted:
            drift_types_list = np.random.choice(self._data_drift_types,
                                                size=np.random.randint(1, len(self._data_drift_types) + 1))
            self._data_generator.generate_drifted_samples(sample_size_to_generate, drift_types_list)
        else:
            self._data_generator.generate_normal_samples(sample_size_to_generate)



class MultipleDatasetGenerationManager(IManager):
    def __init__(self, info_list: List[DataGenerationManagerInfo]):
        self.data_generation_managers: List[DataGenerationManager] = [DataGenerationManager(info) for info in info_list]

    def manage(self) -> List[DataDrift]:
        data_drifts: List[DataDrift] = [manager.manage() for manager in self.data_generation_managers]
        return data_drifts



