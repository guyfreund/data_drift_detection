import logging
import os
import pickle
from typing import List, Union
import pandas as pd
import numpy as np
from ydata_synthetic.synthesizers.gan import BaseModel

from src.pipeline.config import Config
from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.data_generation.data_generator import GANDataGenerator
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType


class DataGenerationManagerInfo:

    def __init__(self, origin_dataset: Dataset, model_class: BaseModel,
                 sample_size_to_generate: int, model_path: str,
                 data_drift_types: List[DataDriftType],
                 save_data_path: str, save_data_plus_path: str):
        self.origin_dataset: Dataset = origin_dataset
        # self.dataset_name = type(origin_dataset.__name__)
        self.model_class: BaseModel = model_class
        self.model_path: str = model_path
        self.sample_size_to_generate: int = sample_size_to_generate
        self.data_drift_types: List[DataDriftType] = data_drift_types
        self.save_data_path: str = save_data_path
        self.save_data_plus_path: str = save_data_plus_path


class DataGenerationManager(IManager):
    def __init__(self, info: DataGenerationManagerInfo):
        self._origin_dataset = info.origin_dataset
        self._label_col = self._origin_dataset.label_column_name
        self._data_generator = GANDataGenerator(dataset=self._origin_dataset,
                                                label_col=self._label_col,
                                                model_class=info.model_class,
                                                trained_model_path=info.model_path,
                                                inverse_preprocesser=None) # TODO add inverse
        self._sample_size_to_generate = info.sample_size_to_generate
        self._data_drift_types = info.data_drift_types
        self._save_data_path = info.save_data_path
        self._save_data_plus_path = info.save_data_plus_path

    def manage(self) -> DataDrift:
        is_drifted = np.random.choice([False, True])
        generated_data = self._get_generated_dataset(is_drifted)
        logging.info(f'Finished generating data. (is_drifted={is_drifted}).')
        self._save_data_as_pickle(generated_data)
        logging.info('Done saving generated data.')
        return DataDrift(is_drifted=is_drifted)

    def _save_data_as_pickle(self, generated_dataset):
        dataset_class_name = self._origin_dataset.__class__.__name__


        generated_dataset_plus = generated_dataset.copy()
        generated_dataset_plus[Config().preprocessing.data_drift_model_label_column_name] = DatasetType.Deployment

        generated_dataset.to_csv(self._save_data_path, index=False)
        generated_dataset_plus.to_csv(self._save_data_plus_path, index=False)


    def _get_generated_dataset(self, is_drifted: bool) -> Union[np.array, pd.DataFrame]:
        if is_drifted:
            drift_types_list = np.random.choice(self._data_drift_types,
                                                size=np.random.randint(1, len(self._data_drift_types) + 1),
                                                replace=False)
            return self._data_generator.generate_drifted_samples(self._sample_size_to_generate, drift_types_list)
        else:
            return self._data_generator.generate_normal_samples(self._sample_size_to_generate)


class MultipleDatasetGenerationManager(IManager):
    def __init__(self, info_list: List[DataGenerationManagerInfo]):
        self.managers: List[DataGenerationManager] = [DataGenerationManager(info) for info in info_list]

    def manage(self) -> List[DataDrift]:
        data_drifts: List[DataDrift] = [manager.manage() for manager in self.managers]
        return data_drifts

