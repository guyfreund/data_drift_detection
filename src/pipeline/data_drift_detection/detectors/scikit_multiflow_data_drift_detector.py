import os
import numpy as np
import pandas as pd
from typing import Dict, List, Set
# from skmultiflow.drift_detection.ddm import DDM
# from skmultiflow.drift_detection.eddm import EDDM
# from skmultiflow.drift_detection.hddm_a import HDDM_A
# from skmultiflow.drift_detection.hddm_w import HDDM_W
# from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector as ScikitMultiflowModuleBaseDataDriftDetector

from src.pipeline.data_drift_detection.data_drift import DataDrift
from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.config import Config


class ScikitMultiflowDataDriftDetector(IDataDriftDetector):
    def __init__(self, deployment_dataset: Dataset, training_processed_df_path: str, preprocessor: IPreprocessor):
        # assert os.path.exists(training_processed_df_path)
        self._training_processed_df_path: str = training_processed_df_path
        self._deployment_dataset: Dataset = deployment_dataset
        self._preprocessor: IPreprocessor = preprocessor
        self._processed_df: pd.DataFrame = pd.DataFrame()
        # self._ddm = DDM()
        # self._eddm = EDDM()
        # self._hddm_a = HDDM_A()
        # self._hddm_w = HDDM_W()
        # self._detectors: List[ScikitMultiflowModuleBaseDataDriftDetector] = [self._ddm, self._eddm, self._hddm_a, self._hddm_w]
        # self._drifted_instances_idx: Dict[ScikitMultiflowModuleBaseDataDriftDetector, Set[int]] = {d: set() for d in self._detectors}

    def detect(self) -> DataDrift:
        return DataDrift(is_drifted=False)

    def detect_old(self) -> DataDrift:
        # concatenate the training and deployment processed dataframes
        training_processed_df: pd.DataFrame = pd.read_pickle(self._training_processed_df_path)
        # TODO: think maybe to use pickle here
        deployment_processed_df, _, _ = self._preprocessor.preprocess(dataset=self._deployment_dataset)
        self._processed_df: pd.DataFrame = pd.concat([training_processed_df, deployment_processed_df])

        # detect changes
        for idx, instance in self._processed_df.iterrows():
            for detector in self._detectors:
                detector.add_element(instance.to_numpy())
                if detector.detected_change():
                    self._drifted_instances_idx[detector].add(idx)

        return DataDrift(is_drifted=self._is_drifted())

    def _is_drifted(self) -> bool:
        num_instances, _ = self._deployment_dataset.num_instances
        overlapping_instances_idx: Set[int] = set()
        percent_of_instances_list: List[float] = [
            Config().data.drift.scikit_multiflow.modules.DDM.percent_of_instances,
            Config().data.drift.scikit_multiflow.modules.EDDM.percent_of_instances,
            Config().data.drift.scikit_multiflow.modules.HDDM_A.percent_of_instances,
            Config().data.drift.scikit_multiflow.modules.HDDM_W.percent_of_instances
        ]
        is_ddm_drifted, is_eddm_drifted, is_hddm_a_drifted, is_hddm_w_drifted = False, False, False, False
        is_drifted_list = [is_ddm_drifted, is_eddm_drifted, is_hddm_a_drifted, is_hddm_w_drifted]
        all_instances_idx: List[Set[int]] = [instances_idx for _, instances_idx in self._drifted_instances_idx.items()]

        for detector_idx, instances_idx_i in enumerate(all_instances_idx):
            is_drifted_list[detector_idx] = len(instances_idx_i) / num_instances >= percent_of_instances_list[detector_idx]

            # treat overlapping instances
            for instances_idx_j in all_instances_idx:
                if instances_idx_j != instances_idx_i:
                    overlapping_instances_idx.union(instances_idx_i.intersection(instances_idx_j))

        percent_of_overlapping_instances = len(overlapping_instances_idx) / num_instances
        is_drifted_percent_of_overlapping_instances = percent_of_overlapping_instances >= Config().data.drift.scikit_multiflow.percent_on_overlapping_instances

        weights: np.ndarray = np.array([
            Config().data.drift.scikit_multiflow.modules.DDM.weight,
            Config().data.drift.scikit_multiflow.modules.EDDM.weight,
            Config().data.drift.scikit_multiflow.modules.HDDM_A.weight,
            Config().data.drift.scikit_multiflow.modules.HDDM_W.weight,
            Config().data.drift.scikit_multiflow.weight_on_overlapping_instances
        ])

        return np.dot(
            np.array(is_drifted_list + [is_drifted_percent_of_overlapping_instances]),
            weights
        ) >= Config().data.drift.scikit_multiflow.total_weight
