import pickle
import os
from typing import List, Dict, Tuple
import numpy as np

from src.pipeline.data_drift_detection.constants import DataDriftType
from src.pipeline.data_drift_detection.data_drift import StatisticalBasedDataDrift, DataDrift, MeanDataDrift, \
    VarianceDataDrift, NumNullsDataDrift
from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector
from src.pipeline.datasets.dataset import Dataset
from src.pipeline.preprocessing.constants import FeatureType
from src.pipeline.preprocessing.interfaces.ifeature_metrics import IFeatureMetrics
from src.pipeline.preprocessing.interfaces.ipreprocessor import IPreprocessor
from src.pipeline.config import Config
from src.pipeline import logger

logging = logger.get_logger(__name__)

# TODO: add detector for new unseen values of categorical features

class StatisticalBasedDetector(IDataDriftDetector):
    def __init__(self, deployment_dataset: Dataset, training_feature_metrics_list_path: str, preprocessor: IPreprocessor):
        # assert os.path.exists(training_feature_metrics_list_path)
        self._training_feature_metric_list_path: str = training_feature_metrics_list_path
        self._deployment_dataset: Dataset = deployment_dataset
        self._preprocessor: IPreprocessor = preprocessor

    def detect(self) -> StatisticalBasedDataDrift:
        # data processing
        feature_names, training_feature_metrics_list, deployment_feature_metrics_list = self._get_data()
        data_drifts_per_feature_dict: Dict[str, Dict[DataDriftType, DataDrift]] = self._extract_data_drifts(
            feature_names=feature_names,
            training_feature_metrics_list=training_feature_metrics_list,
            deployment_feature_metrics_list=deployment_feature_metrics_list
        )
        variance_drifted_feature_names, mean_drifted_feature_names, \
            num_nulls_drifted_feature_names = self._get_data_drift_feature_names(data_drifts_per_feature_dict)

        # statistical data drift detection
        # based on the percent of feature that were identified as drifted per data drift type
        percent_features_variance_drifted = len(variance_drifted_feature_names) / len(feature_names)
        percent_features_mean_drifted = len(mean_drifted_feature_names) / len(feature_names)
        percent_features_num_nulls_drifted = len(num_nulls_drifted_feature_names) / len(feature_names)

        is_variance_drifted = percent_features_variance_drifted >= Config().data_drift.internal_data_drift_detector.variance.percent_of_features
        is_mean_drifted = percent_features_mean_drifted >= Config().data_drift.internal_data_drift_detector.mean.percent_of_features
        is_num_nulls_drifted = percent_features_num_nulls_drifted >= Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_of_features

        weighted_sum_on_all_data_drift_types = np.dot(
            np.array([is_variance_drifted, is_mean_drifted, is_num_nulls_drifted]),
            np.array([
                Config().data_drift.internal_data_drift_detector.variance.weight,
                Config().data_drift.internal_data_drift_detector.mean.weight,
                Config().data_drift.internal_data_drift_detector.number_of_nulls.weight
            ])
        )

        is_drifted = weighted_sum_on_all_data_drift_types > Config().data_drift.internal_data_drift_detector.statistical_based_threshold
        logging.info(f"Statistical based data drift detector for {self._deployment_dataset.name}:"
                     f"is_drifted={is_drifted}")
        return StatisticalBasedDataDrift(is_drifted=is_drifted)

    def _get_data(self) -> Tuple[List[str], List[IFeatureMetrics], List[IFeatureMetrics]]:
        with open(self._training_feature_metric_list_path, 'rb') as f:
            training_feature_metrics_list: List[IFeatureMetrics] = pickle.load(f)
        # TODO: think maybe to use pickle here
        _, _, deployment_feature_metrics_list = self._preprocessor.preprocess(dataset=self._deployment_dataset)

        assert len(training_feature_metrics_list) == len(deployment_feature_metrics_list)
        training_feature_names = [fm.name for fm in training_feature_metrics_list]
        assert training_feature_names == [fm.name for fm in deployment_feature_metrics_list]

        return training_feature_names, training_feature_metrics_list, deployment_feature_metrics_list

    @staticmethod
    def _extract_data_drifts(feature_names: List[str], training_feature_metrics_list: List[IFeatureMetrics],
                             deployment_feature_metrics_list: List[IFeatureMetrics]) -> Dict[str, Dict[DataDriftType, DataDrift]]:
        data_drifts_per_feature_dict: Dict[str, Dict[DataDriftType, DataDrift]] = {feature_name: {} for feature_name in feature_names}

        for feature_name, training_fm, deployment_fm in zip(feature_names, training_feature_metrics_list, deployment_feature_metrics_list):
            is_numeric_feature = training_fm.ftype == FeatureType.Numeric

            # extract variance
            training_variance = training_fm.variance
            deployment_variance = deployment_fm.variance
            if is_numeric_feature:
                maximum_change = np.floor(training_variance * Config().data_drift.internal_data_drift_detector.variance.percent_threshold)
                is_variance_drifted = np.abs(training_variance - deployment_variance) > maximum_change
            else:
                is_variance_drifted = False
            data_drifts_per_feature_dict[feature_name].update({DataDriftType.Variance: VarianceDataDrift(is_drifted=is_variance_drifted)})

            # extract mean
            training_mean = training_fm.mean
            deployment_mean = deployment_fm.mean
            if is_numeric_feature:
                maximum_change = np.floor(training_mean * Config().data_drift.internal_data_drift_detector.mean.percent_threshold)
                is_mean_drifted = np.abs(training_mean - deployment_mean) > maximum_change
            else:
                is_mean_drifted = False
            data_drifts_per_feature_dict[feature_name].update({DataDriftType.Mean: MeanDataDrift(is_drifted=is_mean_drifted)})

            # extract number of nulls
            training_num_nulls = training_fm.number_of_nulls
            deployment_num_nulls = deployment_fm.number_of_nulls
            maximum_change = np.floor(training_num_nulls * Config().data_drift.internal_data_drift_detector.number_of_nulls.percent_threshold)
            is_num_nulls_drifted = np.abs(training_num_nulls - deployment_num_nulls) > maximum_change
            data_drifts_per_feature_dict[feature_name].update({DataDriftType.NumNulls: NumNullsDataDrift(is_drifted=is_num_nulls_drifted)})

        return data_drifts_per_feature_dict

    @staticmethod
    def _get_data_drift_feature_names(data_drifts_per_feature_dict: Dict[str, Dict[DataDriftType, DataDrift]]) -> Tuple[List[str], List[str], List[str]]:
        variance_drifted_feature_names, mean_drifted_feature_names, num_nulls_drifted_feature_names = [], [], []

        for fn, fm in data_drifts_per_feature_dict.items():
            if fm[DataDriftType.Variance].is_drifted:
                variance_drifted_feature_names.append(fn)
            if fm[DataDriftType.Mean].is_drifted:
                mean_drifted_feature_names.append(fn)
            if fm[DataDriftType.NumNulls].is_drifted:
                num_nulls_drifted_feature_names.append(fn)

        return variance_drifted_feature_names, mean_drifted_feature_names, num_nulls_drifted_feature_names
