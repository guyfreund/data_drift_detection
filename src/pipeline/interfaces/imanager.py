from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import pandas as pd

from src.pipeline.data_drift_detection.interfaces.idata_drift_detector import IDataDriftDetector


class IManager(ABC):
    """
    Interface for a manager object
    """

    @abstractmethod
    def manage(self) -> Any:
        """ preprocesses the raw dataset
        saves the processed data frame in self._processed_df
        saves the processed dataset as a pickle
        saves the processed dataset plus as a pickle
        saves feature_metrics_list as a pickle

        Args:
            dataset (Dataset): The raw dataset

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[IFeatureMetrics]]:
            processed_dataset (pd.DataFrame): the processed data frame
            processed_dataset_plus (pd.DataFrame): the processed data frame with the addition of the DatasetType column for all instances
            feature_metrics_list (List[IFeatureMetrics]): a list of IFeatureMetric objects per feature
        """
        raise NotImplementedError

