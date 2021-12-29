from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from src.pipeline.datasets.dataset import Dataset
from copy import deepcopy


class LabelProcessor:

    def __init__(self, dataset: Dataset, save_encoder_path: str = None):
        self._label_col = dataset.label_column_name
        self._cat_cols = dataset.categorical_feature_names
        self._numeric_cols = dataset.numeric_feature_names
        self._encoder = None
        self._save_encoder_path = save_encoder_path
        self._original_label_col = dataset.original_label_column_name

    def preprocessed_data(self, dataset_df: pd.DataFrame, dump_encoder: bool = True) -> pd.DataFrame:
        df = dataset_df.copy()
        label_col = self._label_col
        cat_cols = self._cat_cols + [label_col]
        numeric_cols = self._numeric_cols

        if self._original_label_col:
            cat_cols += [self._original_label_col]

        columns = df.columns

        df_cat_cols = df[cat_cols]
        df_numeric_cols = df[numeric_cols]

        encoder_dict = defaultdict(LabelEncoder)
        df_cat_cols_processed = df_cat_cols.apply(lambda x: encoder_dict[x.name].fit_transform(x))

        df_processed = pd.concat([df_numeric_cols, df_cat_cols_processed], axis=1)
        df_processed = df_processed[columns]
        self._encoder = encoder_dict
        if dump_encoder:
            self._save_encoder_dict()

        return df_processed

    def postprocess_data(self, processed_df: pd.DataFrame, df_type: str = "Xy") -> pd.DataFrame:

        encoder_dict = self._encoder if self._encoder else self._load_encoder_dict()
        new_encoder_dict = encoder_dict
        cat_cols = []
        if self._original_label_col:
            cat_cols += [self._original_label_col]

        if df_type == "X":
            new_encoder_dict = deepcopy(encoder_dict)
            cat_cols = self._cat_cols
            del new_encoder_dict[self._label_col]
        elif df_type == "y":
            new_encoder_dict = deepcopy(encoder_dict)
            cat_cols = [self._label_col]
            for k, v in encoder_dict.items():
                if isinstance(v, LabelEncoder):
                    if k != self._label_col:
                        del new_encoder_dict[k]
        elif df_type == "Xy":
            cat_cols = self._cat_cols + [self._label_col]
        else:
            raise NotImplementedError

        inverse_transform_lambda = lambda x: new_encoder_dict[x.name].inverse_transform(x) if x.name in cat_cols else x
        return processed_df.apply(inverse_transform_lambda)

    def _save_encoder_dict(self):
        np.save(self._save_encoder_path, self._encoder)

    def _load_encoder_dict(self):
        np.load(self._save_encoder_path)

    @property
    def encoder(self):
        return self._encoder
