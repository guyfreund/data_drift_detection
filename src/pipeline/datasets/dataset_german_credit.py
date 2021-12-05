import pandas as pd
import os

print(os.getcwd())


from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.constants import DatasetType
from src.pipeline.datasets.paths import GERMAN_CREDIT_DATASET_PATH

class GermanCreditDataset(Dataset):
    def __init__(self):
        super.__init__(dtype=DatasetType.Training, path=GERMAN_CREDIT_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


gc_ds = GermanCreditDataset()