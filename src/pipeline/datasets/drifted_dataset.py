from dataset import Dataset
from src.pipeline.datasets.constants import DatasetType

class DriftedDataset(Dataset):
    def __init__(self, dtype: DatasetType, path: str):
        super().__init__(dtype=dtype, path=path)
        self.