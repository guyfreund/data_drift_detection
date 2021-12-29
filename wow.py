from src.pipeline.datasets.dataset import Dataset
from src.pipeline.datasets.training_datasets import BankMarketingSampledTrainingTrainDataset
from src.pipeline.datasets.deployment_datasets import BankMarketingSampledDeploymentDataset
from src.pipeline.datasets.paths import BANK_MARKETING_RETRAINING_DF

x = Dataset.concatenate(
    dataset_list=[BankMarketingSampledTrainingTrainDataset(), BankMarketingSampledDeploymentDataset()],
    path=BANK_MARKETING_RETRAINING_DF
)