from src.pipeline.model.interfaces.imodel import IModel
from src.pipeline.model.production_models import BankMarketingProductionModel, GermanCreditProductionModel


class BankMarketingDataDriftModel(BankMarketingProductionModel):
    def __init__(self):
        super().__init__()


class GermanCreditDataDriftModel(GermanCreditProductionModel):
    def __init__(self):
        super().__init__()
