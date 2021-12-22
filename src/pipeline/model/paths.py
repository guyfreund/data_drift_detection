import os


# Production Models
BANK_MARKETING_PRODUCTION_MODEL_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "model", "raw_files", "BankMarketingProductionModel.sav"))

# Generator Models
BANK_MARKETING_GEN_CGAN_MODEL_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "model", "raw_files", "BankMarketingCGANGenModel.sav"))
GERMAN_CREDIT_GEN_CGAN_MODEL_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "model", "raw_files", "GermanCreditCGANGenModel.pkl"))

# Data Drift Models
