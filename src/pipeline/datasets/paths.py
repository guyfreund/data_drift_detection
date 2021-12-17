import os

GERMAN_CREDIT_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "german_credit", "german.data"))
GERMAN_CREDIT_NUMERIC_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "german_credit", "german.data-numeric"))
GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH = ''
GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH = ''
GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDataset.pickle"))
GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDatasetPlus.pickle"))
GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDataset_FeatureMetricsList.pickle"))

BANK_MARKETING_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank.csv"))
BANK_MARKETING_FULL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-full.csv"))
BANK_MARKETING_ADDITIONAL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-additional.csv"))
BANK_MARKETING_ADDITIONAL_FULL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-additional-full.csv"))
BANK_MARKETING_DEPLOYMENT_DATASET_PATH = ''
BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH = ''
BANK_MARKETING_TRAINING_PROCESSED_DF_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDataset.pickle"))
BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDatasetPlus.pickle"))
BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDataset_FeatureMetricsList.pickle"))
