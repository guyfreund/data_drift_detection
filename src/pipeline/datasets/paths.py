import os
################################################################################
# ------------------------------ GERMAN CREDIT --------------------------------#
################################################################################

# ------------------------------ RAW DATA ------------------------------#
GERMAN_CREDIT_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "german_credit", "german.data"))
GERMAN_CREDIT_NUMERIC_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "german_credit", "german.data-numeric"))
GERMAN_CREDIT_SAMPLED_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "german_credit", f"sampled_GermanCreditDataset.csv"))

# ------------------------------ DEPLOYMENT (SYNTHESIZED) DATA ------------------------------#
GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_GermanCreditDataset.csv"))
GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_GermanCreditDataset_plus.csv"))
GERMAN_CREDIT_SAMPLED_DEPLOYMENT_DATASET = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_sampled_GermanCreditDataset.csv"))

# ------------------------------ TRAINING DATA ------------------------------#
GERMAN_CREDIT_TRAINING_PROCESSED_DF_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDataset.pickle"))
GERMAN_CREDIT_TRAINING_PROCESSED_DF_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDatasetPlus.pickle"))
GERMAN_CREDIT_TRAINING_FEATURE_METRIC_LIST_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditDataset_FeatureMetricsList.pickle"))
GERMAN_CREDIT_CONCATENATED_DF = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditConcatenatedDataFrame.pickle"))

# ------------------------------ Backup and Testing ------------------------------#
# Not-drifted
SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_GermanCreditDataset_normal.csv"))
SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_GermanCreditDataset_plus_normal.csv"))
GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_GermanCreditDataset_normal.csv"))
GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_GermanCreditDataset_plus_normal.csv"))
# Drifted
SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_GermanCreditDataset_drift.csv"))
SMOTENC_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_GermanCreditDataset_plus_drift.csv"))
GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_GermanCreditDataset_drift.csv"))
GAN_GERMAN_CREDIT_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_GermanCreditDataset_plus_drift.csv"))


################################################################################
# ------------------------------ BANK MARKETING  ------------------------------#
################################################################################

# ------------------------------ RAW DATA ------------------------------#
BANK_MARKETING_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank.csv"))
BANK_MARKETING_FULL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-full.csv"))
BANK_MARKETING_ADDITIONAL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-additional.csv"))
BANK_MARKETING_ADDITIONAL_FULL_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", "bank-additional-full.csv"))
BANK_MARKETING_SAMPLED_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "bank_marketing", f"sampled_BankMarketingDataset.csv"))

# ------------------------------ DEPLOYMENT (SYNTHESIZED) DATA ------------------------------#
BANK_MARKETING_DEPLOYMENT_DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_BankMarketingDataset.csv"))
BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_BankMarketingDataset_plus.csv"))
BANK_MARKETING_SAMPLED_DEPLOYMENT_DATASET = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"generated_sampled_BankMarketingDataset.csv"))


# ------------------------------ TRAINING DATA ------------------------------#
BANK_MARKETING_TRAINING_PROCESSED_DF_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDataset.pickle"))
BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDatasetPlus.pickle"))
BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingDataset_FeatureMetricsList.pickle"))
BANK_MARKETING_CONCATENATED_DF = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BankMarketingConcatenatedDataFrame.pickle"))


# ------------------------------ Backup and Testing ------------------------------#
# Non-Drifted
SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_BankMarketingDataset_normal.csv"))
SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_BankMarketingDataset_plus_normal.csv"))
GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_BankMarketingDataset.csv_normal"))
GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_NORMAL = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_BankMarketingDataset_plus_normal.csv"))
# Drifted
SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_BankMarketingDataset_drift.csv"))
SMOTENC_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"smotenc_generated_BankMarketingDataset_plus_drift.csv"))
GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_BankMarketingDataset_drift.csv"))
GAN_BANK_MARKETING_DEPLOYMENT_DATASET_PLUS_PATH_DRIFT = os.path.abspath(os.path.join(__file__, "..", "..", "data_generation", "raw_files", f"gan_generated_BankMarketingDataset_plus_drift.csv"))
