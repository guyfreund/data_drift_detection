import os

# Encoding Mapping
BANK_MARKETING_LABEL_ENCODER_PATH_DEPLOYMENT = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BanMarketingLabelEncoding.npy"))
GERMAN_CREDIT_LABEL_ENCODER_PATH_DEPLOYMENT = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditLabelEncoding.npy"))

# Encoding Mapping
BANK_MARKETING_LABEL_ENCODER_PATH_TRAINING = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "BanMarketingLabelEncoding.npy"))
GERMAN_CREDIT_LABEL_ENCODER_PATH_TRAINING = os.path.abspath(os.path.join(__file__, "..", "..", "preprocessing", "raw_files", "GermanCreditLabelEncoding.npy"))
