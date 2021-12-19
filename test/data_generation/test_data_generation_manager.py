
class TestDatagenerationManager:
    def test_data_generation(self):
        bank_marketing_info = DataDriftDetectionManagerInfo(
            deployment_dataset_plus=BankMarketingDataset(),
            training_processed_df_plus_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PLUS_PATH,
            preprocessor=Preprocessor(),  # TODO: fix
            model=BankMarketingProductionModel(),  # TODO: fix
            deployment_dataset=BankMarketingDataset(),
            training_feature_metrics_list_path=BANK_MARKETING_TRAINING_FEATURE_METRIC_LIST_PATH,
            training_processed_df_path=BANK_MARKETING_TRAINING_PROCESSED_DF_PATH
        )
        data_drift_detection_manager = MultipleDatasetDataDriftDetectionManager(info_list=[bank_marketing_info])
        data_drift: DataDrift = data_drift_detection_manager.manage()[0]
        assert not data_drift.is_drifted
