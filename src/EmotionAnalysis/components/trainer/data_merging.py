import logging

from EmotionAnalysis.config import ConfigurationManager
from EmotionAnalysis.components.data_preparation.data_merging import DataMerger
from EmotionAnalysis.entity import DataMergingConfig


class Data_Merging_Trainer:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Data Merging
            self.logger.info("Starting data merging")
            merger = DataMerger(self.config_manager.get_data_merging_config())
            merger.merge_data()
            
            self.logger.info("All processing completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise