import logging


from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.entity import TextCleaningConfig
from EmotionAnalysis.components.data_cleaning.text_cleaning import TextDataCleaner

class Text_Cleaning_Trainer:
    def __init__(self, config_path, params_path):
        self.config_manager = ConfigurationManager(config_path, params_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Text Data Cleaning
            self.logger.info("Starting data cleaning")
            cleaner = TextDataCleaner(self.config_manager.get_data_cleaning_config())
            cleaner.clean_files()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise