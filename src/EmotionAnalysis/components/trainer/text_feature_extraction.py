import logging


from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.components.feature_extractor.text_feature_extraction_RoBERTa import TextFeatureExtractor
from EmotionAnalysis.entity import TextFeatureExtractionConfig

class Text_Feature_Extraction_Trainer:
    def __init__(self, config_path, params_path):
        self.config_manager = ConfigurationManager(config_path, params_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Text Feature Extraction
            self.logger.info("Starting text feature extraction")
            text_extractor = TextFeatureExtractor(self.config_manager.get_text_feature_config())
            text_extractor.run()
            
            self.logger.info("All processing completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise