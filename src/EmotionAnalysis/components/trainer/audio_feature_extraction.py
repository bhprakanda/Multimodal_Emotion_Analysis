import logging

from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.components.feature_extractor.audio_feature_extraction_OpenSMILE import AudioFeatureExtractor

class Audio_Feature_Extraction_Trainer:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Audio Feature Extraction
            self.logger.info("Starting audio feature extraction")
            audio_extractor = AudioFeatureExtractor(self.config_manager.get_audio_feature_config())
            audio_extractor.run()

            self.logger.info("All processing completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise