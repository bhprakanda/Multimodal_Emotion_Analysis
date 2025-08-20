from EmotionAnalysis.utils.config_manager import ConfigurationManager
from src.data_processing.data_cleaner import DataCleaner, DataCleaningConfig
from src.data_processing.data_merger import DataMerger, DataMergingConfig
from src.feature_extraction.text_features import TextFeatureExtractor, TextFeatureExtractionConfig
from src.feature_extraction.audio_features import AudioFeatureExtractor, AudioFeatureExtractionConfig
from src.feature_extraction.video_features import VideoProcessor, VideoFeatureExtractionConfig
import logging

class Trainer:
    def __init__(self, config_path, params_path):
        self.config_manager = ConfigurationManager(config_path, params_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Data Cleaning
            self.logger.info("Starting data cleaning")
            cleaner = DataCleaner(self.config_manager.get_data_cleaning_config())
            cleaner.clean_files()
            
            # Data Merging
            self.logger.info("Starting data merging")
            merger = DataMerger(self.config_manager.get_data_merging_config())
            merger.merge_data()
            
            # Text Feature Extraction
            self.logger.info("Starting text feature extraction")
            text_extractor = TextFeatureExtractor(self.config_manager.get_text_feature_config())
            text_extractor.run()
            
            # Audio Feature Extraction
            self.logger.info("Starting audio feature extraction")
            audio_extractor = AudioFeatureExtractor(self.config_manager.get_audio_feature_config())
            audio_extractor.run()
            
            # Video Feature Extraction
            self.logger.info("Starting video feature extraction")
            video_processor = VideoProcessor(self.config_manager.get_video_feature_config())
            video_processor.run()
            
            self.logger.info("All processing completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise