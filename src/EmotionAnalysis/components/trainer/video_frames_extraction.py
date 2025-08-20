import logging

from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.entity import VideoFeatureExtractionConfig
from EmotionAnalysis.components.feature_extractor.speaker_frame_extraction import VideoFeatureExtractor


class Video_Feature_Extraction_Trainer:
    def __init__(self, config_path, params_path):
        self.config_manager = ConfigurationManager(config_path, params_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def run_pipeline(self):
        try:
            # Video Frame Extraction
            self.logger.info("Starting video frame extraction")
            video_extractor = VideoFeatureExtractor(
                self.config_manager.get_video_feature_config()
            )
            video_extractor.run()
            
            self.logger.info("Video frame extraction completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Video frame extraction failed: {str(e)}")
            raise