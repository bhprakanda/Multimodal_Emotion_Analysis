from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.utils.common_utils import create_directories
from EmotionAnalysis.components.data_ingestion.data_ingestion import DataIngestion
from EmotionAnalysis.components.data_cleaning.remove_corrupted_video_audio import CorruptedVideoRemover


def main():
    # Setup configuration
    config_manager = ConfigurationManager()
    
    # Data Ingestion
    ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(ingestion_config)
    data_ingestion.run()
    
    # Remove Corrupted Video and Audio Data
    cleaning_config = config_manager.get_corrupted_video_removing_config()
    data_cleaning = CorruptedVideoRemover(cleaning_config)
    data_cleaning.run()

if __name__ == '__main__':
    main()