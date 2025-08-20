from pathlib import Path

from EmotionAnalysis.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from EmotionAnalysis.utils.common_utils import read_yaml, create_directories
from EmotionAnalysis.entity import (
    DataIngestionConfig,
    CorruptedVideoRemovingConfig,
    TextCleaningConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PlotVisualizationConfig,
    LoggingConfig,
    FeatureExtractionConfig_RoBERTa,
    FeatureExtractionConfig,
    FeatureCombiningConfig,
    DataMergingConfig,
    TextFeatureExtractionConfig,
    AudioFeatureExtractionConfig,
    VideoFramesExtractionConfig
)


class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)   # Returns ConfigBox
        self.params = read_yaml(params_file_path)   # Returns ConfigBox

        # Create artifacts root
        create_directories([self.config.PROJECT.ARTIFACTS_ROOT])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.RAW_DATA
        return DataIngestionConfig(
            ROOT_DIR=Path(cfg.WORKING_DIR),
            RAW_DATA=cfg,
            INITIAL_JSON_FILES=self.config.INITIAL_JSON_FILES,
            VIDEO_PATHS=self.config.VIDEO_PATHS,
            AUDIO_PATHS=self.config.AUDIO_PATHS
        )

    def get_data_transformation_configuration(self) -> DataTransformationConfig:
        cfg = self.config.PIPELINE.TRANSFORMATION
        create_directories([cfg.OUTPUT_DIR])
        return DataTransformationConfig(
            ROOT_DIR=cfg.OUTPUT_DIR,
            ENABLED=cfg.ENABLED
        )

    def get_model_trainer_configuration(self) -> ModelTrainerConfig:
        cfg = self.config.PIPELINE.TRAINING
        params = self.params.TRAINING_ARGUMENTS

        create_directories([cfg.OUTPUT_DIR])

        return ModelTrainerConfig(
            ROOT_DIR=cfg.OUTPUT_DIR,
            ENABLED=cfg.ENABLED,
            MODEL_TYPE=self.config.EXPERIMENT.MODEL_TYPE,
            DESCRIPTION=self.config.EXPERIMENT.DESCRIPTION,

            # RoBERTa
            BATCH_SIZE_ROBERTA=params.RoBERTa.Training.BATCH_SIZE,

            # SlowFast
            NUM_FRAMES_SLOWFAST=params.SlowFast.MODEL.NUM_FRAMES,
            CROP_SIZE_SLOWFAST=params.SlowFast.MODEL.CROP_SIZE,
            DATASET_MEAN_SLOWFAST=params.SlowFast.MODEL.DATASET_MEAN,
            DATASET_STD_SLOWFAST=params.SlowFast.MODEL.DATASET_STD,
            SEED_SLOWFAST=params.SlowFast.TRAINING.SEED,
            BATCH_SIZE_SLOWFAST=params.SlowFast.TRAINING.BATCH_SIZE,
            MAX_EPOCHS_SLOWFAST=params.SlowFast.TRAINING.MAX_EPOCHS,
            MIN_EPOCHS_SLOWFAST=params.SlowFast.TRAINING.MIN_EPOCHS,
            NO_IMPROVEMENT_THRESHOLD_SLOWFAST=params.SlowFast.TRAINING.NO_IMPROVEMENT_THRESHOLD,
            PATIENCE_SLOWFAST=params.SlowFast.TRAINING.PATIENCE,
            CHECKPOINT_FREQUENCY_SLOWFAST=params.SlowFast.TRAINING.CHECKPOINT_FREQUENCY,
            BASE_LR_SLOWFAST=params.SlowFast.OPTIMIZER.BASE_LR,
            MAX_LR_SLOWFAST=params.SlowFast.OPTIMIZER.MAX_LR,
            GRAD_CLIP_SLOWFAST=params.SlowFast.SCHEDULER.GRAD_CLIP,
            ACCUMULATION_STEPS_SLOWFAST=params.SlowFast.SCHEDULER.ACCUMULATION_STEPS,
            WEIGHT_DECAY_SLOWFAST=params.SlowFast.OPTIMIZER.WEIGHT_DECAY,
            RESIZE_SIZE_SLOWFAST=params.SlowFast.DATA.RESIZE_SIZE,
            LOG_SAMPLES_FREQ_SLOWFAST=params.SlowFast.TRAINING.LOG_SAMPLES_FREQ,
            INPUT_DATA_PATH_SLOWFAST=params.SlowFast.DATA.INPUT_DATA_PATH,
            CLASS_NAME_SLOWFAST=params.SlowFast.DATA.CLASS_NAME,

            # BiLSTM
            BATCH_SIZE_BILSTM=params.BiLSTM.TRAINING.BATCH_SIZE,
            MAX_EPOCH_BILSTM=params.BiLSTM.TRAINING.MAX_EPOCH,
            LEARNING_RATE_BILSTM=params.BiLSTM.OPTIMIZER.LEARNING_RATE,
            GAMMA_BASE_BILSTM=params.BiLSTM.LOSS.GAMMA_BASE,
            WEIGHT_DECAY_BILSTM=params.BiLSTM.OPTIMIZER.WEIGHT_DECAY,
            HIDDEN_SIZE_BILSTM=params.BiLSTM.ARCHITECTURE.HIDDEN_SIZE,
            NUM_LAYERS_BILSTM=params.BiLSTM.ARCHITECTURE.NUM_LAYERS,
            DROPOUT_BILSTM=params.BiLSTM.ARCHITECTURE.DROPOUT,
            PATIENCE_BILSTM=params.BiLSTM.TRAINING.PATIENCE,
            MINORITY_CLASSES_BILSTM=params.BiLSTM.TRAINING.MINORITY_CLASSES,
            MODALITY_DIMS_BILSTM=params.BiLSTM.ARCHITECTURE.MODALITY_DIMS,
            OUTPUT_SIZE_BILSTM=params.BiLSTM.ARCHITECTURE.OUTPUT_SIZE,
            DATA_PATH_BILSTM=params.BiLSTM.DATA.DATA_PATH,
            OUTPUT_PATH_BILSTM=params.BiLSTM.DATA.OUTPUT_PATH,
            MODEL_SAVE_PATH_BILSTM=params.BiLSTM.DATA.MODEL_SAVE_PATH,
            NAME_BILSTM=params.BiLSTM.ARCHITECTURE.NAME,
            SMOOTHING_BILSTM=params.BiLSTM.LOSS.SMOOTHING,
            ALPHA_BILSTM=params.BiLSTM.LOSS.ALPHA,
            CLASS_COUNTS_BiLSTM=params.BiLSTM.DATA.CLASS_COUNTS_BiLSTM,

            # GNN
            BATCH_SIZE_GNN=params.GNN.TRAINING.BATCH_SIZE,
            HIDDEN_DIM_GNN=params.GNN.MODEL.HIDDEN_DIM,
            OUTPUT_DIM_GNN=params.GNN.MODEL.OUTPUT_DIM,
            DROPOUT_GNN=params.GNN.MODEL.DROPOUT,
            EDGE_DROPOUT_GNN=params.GNN.MODEL.EDGE_DROPOUT,
            NUM_SPEAKERS_GNN=params.GNN.MODEL.NUM_SPEAKERS,
            GAMMA_BASE_GNN=params.GNN.LOSS.GAMMA_BASE,
            SMOOTHING_GNN=params.GNN.LOSS.SMOOTHING,
            ALPHA_GNN=params.GNN.LOSS.ALPHA,
            BETA_GNN=params.GNN.LOSS.BETA,
            PENALTY_FACTOR_GNN=params.GNN.LOSS.PENALTY_FACTOR,
            MINORITY_CLASSES_GNN=self.params.EVALUATION_ARGUMENTS.GNN.EVALUATION.MINORITY_CLASSES,
            LR_GNN=params.GNN.OPTIMIZER.LR,
            WEIGHT_DECAY_GNN=params.GNN.OPTIMIZER.WEIGHT_DECAY,
            WARMUP_EPOCHS_GNN=params.GNN.SCHEDULER.WARMUP_EPOCHS,
            ACCUMULATION_STEPS_GNN=params.GNN.TRAINING.ACCUMULATION_STEPS,
            MAX_EPOCHS_GNN=params.GNN.TRAINING.MAX_EPOCHS,
            PATIENCE_GNN=params.GNN.TRAINING.PATIENCE,
            CLASS_COUNTS_GNN=params.GNN.DATA.CLASS_COUNTS_GNN,
            DATA_PATH_GNN=params.GNN.DATA.DATA_PATH,
            MODEL_SAVE_PATH_GNN=params.GNN.DATA.MODEL_SAVE_PATH,
            LABEL_MAP_GNN=params.GNN.LABEL_MAP,

            # Logging
            LOGGING_LEVEL=self.config.LOGGING.LEVEL,
            LOG_TO_FILE=self.config.LOGGING.LOG_TO_FILE,
            LOG_FILE_PATH=self.config.LOGGING.LOG_FILE_PATH,

            # Visualization
            BG_COLOR=self.config.VISUALIZATION.THEME.BACKGROUND_COLOR,
            TEXT_COLOR=self.config.VISUALIZATION.THEME.TEXT_COLOR,
            LABEL_COLOR=self.config.VISUALIZATION.THEME.LABEL_COLOR,
            TITLE_COLOR=self.config.VISUALIZATION.THEME.TITLE_COLOR
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config.PIPELINE.EVALUATION
        create_directories([cfg.OUTPUT_DIR])
        return ModelEvaluationConfig(
            ROOT_DIR=cfg.OUTPUT_DIR,
            ENABLED=cfg.ENABLED,
            MINORITY_CLASSES=self.params.EVALUATION_ARGUMENTS.GNN.EVALUATION.MINORITY_CLASSES,
            SEEDS=self.params.EVALUATION_ARGUMENTS.GNN.EVALUATION.SEEDS,
            MODEL_TYPE=self.config.EXPERIMENT.MODEL_TYPE,
            LOGGING_LEVEL=self.config.LOGGING.LEVEL,
            LOG_TO_FILE=self.config.LOGGING.LOG_TO_FILE,
            LOG_FILE_PATH=self.config.LOGGING.LOG_FILE_PATH
        )

    def get_plot_visualization_config(self) -> PlotVisualizationConfig:
        cfg = self.config.VISUALIZATION
        return PlotVisualizationConfig(
            BACKGROUND_COLOR=cfg.THEME.BACKGROUND_COLOR,
            TEXT_COLOR=cfg.THEME.TEXT_COLOR,
            LABEL_COLOR=cfg.THEME.LABEL_COLOR,
            TITLE_COLOR=cfg.THEME.TITLE_COLOR
        )
    
    def get_logger_config(self) -> LoggingConfig:
        cfg = self.config.LOGGING
        return LoggingConfig(
            LEVEL=cfg.LEVEL,
            LOG_TO_FILE=cfg.LOG_TO_FILE,
            LOG_FILE_PATH=cfg.LOG_FILE_PATH
        )
    
    def get_feature_extraction_RoBERTa_config(self) -> FeatureExtractionConfig_RoBERTa:
        params = self.params.FEATURE_EXTRACTION_ARGUMENTS
        return FeatureExtractionConfig_RoBERTa(
            TOKENIZER_NAME=params.TOKENIZER_NAME,
            MODEL_NAME=params.MODEL_NAME,
            PADDING_STRATEGY=params.PADDING_STRATEGY,
            BATCH_SIZE=params.BATCH_SIZE,
            MAX_LENGTH=params.MAX_LENGTH,
            TRUNCATION=params.TRUNCATION,
            ADD_SPECIAL_TOKENS=params.ADD_SPECIAL_TOKENS
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config
        
        return DataIngestionConfig(
            root_dir=Path(config.raw_data.working_dir),
            raw_data=config.raw_data,
            initial_json_files=config.initial_json_files,
            video_paths=config.video_paths,
            audio_paths=config.audio_paths
        )
    
    def get_corrupted_video_removing_config(self) -> CorruptedVideoRemovingConfig:
        cfg = self.config.RAW_DATA
        return CorruptedVideoRemovingConfig(
            ROOT_DIR=Path(cfg.WORKING_DIR),
            REMOVED_CORRUPTED_DATA=self.config.REMOVED_CORRUPTED_DATA,
            DATASET=self.config.DATASET,
            VIDEO_PATHS=self.config.VIDEO_PATHS,
            AUDIO_PATHS=self.config.AUDIO_PATHS
        )


    def get_text_cleaning_config(self) -> TextCleaningConfig:
        cfg = self.config.CLEANING
        params = self.params.DATA_CLEANING
        return TextCleaningConfig(
            INPUT_PATHS=cfg.INPUT_PATHS,
            OUTPUT_DIR=cfg.OUTPUT_DIR,
            ENCODING=params.ENCODING
        )


    def get_data_merging_config(self) -> DataMergingConfig:
        cfg = self.config.FINAL_JSON
        labels = self.config.DATASET.LABEL_INDEX
        base_output_dir = Path(self.config.CLEANING.OUTPUT_DIR)
        return DataMergingConfig(
            TRAIN_PATH=base_output_dir / "train_sent_emo_processed.csv",
            DEV_PATH=base_output_dir / "dev_sent_emo_processed.csv",
            TEST_PATH=base_output_dir / "test_sent_emo_processed.csv",
            MELD_DATA_PATH=cfg.MELD_DATA,
            MELD_TEXTUAL_PATH=cfg.MELD_TEXTUAL,
            LABEL_INDEX=labels
        )


    def get_text_feature_config(self) -> TextFeatureExtractionConfig:
        cfg = self.config.TEXT_FEATURES
        params = self.params.TEXT_FEATURES
        return TextFeatureExtractionConfig(
            INPUT_JSON=self.config.FINAL_JSON.MELD_TEXTUAL,
            OUTPUT_JSON=cfg.OUTPUT,
            TOKENIZER_NAME=params.TOKENIZER_NAME,
            PADDING_STRATEGY=params.PADDING_STRATEGY,
            BATCH_SIZE=params.BATCH_SIZE
        )


    def get_audio_feature_config(self) -> AudioFeatureExtractionConfig:
        cfg = self.config.AUDIO_FEATURES
        params = self.params.AUDIO_FEATURES
        return AudioFeatureExtractionConfig(
            INPUT_JSON=self.config.FINAL_JSON.MELD_AUDIO,
            OUTPUT_JSON=cfg.OUTPUT,
            FEATURE_SET=params.FEATURE_SET,
            FEATURE_LEVEL=params.FEATURE_LEVEL,
            DEVICE=params.DEVICE
        )


    def get_video_feature_config(self) -> VideoFramesExtractionConfig:
        cfg = self.config.VIDEO_FEATURES
        params = self.params.VIDEO_FEATURES
        return VideoFramesExtractionConfig(
            INPUT_JSON=cfg.INPUT_JSON,
            OUTPUT_DIR=cfg.OUTPUT_DIR,
            CHECKPOINT_FILE=cfg.CHECKPOINT,
            EMERGENCY_DIR=cfg.EMERGENCY_DIR,
            NUM_FRAMES=params.NUM_FRAMES,
            CROP_SIZE=tuple(params.CROP_SIZE),
            FACE_CONFIDENCE=params.FACE_CONFIDENCE,
            SPEAKER_HISTORY=params.SPEAKER_HISTORY,
            SIMILARITY_THRESHOLD=params.SIMILARITY_THRESHOLD,
            CENTRALITY_WEIGHT=params.CENTRALITY_WEIGHT,
            MIN_SPEAKING_FRAMES=params.MIN_SPEAKING_FRAMES,
            BBOX_EXPAND_RATIO=params.BBOX_EXPAND_RATIO,
            TRACKING_QUALITY_THRESHOLD=params.TRACKING_QUALITY_THRESHOLD,
            MAX_TRACKING_FAILURES=params.MAX_TRACKING_FAILURES,
            MAX_RETRIES=params.MAX_RETRIES,
            CHECKPOINT_BACKUPS=params.CHECKPOINT_BACKUPS,
            FRAME_VALIDATION_THRESHOLD=params.FRAME_VALIDATION_THRESHOLD,
            MAX_RUNTIME=params.MAX_RUNTIME
        )