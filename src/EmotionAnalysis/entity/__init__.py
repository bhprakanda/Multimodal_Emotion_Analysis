from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    ROOT_DIR: Path
    ENABLED: bool   
    INITIAL_JSON_FILES: dict
    VIDEO_PATHS: dict
    AUDIO_PATHS: dict 


# Data Cleaning
class CorruptedVideoRemovingConfig:
    ROOT_DIR: Path 
    REMOVED_CORRUPTED_DATA: dict 
    DATASET: dict 
    VIDEO_PATHS: dict 
    AUDIO_PATHS: dict


# Data Transformation
@dataclass(frozen=True)
class DataTransformationConfig:
    ROOT_DIR: Path
    ENABLED: bool


# Model Training
@dataclass(frozen=True)
class ModelTrainerConfig:
    ROOT_DIR: Path
    ENABLED: bool
    MODEL_TYPE: str
    DESCRIPTION: str

    # ---------- RoBERTa ----------
    BATCH_SIZE_ROBERTA: int

    # ---------- SlowFast ----------
    NUM_FRAMES_SLOWFAST: int
    CROP_SIZE_SLOWFAST: List[int]
    DATASET_MEAN_SLOWFAST: List[float]
    DATASET_STD_SLOWFAST: List[float]
    SEED_SLOWFAST: int
    BATCH_SIZE_SLOWFAST: int
    MAX_EPOCHS_SLOWFAST: int
    MIN_EPOCHS_SLOWFAST: int
    NO_IMPROVEMENT_THRESHOLD_SLOWFAST: float
    PATIENCE_SLOWFAST: int
    CHECKPOINT_FREQUENCY_SLOWFAST: int
    BASE_LR_SLOWFAST: float
    MAX_LR_SLOWFAST: float
    GRAD_CLIP_SLOWFAST: float
    ACCUMULATION_STEPS_SLOWFAST: int
    WEIGHT_DECAY_SLOWFAST: float
    RESIZE_SIZE_SLOWFAST: List[int]
    LOG_SAMPLES_FREQ_SLOWFAST: int
    INPUT_DATA_PATH_SLOWFAST: str
    CLASS_NAME_SLOWFAST: List[str]

    # ---------- BiLSTM ----------
    BATCH_SIZE_BILSTM: int
    MAX_EPOCH_BILSTM: int
    LEARNING_RATE_BILSTM: float
    GAMMA_BASE_BILSTM: float
    WEIGHT_DECAY_BILSTM: float
    HIDDEN_SIZE_BILSTM: int
    NUM_LAYERS_BILSTM: int
    DROPOUT_BILSTM: float
    PATIENCE_BILSTM: int
    MINORITY_CLASSES_BILSTM: List[int]
    MODALITY_DIMS_BILSTM: List[int]
    OUTPUT_SIZE_BILSTM: int
    DATA_PATH_BILSTM: str
    OUTPUT_PATH_BILSTM: str
    MODEL_SAVE_PATH_BILSTM: str
    NAME_BILSTM: str
    SMOOTHING_BILSTM: float
    ALPHA_BILSTM: float
    CLASS_COUNTS_BiLSTM: int

    # ---------- GNN ----------
    BATCH_SIZE_GNN: int
    HIDDEN_DIM_GNN: int
    OUTPUT_DIM_GNN: int
    DROPOUT_GNN: float
    EDGE_DROPOUT_GNN: float
    NUM_SPEAKERS_GNN: Optional[int]
    GAMMA_BASE_GNN: float
    CLASS_COUNTS_GNN: int
    SMOOTHING_GNN: float
    ALPHA_GNN: float
    BETA_GNN: float
    PENALTY_FACTOR_GNN: float
    MINORITY_CLASSES_GNN: List[int]
    LR_GNN: float
    WEIGHT_DECAY_GNN: float
    WARMUP_EPOCHS_GNN: int
    ACCUMULATION_STEPS_GNN: int
    MAX_EPOCHS_GNN: int
    PATIENCE_GNN: int
    CLASS_COUNTS_GNN: int
    DATA_PATH_GNN: str
    MODEL_SAVE_PATH_GNN: str
    LABEL_MAP_GNN: Dict[int, str]

    # ---------- Logging ----------
    LOGGING_LEVEL: str
    LOG_TO_FILE: bool
    LOG_FILE_PATH: Path

    # ---------- Visualization ----------
    BG_COLOR: str
    TEXT_COLOR: str
    LABEL_COLOR: str
    TITLE_COLOR: str


# Model Evaluation
@dataclass(frozen=True)
class ModelEvaluationConfig:
    ROOT_DIR: Path
    ENABLED: bool
    MINORITY_CLASSES: List[int]
    SEEDS: List[int]
    MODEL_TYPE: str
    LOGGING_LEVEL: str
    LOG_TO_FILE: bool
    LOG_FILE_PATH: Path


# Visualization & Logging
@dataclass(frozen=True)
class PlotVisualizationConfig:
    BACKGROUND_COLOR: str
    TEXT_COLOR: str
    LABEL_COLOR: str
    TITLE_COLOR: str


@dataclass(frozen=True)
class LoggingConfig:
    LEVEL: str
    LOG_TO_FILE: bool
    LOG_FILE_PATH: str


# Feature Extraction
@dataclass
class FeatureExtractionConfig_RoBERTa:
    """Configuration class for text processing parameters."""
    TOKENIZER_NAME: str
    MODEL_NAME: str
    PADDING_STRATEGY: str
    BATCH_SIZE: int
    MAX_LENGTH: Optional[int]
    TRUNCATION: bool
    ADD_SPECIAL_TOKENS: bool


@dataclass
class FeatureExtractionConfig:
    CHECKPOINT_PATH: str
    MODEL_REPO_ID: str
    MODEL_FILENAME: str
    OUTPUT_PATH: str
    NUM_FRAMES: int
    CROP_SIZE: tuple
    RESIZE_SIZE: tuple
    DATASET_MEAN: list
    DATASET_STD: list
    BATCH_SIZE: int
    NUM_CLASSES: int


# Feature Combination
@dataclass
class FeatureCombiningConfig:
    TEXT_FEATURES_PATH: str
    AUDIO_FEATURES_PATH: str
    VIDEO_FEATURES_PATH: str
    MELD_CLEANED_PATH: str
    OUTPUT_COMBINED_PATH: str
    OUTPUT_ENHANCED_PATH: str


# Text Cleaning
@dataclass(frozen=True)
class TextCleaningConfig:
    INPUT_PATHS: dict
    OUTPUT_DIR: str
    ENCODING: str


# Data Merging
@dataclass(frozen=True)
class DataMergingConfig:
    TRAIN_PATH: Path
    DEV_PATH: Path
    TEST_PATH: Path
    MELD_DATA_PATH: str
    MELD_TEXTUAL_PATH: str
    LABEL_INDEX: dict


# Text Feature Extraction
@dataclass(frozen=True)
class TextFeatureExtractionConfig:
    INPUT_JSON: str
    OUTPUT_JSON: str
    TOKENIZER_NAME: str
    PADDING_STRATEGY: str
    BATCH_SIZE: int


# Audio Feature Extraction
@dataclass(frozen=True)
class AudioFeatureExtractionConfig:
    INPUT_JSON: str
    OUTPUT_JSON: str
    FEATURE_SET: str
    FEATURE_LEVEL: str
    DEVICE: str


# Video Feature Extraction
@dataclass(frozen=True)
class VideoFramesExtractionConfig:
    INPUT_JSON: str
    OUTPUT_DIR: str
    CHECKPOINT_FILE: str
    EMERGENCY_DIR: str
    NUM_FRAMES: int
    CROP_SIZE: Tuple[int, int]
    FACE_CONFIDENCE: float
    SPEAKER_HISTORY: int
    SIMILARITY_THRESHOLD: float
    CENTRALITY_WEIGHT: float
    MIN_SPEAKING_FRAMES: int
    BBOX_EXPAND_RATIO: float
    TRACKING_QUALITY_THRESHOLD: float
    MAX_TRACKING_FAILURES: int
    MAX_RETRIES: int
    CHECKPOINT_BACKUPS: int
    FRAME_VALIDATION_THRESHOLD: float
    MAX_RUNTIME: int