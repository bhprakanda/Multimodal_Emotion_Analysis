import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

project_name = "EmotionAnalysis"

list_of_filepaths = [
    ".github/workflow/.gitkeep",  # Placeholder to keep the .github/workflow directory in version control

    "artifacts/.gitkeep",  # Directory to store generated files like models, logs, or intermediate results

    f"src/{project_name}/__init__.py",  # Marks the src/{project_name} directory as a Python package

    f"src/{project_name}/components/__init__.py",  # Marks the components directory as a Python package
    f"src/{project_name}/components/data_ingestion.py",  # Data collection components
    f"src/{project_name}/components/data_validation.py",  # Data validation components
    f"src/{project_name}/components/data_transformation.py",  # Data transformation components
    f"src/{project_name}/components/model_trainer.py",  # Model training components
    f"src/{project_name}/components/model_evaluation.py",  # Model evaluation components

    f"src/{project_name}/data/dataset.py",  # Handles dataset loading and preprocessing
    f"src/{project_name}/data/transforms.py",  # Contains data transformation and augmentation utilities

    f"src/{project_name}/models/base_model.py",  # Base model class for extending different models
    f"src/{project_name}/models/custom_model.py",  # Custom model implementation for specific tasks

    f"src/{project_name}/training/trainer.py",  # Handles the training pipeline
    f"src/{project_name}/training/loss.py",  # Contains loss functions
    f"src/{project_name}/training/optimizer.py",  # Configures optimizers
    f"src/{project_name}/training/scheduler.py",  # Configures learning rate schedulers

    f"src/{project_name}/evaluation/evaluation.py",  # Handles model evaluation processes
    f"src/{project_name}/evaluation/metrics.py",  # Defines evaluation metrics

    f"src/{project_name}/utils/common_utils.py",  # General-purpose utility functions
    f"src/{project_name}/utils/logger.py",  # Custom logger setup
    f"src/{project_name}/utils/config.py",  # Configuration management utilities
    f"src/{project_name}/utils/checkpoint.py",  # Model checkpointing utilities
    f"src/{project_name}/utils/visualizer.py",  # Visualization utilities (e.g., training curves)

    f"src/{project_name}/entity/__init__.py",  # Marks the entity directory as a Python package

    f"src/{project_name}/pipeline/__init__.py",  # Marks the pipeline directory as a Python package
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",  # Pipeline for data ingestion
    f"src/{project_name}/pipeline/stage_02_data_validation.py",  # Pipeline for data validation
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",  # Pipeline for data transformation
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",  # Pipeline for model training
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",  # Pipeline for model evaluation

    "scripts/install_system_deps.sh",  # System dependencies installation script
    "scripts/download_models.sh",  # Pre-trained model downloads script
    "scripts/setup_environment.sh",  # Complete environment setup script

    "logs/.gitkeep",  # Directory for storing logs

    "notebooks/.gitkeep",  # Directory for Jupyter notebooks used in experimentation

    "docs/documents/.gitkeep",  # Directory for architecture documentation files
    "docs/artifacts/.gitkeep",  # Directory for architecture diagrams and visualizations

    "configs/default.yaml",  # Default configuration settings
    "configs/train.yaml",  # Configuration for training
    "configs/eval.yaml",  # Configuration for evaluation
    
    "requirements-base.txt",  # Common dependencies for both CPU/GPU
    "requirements-cpu.txt",  # CPU-specific Python dependencies
    "requirements-cuda.txt",  # GPU-specific Python dependencies (CUDA)
    
    ".gitignore",  # Git ignore file
    "Dockerfile",  # Instructions for building a Docker image
    "main.py",  # Main entry point for the project
    "app.py",  # Application entry point (e.g., web app)
    "setup.py"  # Script for installing the project as a package
]

for filepath in list_of_filepaths:
    filepath = Path(filepath)
    file_directory, file_name = os.path.split(filepath)

    if file_directory != "":
        os.makedirs(file_directory, exist_ok=True)
        logging.info(f"Directory {file_directory} has been created.")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Empty file '{filepath}' has been created.")


logging.info(f"Project '{project_name}' has been successfully initialized.")