# Imports
import os
import json
import yaml
import random
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi

from EmotionAnalysis.utils.logger import logger


# File I/O Utilities
@ensure_annotations
def save_to_json(data: Any, file_path: Path) -> bool:
    """
    Save a Python object to a JSON file.

    Args:
        data (Any): Data to serialize.
        file_path (Path): Destination file path.

    Returns:
        bool: True on success, False otherwise.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Failed to save JSON: {e}")
        return False


@ensure_annotations
def read_from_json(file_path: Path) -> Any:
    """
    Load a JSON file into a Python object.

    Args:
        file_path (Path): Path to JSON file.

    Returns:
        Any: Parsed object or None on failure.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return None


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return a ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file.

    Returns:
        ConfigBox: Parsed configuration.

    Raises:
        ValueError: If YAML is empty.
        Exception: On general failure.
    """
    try:
        with open(path_to_yaml, "r") as f:
            content = yaml.safe_load(f)
            logger.info(f"YAML file loaded: {path_to_yaml}")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"The YAML file at '{path_to_yaml}' is empty.")
    except Exception as e:
        raise e


# Directory Management
@ensure_annotations
def create_directories(paths: list[Path], verbose: bool = True) -> None:
    """
    Create directories if they don't already exist.

    Args:
        paths (list[Path]): List of paths to create.
        verbose (bool): Enable logging output.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


@ensure_annotations
def get_file_size(path: Path) -> str:
    """
    Get file size in kilobytes.

    Args:
        path (Path): File path.

    Returns:
        str: File size as string (e.g., "234 KB").
    """
    size_kb = round(os.path.getsize(path) / 1024)
    return f"{size_kb} KB"


# API & Authentication
def setup_api_tokens() -> tuple[str, str, HfApi]:
    """
    Authenticate with HuggingFace and Weights & Biases.

    Returns:
        tuple: HF token, W&B token, and HuggingFace API instance.
    """
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
    wandb_token = secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_token)
    return hf_token, wandb_token, HfApi(token=hf_token)


def init_api_tokens() -> tuple[str, str]:
    """
    Initialize API tokens using Kaggle Secrets.

    Returns:
        tuple: HuggingFace and W&B tokens.
    """
    user_secrets = UserSecretsClient()
    return user_secrets.get_secret("HF_TOKEN"), user_secrets.get_secret("WANDB_API_KEY")


def get_api_tokens() -> dict:
    """
    Retrieve API tokens from environment variables.

    Returns:
        dict: Contains 'HF_TOKEN' and 'WANDB_API_KEY'.
    """
    return {
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
    }


# Reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set seeds for random, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    Ensure determinism for data loader workers.

    Args:
        worker_id (int): Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# Model Utilities
def compute_class_weights(labels: list[int]) -> tuple[torch.Tensor, np.ndarray]:
    """
    Compute class weights to address class imbalance.

    Args:
        labels (list[int]): List of integer class labels.

    Returns:
        tuple: 
            - torch.Tensor: Normalized class weights.
            - np.ndarray: Raw class counts.
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts + 1e-9)
    normalized = weights / weights.sum() * len(classes)
    return torch.tensor(normalized, dtype=torch.float32), counts


# Warning Suppression
def suppress_warnings() -> None:
    """
    Suppress common warning messages (e.g., FutureWarnings, UserWarnings).
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


import pandas as pd
from pathlib import Path
from typing import Union


def save_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False, encoding: str = "utf-8") -> None:
    """
    Saves a DataFrame to a CSV file at the specified path.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Union[str, Path]): File path where the CSV will be saved.
        index (bool): Whether to include the DataFrame index. Defaults to False.
        encoding (str): Encoding type. Defaults to "utf-8".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)
    print(f"Saved: {path}")


def load_csv(path: Union[str, Path], encoding: str = "utf-8") -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Args:
        path (Union[str, Path]): Path to the CSV file.
        encoding (str): Encoding type. Defaults to "utf-8".

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    df = pd.read_csv(path, encoding=encoding)
    print(f"Loaded: {path}")
    return df