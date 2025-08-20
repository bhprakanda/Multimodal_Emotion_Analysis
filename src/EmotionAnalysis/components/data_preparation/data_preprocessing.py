import json
import torch
import numpy as np

def load_and_preprocess_data(data_path):
    with open(data_path, "r") as f:
        original_data = json.load(f)
    
    for split in ['train', 'dev', 'test']:
        for item in original_data[split]:
            for key in item:
                if key.endswith("_Concatenated"):
                    parts = key.split("_")
                    item["dialogue_id"] = parts[0]
                    item["utt_id"] = parts[1]
                    break
    return original_data

def compute_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts + 1e-9)
    normalized_weights = weights / weights.sum() * len(classes)
    return torch.tensor(normalized_weights, dtype=torch.float32), counts