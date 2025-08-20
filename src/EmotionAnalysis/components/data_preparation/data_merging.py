import pandas as pd
import json
from dataclasses import dataclass
from pathlib import Path

from EmotionAnalysis.utils.common_utils import save_to_json
from EmotionAnalysis.entity import DataMergingConfig


class DataMerger:
    def __init__(self, config: DataMergingConfig):
        self.config = config
    
    def merge_data(self):
        # Read CSV files
        train_df = pd.read_csv(self.config.train_path)
        dev_df = pd.read_csv(self.config.dev_path)
        test_df = pd.read_csv(self.config.test_path)
        
        # Concatenate data
        df = pd.concat([train_df, dev_df, test_df])
        
        # Initialize data structure
        MELD_data = {
            "data": [],
            "max_sentence_length": 30,
            "label_index": self.config.label_index
        }
        
        # Helper function to find split
        def get_split(row):
            row_without_srno = row.drop('Sr No.')
            if (train_df.drop('Sr No.', axis=1) == row_without_srno).all(axis=1).any():
                return "train"
            elif (dev_df.drop('Sr No.', axis=1) == row_without_srno).all(axis=1).any():
                return "dev"
            elif (test_df.drop('Sr No.', axis=1) == row_without_srno).all(axis=1).any():
                return "test"
            return None
        
        # Process each row
        for _, row in df.iterrows():
            split = get_split(row)
            if split is None:
                print(f"Row mismatch: {row['Season']}, {row['Episode']}, {row['Dialogue_ID']}, {row['Utterance_ID']}")
                continue
            
            entry = {
                "text": row["Utterance"],
                "split": split,
                "y": self.config.label_index.get(row["Emotion"].lower(), -1),
                "dialog": row["Dialogue_ID"],
                "utterance": row["Utterance_ID"],
                "season": row["Season"],
                "episode": row["Episode"],
                "num_words": len(row["Utterance"].split()),
                "dia_utt": f"{row['Dialogue_ID']}_{row['Utterance_ID']}",
                "speaker": row["Speaker"]
            }
            MELD_data["data"].append(entry)
        
        # Save MELD data
        save_to_json(MELD_data, self.config.meld_data_path)
        print(f"MELD data saved to: {self.config.meld_data_path}")
        
        # Create textual data
        MELD_textual_data = {"train": [], "dev": [], "test": []}
        for entry in MELD_data["data"]:
            text_key = entry["dia_utt"]
            emotion_label = list(self.config.label_index.keys())[entry["y"]]
            text_entry = {
                text_key: entry["text"],
                "y": entry["y"],
                "label": emotion_label
            }
            MELD_textual_data[entry["split"]].append(text_entry)
        
        # Save textual data
        save_to_json(MELD_textual_data, self.config.meld_textual_path)
        print(f"Textual data saved to: {self.config.meld_textual_path}")