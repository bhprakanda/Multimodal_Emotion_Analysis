import pandas as pd
import os
from dataclasses import dataclass
from pathlib import Path

from EmotionAnalysis.entity import TextCleaningConfig


class TextDataCleaner:
    def __init__(self, config: TextCleaningConfig):
        self.config = config
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def clean_files(self):
        for split, path in self.config.input_paths.items():
            df = pd.read_csv(path, encoding=self.config.encoding)
            df['Utterance'] = df['Utterance'].str.replace('Â', '', regex=False)
            
            output_path = Path(self.config.output_dir) / f"{split}_sent_emo_processed.csv"
            df.to_csv(output_path, index=False)
            print(f"Processed '{split}' data saved to: {output_path}")