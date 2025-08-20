import os
import json
import pandas as pd
from moviepy.editor import VideoFileClip
from pathlib import Path

from EmotionAnalysis.utils.common_utils import read_from_json, save_to_json, load_csv
from EmotionAnalysis.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def extract_audio(self):
        # Install ffmpeg
        os.system("apt-get install -y ffmpeg")
        
        def extract_audio_from_video(video_path, output_audio_path):
            try:
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(output_audio_path)
                video.close()
            except Exception as e:
                print(f"Error extracting audio: {e}")

        # Load video data
        video_data_path = Path(self.config.raw_data.working_dir) / self.config.initial_json_files.meld_video_data
        MELD_video_data = read_from_json(video_data_path)

        # Process data
        meld_audio_data = {}
        for dataset_type, videos in MELD_video_data.items():
            output_folder = Path(self.config.audio_paths.working_dir) / f"audio_{dataset_type}"
            os.makedirs(output_folder, exist_ok=True)
            
            meld_audio_data[dataset_type] = []
            
            for video_entry in videos:
                for key, video_path in video_entry.items():
                    if key not in ["y", "label"]:
                        audio_file = f"{Path(video_path).stem}.wav"
                        output_path = output_folder / audio_file
                        
                        extract_audio_from_video(video_path, str(output_path))
                        
                        meld_audio_data[dataset_type].append({
                            key: str(output_path),
                            "y": video_entry["y"],
                            "label": video_entry["label"]
                        })
        
        # Save audio data
        audio_output_path = Path(self.config.raw_data.working_dir) / self.config.initial_json_files.meld_audio_video
        save_to_json(meld_audio_data, audio_output_path)
        
    def create_base_datasets(self):
        # Load CSVs
        train_df = pd.read_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.train
        )
        dev_df = pd.read_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.dev
        )
        test_df = pd.read_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.test
        )
        
        # Combine datasets
        df = pd.concat([train_df, dev_df, test_df])
        
        # Create MELD data structure
        MELD_data = {
            "data": [],
            "max_sentence_length": self.config.dataset.max_sentence_length,
            "label_index": self.config.dataset.label_index
        }
        
        # Process rows
        for _, row in df.iterrows():
            entry = {
                "text": row["Utterance"],
                "split": "train" if row["Dialogue_ID"] in train_df["Dialogue_ID"].values else 
                         "dev" if row["Dialogue_ID"] in dev_df["Dialogue_ID"].values else "test",
                "y": self.config.dataset.label_index[row["Emotion"].lower()],
                "dialog": row["Dialogue_ID"],
                "utterance": row["Utterance_ID"],
                "season": row["Season"],
                "episode": row["Episode"],
                "num_words": len(row["Utterance"].split()),
                "dia_utt": f"{row['Dialogue_ID']}_{row['Utterance_ID']}",
                "speaker": row["Speaker"]
            }
            MELD_data["data"].append(entry)
        
        # Save datasets
        save_to_json(MELD_data, Path(self.config.raw_data.working_dir) / self.config.initial_json_files.meld_data)
        
        # Create textual data
        MELD_textual_data = {"train": [], "dev": [], "test": []}
        for entry in MELD_data["data"]:
            split = entry["split"]
            text_entry = {
                entry["dia_utt"]: entry["text"],
                "y": entry["y"],
                "label": next(k for k,v in self.config.dataset.label_index.items() if v == entry["y"])
            }
            MELD_textual_data[split].append(text_entry)
        
        save_to_json(
            MELD_textual_data,
            Path(self.config.raw_data.working_dir) / self.config.initial_json_files.meld_textual_data
        )
        
        # Create video data
        MELD_video_data = {"train": [], "dev": [], "test": []}
        for entry in MELD_data["data"]:
            split = entry["split"]
            video_filename = f"dia{entry['dia_utt'].replace('_', '_utt')}.mp4"
            video_path = Path(self.config.video_paths[split]) / video_filename
            
            video_entry = {
                entry["dia_utt"]: str(video_path),
                "y": entry["y"],
                "label": next(k for k,v in self.config.dataset.label_index.items() if v == entry["y"])
            }
            MELD_video_data[split].append(video_entry)
        
        save_to_json(
            MELD_video_data,
            Path(self.config.raw_data.working_dir) / self.config.initial_json_files.meld_video_data
        )
    
    def run(self):
        self.create_base_datasets()
        self.extract_audio()