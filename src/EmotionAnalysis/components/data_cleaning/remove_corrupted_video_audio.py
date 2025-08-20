import cv2
import wave
import json
import pandas as pd
from pathlib import Path

from EmotionAnalysis.utils.common_utils import read_from_json, save_to_json, save_csv, load_csv
from EmotionAnalysis.entity import CorruptedVideoRemovingConfig



class CorruptedVideoRemover:
    def __init__(self, config: CorruptedVideoRemovingConfig):
        self.config = config
        
    def check_video_integrity(self, video_path: Path) -> bool:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            video.release()
            return False
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return frame_count > 0 and fps > 0

    def check_audio_integrity(self, audio_path: Path) -> bool:
        try:
            with wave.open(str(audio_path), 'rb') as audio:
                params = audio.getparams()
                n_frames = audio.getnframes()
                return n_frames > 0 and params.nchannels > 0 and params.framerate > 0
        except:
            return False

    def find_corrupted_media(self):
        # Check videos
        video_data = read_from_json(
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.json.meld_video_data
        )
        
        corrupted_videos = {"train": [], "dev": [], "test": []}
        for split in video_data:
            for entry in video_data[split]:
                for key, path in entry.items():
                    if key not in ["y", "label"] and not self.check_video_integrity(Path(path)):
                        corrupted_videos[split].append(entry)
        
        save_to_json(
            corrupted_videos,
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.json.meld_corrupted_video_data
        )
        
        # Check audio
        audio_data = read_from_json(
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.json.meld_audio_data_updated
        )
        
        corrupted_audios = {"train": [], "dev": [], "test": []}
        for split in audio_data:
            for entry in audio_data[split]:
                for key, path in entry.items():
                    if key not in ["y", "label"] and not self.check_audio_integrity(Path(path)):
                        corrupted_audios[split].append(entry)
        
        save_to_json(
            corrupted_audios,
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.json.meld_corrupted_audio_data
        )
        
        return corrupted_videos, corrupted_audios

    def remove_corrupted_records(self):
        # Load CSVs
        train_df = load_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.train
        )
        dev_df = load_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.dev
        )
        test_df = load_csv(
            Path(self.config.raw_data.input_dir) / self.config.raw_data.original.test
        )
        
        # Get corrupted data
        corrupted_videos = read_from_json(
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.json.meld_corrupted_video_data
        )
        
        # Filter DataFrames
        def filter_df(df, corrupted_entries, split):
            corrupted_keys = set()
            for entry in corrupted_entries.get(split, []):
                for key in entry:
                    if key not in ["y", "label"] and '_' in key:
                        corrupted_keys.add(tuple(map(int, key.split('_'))))
            
            return df[~df[['Dialogue_ID', 'Utterance_ID']].apply(tuple, axis=1).isin(corrupted_keys)]
        
        train_cleaned = filter_df(train_df, corrupted_videos, "train")
        dev_cleaned = filter_df(dev_df, corrupted_videos, "dev")
        test_cleaned = filter_df(test_df, corrupted_videos, "test")
        
        # Save cleaned data
        save_csv(
            train_cleaned, 
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.cleaned_output.csv.train_sent_emo_cleaned
        )
        save_csv(
            dev_cleaned, 
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.cleaned_output.csv.dev_sent_emo_cleaned
        )
        save_csv(
            test_cleaned, 
            Path(self.config.removed_corrupted_data.working_dir) / 
            self.config.removed_corrupted_data.cleaned_output.csv.test_sent_emo_cleaned
        )
    
    def run(self):
        self.find_corrupted_media()
        self.remove_corrupted_records()