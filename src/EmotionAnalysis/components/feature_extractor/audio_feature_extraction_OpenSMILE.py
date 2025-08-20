import opensmile
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import tempfile
import json
from dataclasses import dataclass


from EmotionAnalysis.utils.common_utils import read_from_json, save_to_json
from EmotionAnalysis.entity import AudioFeatureExtractionConfig

class AudioFeatureExtractor:
    def __init__(self, config: AudioFeatureExtractionConfig):
        self.config = config
        self.smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, config.feature_set),
            feature_level=getattr(opensmile.FeatureLevel, config.feature_level)
        )
    
    def preprocess_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
            reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
            max_amp = np.max(np.abs(reduced_noise))
            normalized_audio = reduced_noise * (10 ** (-3 / 20) / max_amp) if max_amp > 0 else reduced_noise
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, normalized_audio, sr, subtype='PCM_16')
            return temp_file.name
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def extract_features(self, audio_path):
        try:
            processed_path = self.preprocess_audio(audio_path)
            if not processed_path:
                return None
                
            features_df = self.smile.process_file(processed_path)
            features_list = features_df.values.flatten().tolist()
            return features_list
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def process_data(self, audio_data):
        features = {split: [] for split in audio_data.keys()}
        for split, items in audio_data.items():
            for item in items:
                for key, audio_path in item.items():
                    if key not in ["y", "label"]:
                        features_list = self.extract_features(audio_path)
                        if features_list:
                            temp_dict = {
                                key: audio_path,
                                f"{key}_opensmile_features": features_list,
                                "y": item.get("y"),
                                "label": item.get("label")
                            }
                            features[split].append(temp_dict)
        return features
    
    def run(self):
        audio_data = read_from_json(self.config.input_json)
        features = self.process_data(audio_data)
        save_to_json(features, self.config.output_json)
        print(f"Audio features saved to: {self.config.output_json}")