import json
from collections import defaultdict
from pathlib import Path

from EmotionAnalysis.entity import FeatureCombiningConfig


class FeatureCombiner:
    def __init__(self, config: FeatureCombiningConfig):
        self.config = config
        
    def combine_features(self):
        # Load feature files
        with open(self.config.text_features_path) as f:
            text_data = json.load(f)
        with open(self.config.audio_features_path) as f:
            audio_data = json.load(f)
        with open(self.config.video_features_path) as f:
            video_data = json.load(f)
        
        # Combine features
        combined_data = {"train": [], "dev": [], "test": []}
        for split in ["train", "dev", "test"]:
            audio_lookup = self._create_audio_lookup(audio_data[split])
            video_lookup = self._create_video_lookup(video_data[split])
            
            for item in text_data[split]:
                text_key = next(k for k in item.keys() 
                               if k not in ['y', 'label'] and not k.endswith('_RoBERTa'))
                
                if text_key in audio_lookup and text_key in video_lookup:
                    combined_features = (
                        item[f"{text_key}_RoBERTa"] + 
                        audio_lookup[text_key]["features"] + 
                        video_lookup[text_key]["features"]
                    )
                    
                    new_entry = {
                        text_key: item[text_key],
                        f"{text_key}_audio_path": audio_lookup[text_key]["path"],
                        f"{text_key}_video_path": video_lookup[text_key]["video_path"],
                        "frames_dir": video_lookup[text_key]["frames_dir"],
                        "mask_info": video_lookup[text_key]["mask_info"],
                        f"{text_key}_Concatenated": combined_features,
                        "y": item["y"],
                        "label": item["label"]
                    }
                    combined_data[split].append(new_entry)
        
        # Save combined data
        Path(self.config.output_combined_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_combined_path, "w") as f:
            json.dump(combined_data, f, indent=4)
        return combined_data

    def add_speaker_info(self, combined_data):
        with open(self.config.meld_cleaned_path) as f:
            meld_data = json.load(f)
        
        # Create speaker mapping
        speaker_mapping = {}
        for item in meld_data['data']:
            key = (item['split'], f"{item['dialog']}_{item['utterance']}")
            speaker_mapping[key] = item['speaker']
        
        # Add speaker information
        for split in ["train", "dev", "test"]:
            for entry in combined_data[split]:
                base_key = next(k for k in entry.keys() 
                               if k not in ['y', 'label', 'frames_dir', 'mask_info'])
                map_key = (split, base_key)
                entry['speaker'] = speaker_mapping.get(map_key, "Unknown")
        
        # Save enhanced data
        with open(self.config.output_enhanced_path, "w") as f:
            json.dump(combined_data, f, indent=4)

    def _create_audio_lookup(self, audio_items):
        lookup = {}
        for item in audio_items:
            audio_key = next(k for k in item.keys() 
                            if k not in ['y', 'label'] and not k.endswith('_opensmile_features'))
            lookup[audio_key] = {
                "path": item[audio_key],
                "features": item[f"{audio_key}_opensmile_features"]
            }
        return lookup

    def _create_video_lookup(self, video_items):
        lookup = {}
        for item in video_items:
            video_key = next(k for k in item.keys() 
                            if k not in ['y', 'label', 'frames_dir', 'mask_info'])
            lookup[video_key] = {
                "video_path": item[video_key],
                "features": item[f"{video_key}__slowfast_features"],
                "frames_dir": item["frames_dir"],
                "mask_info": item["mask_info"]
            }
        return lookup

    def run(self):
        combined_data = self.combine_features()
        self.add_speaker_info(combined_data)
        return combined_data