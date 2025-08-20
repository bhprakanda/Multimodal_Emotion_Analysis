import os
import re
import cv2
import torch
import random

import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from EmotionAnalysis.config.configuration import ModelTrainerConfig


class MELDDataset(Dataset):
    """Enhanced Dataset with Temporal Augmentations"""
    
    def __init__(self, model_trainer_config: ModelTrainerConfig, metadata, split, train=True):
        self.model_trainer_config = model_trainer_config
        self.data = metadata[split]
        self.train = train
        self.transform = self._build_transforms()
        self.error_log = open("dataset_errors.log", "a")

    def _build_transforms(self):
        """Build data augmentation transforms"""
        normalize = A.Normalize(
            mean=self.model_trainer_config.dataset_mean_SlowFast,
            std=self.model_trainer_config.dataset_std_SlowFast,
            max_pixel_value=255.0
        )
        
        if self.train:
            return A.Compose([
                A.Resize(self.model_trainer_config.resize_size_SlowFast[0], self.model_trainer_config.resize_size_SlowFast[1]),
                A.RandomCrop(height=self.model_trainer_config.crop_size_SlowFast[0], width=self.model_trainer_config.crop_size_SlowFast[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                normalize,
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.model_trainer_config.resize_size_SlowFast[0], self.model_trainer_config.resize_size_SlowFast[1]),
                A.CenterCrop(height=self.model_trainer_config.crop_size_SlowFast[0], width=self.model_trainer_config.crop_size_SlowFast[1], p=1.0),
                normalize,
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        item = self.data[idx]
        frames_dir = item['frames_dir']
        mask_info = item['mask_info']
        label = item['y']
        
        try:
            if not os.path.exists(frames_dir):
                raise FileNotFoundError(f"Directory not found: {frames_dir}")
            
            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
                key=lambda x: int(re.search(r'^(\d+)', x).group(1))
            )
            
            if len(frame_files) < self.model_trainer_config.num_frames_SlowFast:
                raise ValueError(f"Only {len(frame_files)} frames found, need {self.model_trainer_config.num_frames_SlowFast}")

            # FIXED MASK HANDLING
            if len(frame_files) > self.model_trainer_config.num_frames_SlowFast:
                start_idx = random.randint(0, len(frame_files) - self.model_trainer_config.num_frames_SlowFast)
                selected_files = frame_files[start_idx:start_idx+self.model_trainer_config.num_frames_SlowFast]
                selected_mask_info = mask_info[start_idx:start_idx+self.model_trainer_config.num_frames_SlowFast]
            else:
                selected_files = frame_files
                selected_mask_info = mask_info[:len(selected_files)]
                if len(selected_mask_info) < self.model_trainer_config.num_frames_SlowFast:
                    pad_length = self.model_trainer_config.num_frames_SlowFast - len(selected_mask_info)
                    selected_mask_info = selected_mask_info + [0] * pad_length

            frames = []
            for i, fname in enumerate(selected_files):
                frame_path = os.path.join(frames_dir, fname)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise IOError(f"Failed to read {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                transformed = self.transform(image=frame)["image"]
                frames.append(transformed)

            video_tensor = torch.stack(frames)  # [T, C, H, W]
            slow_pathway = video_tensor[::4].permute(1, 0, 2, 3)  # [C, T/4, H, W]
            fast_pathway = video_tensor.permute(1, 0, 2, 3)       # [C, T, H, W]
            
            mask = torch.tensor(selected_mask_info[:self.model_trainer_config.num_frames_SlowFast], dtype=torch.float32)
            slow_mask = mask[::4]
            fast_mask = mask
            
            return slow_pathway, fast_pathway, slow_mask, fast_mask, label
            
        except Exception as e:
            self.error_log.write(f"Error loading index {idx}: {str(e)}\n")
            slow = torch.zeros(3, self.model_trainer_config.num_frames_SlowFast // 4, *self.model_trainer_config.crop_size_SlowFast)
            fast = torch.zeros(3, self.model_trainer_config.num_frames_SlowFast, *self.model_trainer_config.crop_size_SlowFast)
            slow_mask = torch.ones(self.model_trainer_config.num_frames_SlowFast // 4)
            fast_mask = torch.ones(self.model_trainer_config.num_frames_SlowFast)
            return slow, fast, slow_mask, fast_mask, label

    def __len__(self):
        return len(self.data)

    def __del__(self):
        self.error_log.close()


class FeatureExtractionDataset(MELDDataset):
    def __getitem__(self, idx):
        # Get original data
        slow, fast, slow_mask, fast_mask, label = super().__getitem__(idx)
        
        # Get video ID
        item = self.data[idx]
        standard_keys = {'y', 'label', 'frames_dir', 'mask_info'}
        video_id = next((k for k in item.keys() if k not in standard_keys), None)
        
        return slow, fast, slow_mask, fast_mask, label, video_id