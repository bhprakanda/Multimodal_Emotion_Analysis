import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from EmotionAnalysis.utils.common_utils import set_seed
from EmotionAnalysis.entity import ModelTrainerConfig


class DataHandler:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        self.data_path = model_trainer_config.DATA_PATH_GNN
        self.data = None
        self.speaker_to_idx = {}
        self.class_counts = None
        self.class_weights = None
        
    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        self._compute_class_weights()
        self._create_speaker_mapping()
        
    def _compute_class_weights(self):
        all_labels = []
        for split in ['train', 'dev', 'test']:
            for item in self.data[split]:
                if any(k.endswith("_BiLSTM") for k in item):
                    all_labels.append(item['y'])
        self.class_counts = np.bincount(all_labels, minlength=7)
        self.class_weights = torch.tensor(
            compute_class_weight('balanced', classes=np.arange(7), y=all_labels), 
            dtype=torch.float32
        )
    
    def _create_speaker_mapping(self):
        all_speakers = set()
        for split in ['train', 'dev', 'test']:
            for item in self.data[split]:
                all_speakers.add(item['speaker'])
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(sorted(all_speakers))}
    

    def build_dialogue_graph(dialogue_items, speaker_to_idx, dialogue_id):
        """Enhanced graph construction with semantic relationships"""
        utterances = sorted(
            [item for item in dialogue_items if any(k.endswith("_BiLSTM") for k in item)],
            key=lambda x: int(x['utt_id'])
        )
        
        # Handle cases with 0 utterances
        if len(utterances) == 0:
            return None
        
        node_features = []
        labels = []
        speaker_ids = []
        emotions = []
        
        for item in utterances:
            feature_key = next(k for k in item if k.endswith("_BiLSTM"))
            node_features.append(item[feature_key])
            labels.append(item['y'])
            speaker_ids.append(speaker_to_idx[item['speaker']])
            emotions.append(item['y'])
        
        num_nodes = len(node_features)
        edge_index = []
        edge_attr = []
        
        # Always add self-loops
        for i in range(num_nodes):
            edge_index.append([i, i])
            edge_attr.append([1, 0, 1])  # same_speaker=1, distance=0, emotion_same=1
        
        # Only add connections if >1 node
        if num_nodes > 1:
            # 1. Chronological connections with longer context
            for i in range(num_nodes):
                # Connect to next 3 utterances
                for offset in range(1, 4):
                    j = i + offset
                    if j < num_nodes:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        same_speaker = int(speaker_ids[i] == speaker_ids[j])
                        emotion_same = int(emotions[i] == emotions[j])
                        distance = 1.0 / offset
                        edge_attr.append([same_speaker, distance, emotion_same])
                        edge_attr.append([same_speaker, distance, emotion_same])
            
            # 2. Speaker-based connections (long-range)
            speaker_history = defaultdict(list)
            for i, spk in enumerate(speaker_ids):
                if spk in speaker_history:
                    prev_indices = sorted(speaker_history[spk])
                    for j in prev_indices:
                        if abs(i - j) > 1:
                            edge_index.append([i, j])
                            edge_index.append([j, i])
                            distance = 1.0 / (abs(i - j) + 1e-5)
                            emotion_same = int(emotions[i] == emotions[j])
                            edge_attr.append([1, distance, emotion_same])
                            edge_attr.append([1, distance, emotion_same])
                speaker_history[spk].append(i)
            
            # 3. Emotion-based connections with threshold
            emotion_groups = defaultdict(list)
            for i, emo in enumerate(emotions):
                emotion_groups[emo].append(i)
            
            for emo in sorted(emotion_groups.keys()):
                indices = sorted(emotion_groups[emo])
                if len(indices) > 1:
                    for idx in range(1, len(indices)):
                        i1, i2 = indices[idx-1], indices[idx]
                        if abs(i1 - i2) < 8:  # Increased connection window
                            edge_index.append([i1, i2])
                            edge_index.append([i2, i1])
                            distance = 1.0 / (abs(i1 - i2) + 1e-5)
                            edge_attr.append([0, distance, 1])
                            edge_attr.append([0, distance, 1])
            
            # 4. Cross-speaker reactions with lookback
            for i in range(1, num_nodes):
                # Connect to previous 2 speakers
                for lookback in range(1, 3):
                    if i - lookback >= 0 and speaker_ids[i] != speaker_ids[i-lookback]:
                        edge_index.append([i, i-lookback])
                        edge_index.append([i-lookback, i])
                        emotion_same = int(emotions[i] == emotions[i-lookback])
                        edge_attr.append([0, 1.0/lookback, emotion_same])
                        edge_attr.append([0, 1.0/lookback, emotion_same])
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long),
            speaker_id=torch.tensor(speaker_ids, dtype=torch.long),
            dialogue_id=dialogue_id
        )
    
    def perturb_features(self, features, scale=0.08):
        """Add Gaussian noise to features"""
        noise = torch.randn_like(features) * scale
        return features + noise

    def prepare_datasets(self, seed):
        set_seed(seed)
        
        # Build datasets with minority oversampling
        datasets = {}
        for split in ['train', 'dev', 'test']:
            dialogues = defaultdict(list)
            for item in self.data[split]:
                if any(k.endswith("_BiLSTM") for k in item):
                    dialogues[item['dialogue_id']].append(item)
            
            graphs = []
            for dial_id in sorted(dialogues.keys()):
                items = dialogues[dial_id]
                graph = self.build_dialogue_graph(items, self.speaker_to_idx, dial_id)
                if graph is not None:
                    graphs.append(graph)
                    
                    # Oversample minority classes in training set
                    if split == 'train':
                        minority_count = sum(1 for item in items if item['y'] in self.model_trainer_config.MINORITY_CLASSES_GNN)
                        oversample_factor = min(6, 1 + minority_count * 2)
                        for _ in range(oversample_factor):
                            dup_graph = self.build_dialogue_graph(items, self.speaker_to_idx, dial_id)
                            if dup_graph is None:
                                continue
                            minority_mask = torch.isin(dup_graph.y, torch.tensor(self.model_trainer_config.MINORITY_CLASSES_GNN))
                            if minority_mask.any():
                                dup_graph.x[minority_mask] = self.perturb_features(dup_graph.x[minority_mask], scale=0.1)
                            graphs.append(dup_graph)
            
            datasets[split] = graphs
        
        return datasets

    def create_loaders(self, datasets, seed):
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        train_loader = DataLoader(
            datasets['train'], 
            batch_size=self.model_trainer_config.BATCH_SIZE_GNN, 
            shuffle=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed)
        )
        dev_loader = DataLoader(
            datasets['dev'], 
            batch_size=self.model_trainer_config.BATCH_SIZE_GNN, 
            shuffle=False,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed)
        )
        test_loader = DataLoader(
            datasets['test'], 
            batch_size=self.model_trainer_config.BATCH_SIZE_GNN, 
            shuffle=False,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed)
        )
        
        return train_loader, dev_loader, test_loader