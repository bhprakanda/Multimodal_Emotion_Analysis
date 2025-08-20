import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DialogueDataset(Dataset):
    def __init__(self, data, minority_classes=None, oversample_factor=2):
        self.dialogues = defaultdict(list)
        self.utterance_map = {}
        
        for item in data:
            feature_key = next((k for k in item if k.endswith("_Concatenated")), None)
            if not feature_key:
                continue
                
            dialogue_id, utt_id = feature_key.split("_")[:2]
            utterance_id = f"{dialogue_id}_{utt_id}"
            
            self.utterance_map[utterance_id] = item
            self.dialogues[dialogue_id].append((int(utt_id), item))
        
        self.dialogue_list = []
        for d_id, utterances in self.dialogues.items():
            utterances_sorted = sorted(utterances, key=lambda x: x[0])
            self.dialogue_list.append([item for _, item in utterances_sorted])
        
        if minority_classes:
            self._apply_utterance_oversampling(minority_classes, oversample_factor)
    
    def _apply_utterance_oversampling(self, minority_classes, factor):
        new_dialogues = []
        for dialogue in self.dialogue_list:
            new_dialogues.append(dialogue)
            minority_utterances = [
                (idx, item) for idx, item in enumerate(dialogue)
                if item['y'] in minority_classes
            ]
            for _ in range(factor):
                augmented = dialogue.copy()
                for pos, item in minority_utterances:
                    augmented.insert(pos + 1, item.copy())
                new_dialogues.append(augmented)
        self.dialogue_list = new_dialogues
    
    def __len__(self):
        return len(self.dialogue_list)
    
    def __getitem__(self, idx):
        dialogue = self.dialogue_list[idx]
        embeddings = []
        labels = []
        for item in dialogue:
            feature_key = next(k for k in item if k.endswith("_Concatenated"))
            emb = torch.tensor(item[feature_key], dtype=torch.float32)
            embeddings.append(emb)
            labels.append(torch.tensor(item["y"], dtype=torch.long))
        return torch.stack(embeddings), torch.stack(labels)


def dialogue_collate_fn(batch):
    dialogues, labels = zip(*batch)
    lengths = [d.shape[0] for d in dialogues]
    padded_dialogues = pad_sequence(dialogues, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_dialogues, padded_labels, lengths