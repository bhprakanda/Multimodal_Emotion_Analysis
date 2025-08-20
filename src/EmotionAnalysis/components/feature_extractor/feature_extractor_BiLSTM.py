import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from EmotionAnalysis.data.dialogue_dataset_BiLSTM import DialogueDataset


class FeatureExtractor_BiLSTM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def extract_features(self, dataset, original_data, split, normalize=True):
        """Extract features and store them back in the original data structure"""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                          collate_fn=DialogueDataset.collate_fn)
        
        with torch.no_grad():
            for batch_idx, (dialogues, labels, lengths) in enumerate(tqdm(loader)):
                dialogues, labels = dialogues.to(self.device), labels.to(self.device)
                
                # Forward pass to populate features
                _, _ = self.model(dialogues, lengths)
                
                # Get contextual features
                features = self.model.get_contextual_features(normalize=normalize)
                
                # Process each utterance in the dialogue
                for utt_idx in range(lengths[0]):
                    # Get the original utterance ID
                    dialogue_id = dataset.dialogue_list[batch_idx][utt_idx]['dialogue_id']
                    utt_id = dataset.dialogue_list[batch_idx][utt_idx]['utt_id']
                    utterance_key = f"{dialogue_id}_{utt_id}"
                    
                    # Find the corresponding item in the original data
                    for item in original_data[split]:
                        if f"{item['dialogue_id']}_{item['utt_id']}" == utterance_key:
                            # Create new key name by replacing 'Concatenated' with 'BiLSTM'
                            new_key = utterance_key + "_BiLSTM"
                            
                            # Convert to numpy and store
                            item[new_key] = features[0, utt_idx].cpu().numpy().tolist()
                            break
        
        print(f"Extracted features for {len(original_data[split])} utterances in {split} split")
        return original_data