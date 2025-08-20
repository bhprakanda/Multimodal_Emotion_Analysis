import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclas

from EmotionAnalysis.utils.common_utils import read_from_json, save_to_json
from EmotionAnalysis.entity import TextFeatureExtractionConfig

class TextFeatureExtractor:
    def __init__(self, config: TextFeatureExtractionConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model = AutoModel.from_pretrained(config.tokenizer_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def tokenize_datasets(self, dataset_dict):
        max_sequence_length = 0
        for data_list in dataset_dict.values():
            for data_entry in data_list:
                text_key = next(iter(data_entry))
                text = data_entry[text_key]
                tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)
                max_sequence_length = max(max_sequence_length, len(tokens["input_ids"]))
        
        tokenized_datasets = {}
        for split, data_list in dataset_dict.items():
            tokenized_list = []
            for i in range(0, len(data_list), self.config.batch_size):
                batch = [data_entry[next(iter(data_entry))] for data_entry in data_list[i:i+self.config.batch_size]]
                tokenized = self.tokenizer(
                    batch,
                    padding=self.config.padding_strategy,
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt"
                )
                tokenized_list.append(tokenized)
            
            tokenized_datasets[split] = {
                'input_ids': torch.cat([t['input_ids'] for t in tokenized_list], dim=0),
                'attention_mask': torch.cat([t['attention_mask'] for t in tokenized_list], dim=0)
            }
        return tokenized_datasets
    
    def extract_features(self, tokenized_datasets, dataset_dict):
        features_datasets = {"train": [], "dev": [], "test": []}
        with torch.no_grad():
            for split, tokenized in tokenized_datasets.items():
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_features = outputs.last_hidden_state[:, 0, :]
                
                for i, data_entry in enumerate(dataset_dict[split]):
                    text_key = next(iter(data_entry))
                    temp_dict = {
                        text_key: data_entry[text_key],
                        f"{text_key}_RoBERTa": cls_features[i].cpu().tolist(),
                        "y": data_entry["y"],
                        "label": data_entry["label"]
                    }
                    features_datasets[split].append(temp_dict)
        return features_datasets
    
    def run(self):
        dataset_dict = read_from_json(self.config.input_json)
        tokenized = self.tokenize_datasets(dataset_dict)
        features = self.extract_features(tokenized, dataset_dict)
        save_to_json(features, self.config.output_json)
        print(f"Text features saved to: {self.config.output_json}")