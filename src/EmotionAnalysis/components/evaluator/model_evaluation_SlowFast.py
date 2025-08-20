import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report

class Evaluator_SlowFast:
    def __init__(self, model, data_loader, criterion, device, class_names):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.class_names = class_names

    def evaluate(self, split='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                slow, fast, slow_mask, fast_mask, labels = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                    batch[3].to(self.device),
                    batch[4].to(self.device)
                )
                
                outputs = self.model(slow, fast, slow_mask, fast_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.data_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_file = self.plot_confusion_matrix(cm, split)
        
        # Classification report
        report = classification_report(all_labels, all_preds, 
                                      target_names=self.class_names, 
                                      output_dict=True)
        
        return (
            avg_loss, accuracy, macro_f1, weighted_f1, 
            cm, plot_file, np.array(all_labels), 
            np.array(all_probs), report
        )

    def plot_confusion_matrix(self, cm, split='val'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'{split.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plot_file = f'confusion_matrix_{split}.png'
        plt.savefig(plot_file)
        plt.close()
        return plot_file