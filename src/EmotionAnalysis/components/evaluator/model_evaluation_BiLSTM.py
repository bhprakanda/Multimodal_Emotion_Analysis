import torch
import wandb
from tqdm import tqdm

from EmotionAnalysis.evaluation.metrics import generate_classification_report
from EmotionAnalysis.utils.visualizer import plot_styled_confusion_matrix

class Evaluator_BiLSTM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for dialogues, labels, lengths in tqdm(test_loader, desc="Evaluating"):
                dialogues, labels = dialogues.to(self.device), labels.to(self.device)
                outputs, _ = self.model(dialogues, lengths)
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                labels_flat = labels.view(-1)
                valid_mask = labels_flat != -100
                
                if valid_mask.any():
                    preds = outputs_flat.argmax(dim=-1)
                    all_preds.extend(preds[valid_mask].cpu().tolist())
                    all_labels.extend(labels_flat[valid_mask].cpu().tolist())
        
        # Generate reports
        test_report, test_f1, test_acc = generate_classification_report(all_labels, all_preds)
        
        print("\nTest Performance:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Weighted F1: {test_f1:.4f}")
        print("Classification Report:")
        print(test_report)
        
        # Generate confusion matrix
        cm_path = plot_styled_confusion_matrix(
            all_labels, 
            all_preds, 
            "Multimodal Emotion Classification Confusion Matrix"
        )
        
        # Log to WandB
        report_lines = test_report.split('\n')
        table_data = []
        for line in report_lines[2:-2]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    table_data.append(parts[:5])
        
        wandb.log({
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_report": wandb.Table(
                columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                data=table_data
            ),
            "confusion_matrix": wandb.Image(cm_path)
        })
        
        return test_f1, test_acc