import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

from EmotionAnalysis.utils.visualizer import plot_styled_confusion_matrix
from EmotionAnalysis.entity import PlotVisualizationConfig


class Evaluator_GNN:
    def __init__(self, plot_visualization_config:PlotVisualizationConfig, model, test_loader, device):
        self.plot_visualization_config = plot_visualization_config
        self.model = model
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self, seed):
        self.model.eval()
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                main_out, _ = self.model(data)
                
                mask = data.y != -100
                if mask.sum() == 0:
                    continue
                    
                preds = main_out.argmax(dim=1)[mask]
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(data.y[mask].cpu().tolist())
        
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_acc = accuracy_score(test_labels, test_preds)
        macro_f1 = f1_score(test_labels, test_preds, average='macro')
        
        print("\nOverall Test Performance:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Weighted F1: {test_f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, digits=4))
        
        plot_styled_confusion_matrix(self.plot_visualization_config, test_labels, test_preds, seed)
        
        return test_f1