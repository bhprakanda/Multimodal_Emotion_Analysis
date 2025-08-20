import torch
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from core.feature_extractor import FeatureExtractor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from EmotionAnalysis.evaluation.metrics import generate_classification_report


class Trainer_BiLSTM:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.feature_extractor = FeatureExtractor(model, device)
        
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        for dialogues, labels, lengths in tqdm(train_loader, desc="Training"):
            dialogues, labels = dialogues.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs, _ = self.model(dialogues, lengths)
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)
            valid_mask = labels_flat != -100
            
            if valid_mask.any():
                loss = criterion(outputs_flat[valid_mask], labels_flat[valid_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                
                with torch.no_grad():
                    preds = outputs_flat.argmax(dim=-1)
                    all_preds.extend(preds[valid_mask].cpu().tolist())
                    all_labels.extend(labels_flat[valid_mask].cpu().tolist())
        
        return total_loss / len(train_loader), all_preds, all_labels
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for dialogues, labels, lengths in tqdm(val_loader, desc="Validating"):
                dialogues, labels = dialogues.to(self.device), labels.to(self.device)
                outputs, _ = self.model(dialogues, lengths)
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                labels_flat = labels.view(-1)
                valid_mask = labels_flat != -100
                
                if valid_mask.any():
                    loss = criterion(outputs_flat[valid_mask], labels_flat[valid_mask])
                    total_loss += loss.item()
                    preds = outputs_flat.argmax(dim=-1)
                    all_preds.extend(preds[valid_mask].cpu().tolist())
                    all_labels.extend(labels_flat[valid_mask].cpu().tolist())
        
        return total_loss / len(val_loader), all_preds, all_labels
    
    def train(self, train_loader, val_loader, criterion):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        best_val_f1 = 0.0
        epochs_no_improve = 0
        
        for epoch in range(self.config.max_epoch):
            if epochs_no_improve >= self.config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Training phase
            train_loss, train_preds, train_labels = self.train_epoch(train_loader, criterion, optimizer)
            train_report, train_f1, train_acc = generate_classification_report(train_labels, train_preds)
            
            # Validation phase
            val_loss, val_preds, val_labels = self.validate(val_loader, criterion)
            val_report, val_f1, val_acc = generate_classification_report(val_labels, val_preds)
            
            # Update scheduler
            scheduler.step(val_f1)
            
            # Log metrics
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"New best model at epoch {epoch+1} with val F1: {val_f1:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{self.config.patience} epochs")
        
        return self.model
    
    def extract_all_features(self, datasets, original_data):
        """Extract features for all splits"""
        for split, dataset in datasets.items():
            original_data = self.feature_extractor.extract_features(
                dataset, original_data, split
            )
        return original_data