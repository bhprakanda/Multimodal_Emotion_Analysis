import torch
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

from EmotionAnalysis.entity import ModelTrainerConfig


class Trainer_GNN:
    def __init__(self, model_trainer_config: ModelTrainerConfig, model, criterion, train_loader, dev_loader, device, seed):
        self.model_trainer_config = model_trainer_config
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.seed = seed
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.model_trainer_config.LR_GNN,
            weight_decay=self.model_trainer_config.WEIGHT_DECAY_GNN
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.75, patience=model_trainer_config.PATIENCE_GNN, verbose=True
        )
        self.best_val_f1 = 0
        self.early_stop_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        train_preds, train_labels = [], []
        total_loss = 0
        self.optimizer.zero_grad()
        
        # Learning rate warmup
        if epoch < self.model_trainer_config.WARMUP_EPOCHS_GNN:
            lr_scale = min(1.0, float(epoch + 1) / self.model_trainer_config.WARMUP_EPOCHS_GNN)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_scale * self.model_trainer_config.LR_GNN
        
        for i, data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(self.device)
            main_out, aux_out = self.model(data)
            
            mask = data.y != -100
            if mask.sum() == 0:
                continue
                
            loss_main = self.criterion(main_out[mask], data.y[mask])
            loss_aux = self.criterion(aux_out[mask], data.y[mask])
            loss = 0.9 * loss_main + 0.1 * loss_aux
            loss = loss / self.model_trainer_config.ACCUMULATION_STEPS_GNN
            
            loss.backward()
            
            if (i + 1) % self.model_trainer_config.ACCUMULATION_STEPS_GNN == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.model_trainer_config.ACCUMULATION_STEPS_GNN
            preds = main_out.argmax(dim=1)[mask]
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(data.y[mask].cpu().tolist())
        
        # Handle remaining gradients
        if len(self.train_loader) % self.model_trainer_config.ACCUMULATION_STEPS_GNN != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        train_f1 = f1_score(train_labels, train_preds, average='weighted') if train_labels else 0
        train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0
        return total_loss/len(self.train_loader), train_acc, train_f1

    def validate(self):
        self.model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for data in self.dev_loader:
                data = data.to(self.device)
                main_out, aux_out = self.model(data)
                
                mask = data.y != -100
                if mask.sum() == 0:
                    continue
                    
                loss_main = self.criterion(main_out[mask], data.y[mask])
                loss_aux = self.criterion(aux_out[mask], data.y[mask])
                loss = 0.9 * loss_main + 0.1 * loss_aux
                val_loss += loss.item()
                
                preds = main_out.argmax(dim=1)[mask]
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(data.y[mask].cpu().tolist())
                
        if not val_labels:
            return 0, 0, 0
            
        val_loss_avg = val_loss / len(self.dev_loader)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)
        return val_loss_avg, val_acc, val_f1

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step(val_f1)
            
            # Early stopping check
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.model_trainer_config.MODEL_SAVE_PATH_GNN.format(self.seed))
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.model_trainer_config.PATIENCE_GNN:
                    break
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")