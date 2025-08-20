import os
import torch
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score

from EmotionAnalysis.utils.common_utils import set_seed
from EmotionAnalysis.components.evaluator.model_evaluation_SlowFast import Evaluator_SlowFast
from EmotionAnalysis.entity import ModelTrainerConfig


class Trainer_SlowFast:
    def __init__(self,model_trainer_config: ModelTrainerConfig, model, train_loader, val_loader, optimizer, scheduler, 
                 criterion, device, class_names, scaler, hf_api=None):
        self.model_trainer_config = model_trainer_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.class_names = class_names
        self.scaler = scaler
        self.hf_api = hf_api
        self.best_val_weighted_f1 = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
        # Initialize metrics tracking
        self.epoch_metrics = {
            'train/loss': [],
            'val/loss': [],
            'train/macro_f1': [],
            'val/macro_f1': [],
            'train/weighted_f1': [],
            'val/weighted_f1': [],
            'train/accuracy': [],
            'val/accuracy': [],
            'learning_rate': [],
            'grad_norm/epoch_mean': [],
        }

    def train_epoch(self, epoch, global_step):
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_preds = []
        all_labels = []
        epoch_grad_norms = []
        sample_predictions = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
            global_step += 1
            slow, fast, slow_mask, fast_mask, labels = (
                batch[0].to(self.device),
                batch[1].to(self.device),
                batch[2].to(self.device),
                batch[3].to(self.device),
                batch[4].to(self.device)
            )
            
            # Log sample predictions periodically
            if batch_idx % self.model_trainer_config.LOG_SAMPLES_FREQ_SLOWFAST == 0:
                sample_preds = self._log_sample_predictions(
                    slow, fast, slow_mask, fast_mask, labels, 
                    epoch, batch_idx
                )
                sample_predictions.extend(sample_preds)
            
            # Mixed precision training
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(slow, fast, slow_mask, fast_mask)
                loss = self.criterion(outputs, labels)
                
            # Skip batch if NaN loss
            if not torch.isfinite(loss):
                print(f"⚠️ NaN loss at step {global_step}. Skipping batch.")
                self.optimizer.zero_grad()
                continue
                
            self.scaler.scale(loss / self.model_trainer_config.ACCUMULATION_STEPS_SLOWFAST).backward()
            
            # Update metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Gradient accumulation
            if (batch_idx + 1) % self.model_trainer_config.ACCUMULATION_STEPS_SLOWFAST == 0 or batch_idx == len(self.train_loader) - 1:
                # Skip update if NaN gradients
                if self._check_nan_gradients():
                    print(f"⚠️ NaN gradients at step {global_step}. Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                    
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.model_trainer_config.GRAD_CLIP_SLOWFAST)
                
                # Log gradient norms
                batch_grad_norms = self._log_gradient_norms(global_step)
                if batch_grad_norms:
                    epoch_grad_norms.extend(batch_grad_norms)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Scheduler step
                self.scheduler.step()
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb.log({"learning_rate/step": current_lr}, step=global_step)
                
                self.optimizer.zero_grad()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(self.train_loader)
        train_accuracy = train_correct / train_total
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Update epoch metrics
        self.epoch_metrics['train/loss'].append(avg_train_loss)
        self.epoch_metrics['train/macro_f1'].append(train_f1)
        self.epoch_metrics['train/weighted_f1'].append(train_weighted_f1)
        self.epoch_metrics['train/accuracy'].append(train_accuracy)
        self.epoch_metrics['learning_rate'].append(current_lr)
        
        if epoch_grad_norms:
            grad_mean = np.mean(epoch_grad_norms)
            self.epoch_metrics['grad_norm/epoch_mean'].append(grad_mean)
        
        return (
            avg_train_loss, train_accuracy, train_f1, train_weighted_f1,
            all_labels, sample_predictions, epoch_grad_norms, global_step
        )

    def validate(self, evaluator, epoch, global_step):
        val_metrics = evaluator.evaluate(split='val')
        val_loss, val_accuracy, val_f1, val_wf1 = val_metrics[:4]
        
        # Update epoch metrics
        self.epoch_metrics['val/loss'].append(val_loss)
        self.epoch_metrics['val/macro_f1'].append(val_f1)
        self.epoch_metrics['val/weighted_f1'].append(val_wf1)
        self.epoch_metrics['val/accuracy'].append(val_accuracy)
        
        # Build log data
        log_data = {
            "epoch": epoch,
            "train/loss": self.epoch_metrics['train/loss'][-1],
            "train/accuracy": self.epoch_metrics['train/accuracy'][-1],
            "train/macro_f1": self.epoch_metrics['train/macro_f1'][-1],
            "train/weighted_f1": self.epoch_metrics['train/weighted_f1'][-1],
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
            "val/macro_f1": val_f1,
            "val/weighted_f1": val_wf1,
            "learning_rate": self.epoch_metrics['learning_rate'][-1],
            "val/confusion_matrix": wandb.Image(val_metrics[5]),
            "train/class_distribution": wandb.Histogram(np.array(self.epoch_metrics['train/loss'])),
        }
        
        if 'grad_norm/epoch_mean' in self.epoch_metrics and self.epoch_metrics['grad_norm/epoch_mean']:
            log_data.update({
                "grad_norm/epoch_mean": self.epoch_metrics['grad_norm/epoch_mean'][-1],
                "grad_norm/epoch_max": np.max(self.epoch_metrics['grad_norm/epoch_mean']),
                "grad_norm/epoch_min": np.min(self.epoch_metrics['grad_norm/epoch_mean']),
            })
        
        # Log to WandB
        wandb.log(log_data, step=global_step)
        
        return val_loss, val_accuracy, val_f1, val_wf1, global_step

    def fit(self, max_epochs, min_epochs, patience, checkpoint_frequency, 
            global_step=0, start_epoch=0, wandb_run_id=None):
        set_seed(self.model_trainer_config.SEED_SLOWFAST)
        
        # Initialize WandB
        wandb_init_kwargs = {
            "project": "meld-emotion-recognition",
            "config": {
                "architecture": "MaskedSlowFast_R50",
                "dataset": "MELD",
                "epochs": max_epochs,
            },
            "resume": "allow" if wandb_run_id else None
        }
        if wandb_run_id:
            wandb_init_kwargs["id"] = wandb_run_id
        wandb.init(**wandb_init_kwargs)
        wandb.watch(self.model, log="all", log_freq=50)
        
        # Initialize evaluator
        evaluator = Evaluator_SlowFast(
            self.model, self.val_loader, self.criterion, 
            self.device, self.class_names
        )
        
        for epoch in range(start_epoch, max_epochs):
            self.model.unfreeze_layers(epoch)
            
            # Training phase
            train_metrics = self.train_epoch(epoch, global_step)
            global_step = train_metrics[-1]  # Update global step
            
            # Validation phase
            val_metrics = self.validate(evaluator, epoch, global_step)
            val_wf1 = val_metrics[3]
            global_step = val_metrics[-1]
            
            # Print metrics
            print(f"\nEpoch [{epoch + 1}/{max_epochs}]")
            print(f"Train Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}")
            print(f"Val Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.4f}, "
                  f"WF1: {val_wf1:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(epoch, global_step, is_best=False)
            
            # Early stopping
            improvement = val_wf1 - self.best_val_weighted_f1
            if improvement > self.model_trainer_config.NO_IMPROVEMENT_THRESHOLD_SLOWFAST:
                self.best_val_weighted_f1 = val_wf1
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, global_step, is_best=True)
                print(f"🌟 Improvement: +{improvement:.4f}")
            elif improvement > 0:
                self.best_val_weighted_f1 = val_wf1
                self.patience_counter = min(self.patience_counter, patience // 2)
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, global_step, is_best=True)
                print(f"💫 Minor improvement: +{improvement:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ No improvement. Patience: {self.patience_counter}/{patience}")
            
            # Early stopping check
            if epoch >= min_epochs and self.patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        wandb.finish()
        return global_step

    def _save_checkpoint(self, epoch, global_step, is_best=False):
        if not self.hf_api:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_weighted_f1': self.best_val_weighted_f1,
            'patience_counter': self.patience_counter,
            'global_step': global_step,
            'wandb_run_id': wandb.run.id,
        }
        
        # Simplified checkpoint saving logic
        # Actual Hugging Face upload would go here
        print(f"💾 Saving checkpoint for epoch {epoch+1} (best: {is_best})")

    def _check_nan_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False

    def _log_gradient_norms(self, global_step):
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isfinite(param.grad).all():
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                wandb.log({f"grad_norm/{name}": grad_norm}, step=global_step)
        return grad_norms

    def _log_sample_predictions(self, slow, fast, slow_mask, fast_mask, labels, epoch, batch_idx):
        sample_predictions = []
        with torch.no_grad():
            outputs = self.model(slow, fast, slow_mask, fast_mask)
            _, preds = torch.max(outputs, 1)
            
            for i in range(min(3, len(slow))):
                # Simplified visualization logic
                # Actual implementation would create frame visualizations
                sample_predictions.append([
                    epoch,
                    batch_idx,
                    self.class_names[preds[i].item()],
                    self.class_names[labels[i].item()],
                    None  # Placeholder for image
                ])
        return sample_predictions