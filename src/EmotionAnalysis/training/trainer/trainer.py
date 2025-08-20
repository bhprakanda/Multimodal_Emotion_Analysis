import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from EmotionAnalysis.training.Loss.loss import FocalLoss
from EmotionAnalysis.data.meld_dataset_SlowFast import MELDDataset
from EmotionAnalysis.models.architecture_MaskedSlowFast import MaskedSlowFast



class Trainer_SlowFast:
    """Main training class that orchestrates the entire training process"""
    
    def __init__(self, config, device, class_names, hf_api):
        self.config = config
        self.device = device
        self.class_names = class_names
        self.hf_api = hf_api
        self.scaler = torch.amp.GradScaler()
        
    def train_model(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        total_steps: int,
        wandb_run_id: Optional[str] = None,
        start_epoch: int = 0,
        best_val_loss: float = float('inf'),
        best_val_weighted_f1: float = 0.0,
        best_model_state: Optional[Dict[str, torch.Tensor]] = None, 
        patience_counter: int = 0,
        initial_global_step: int = 0
    ) -> Dict[str, Any]:
        
        # Initialize training state
        global_step = initial_global_step
        best_val_weighted_f1 = best_val_weighted_f1
        patience_counter = patience_counter
        best_model_state = best_model_state or model.state_dict().copy()
        
        # Initialize WandB
        wandb_init_kwargs = self._build_wandb_init_args(
            model, optimizer, wandb_run_id, start_epoch > 0
        )
        wandb_run = wandb.init(**wandb_init_kwargs)
        wandb.watch(model, log="all", log_freq=50)
        
        # Initialize metrics tracking
        epoch_metrics = self._initialize_training_metrics()
        
        # Training loop
        for epoch in range(start_epoch, self.config.max_epochs):
            model.train()
            model.unfreeze_layers(epoch)
            
            # Training phase
            train_results = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                epoch, global_step, epoch_metrics
            )
            global_step = train_results['global_step']
            
            # Validation phase
            val_results = self._evaluate_model(
                model, val_loader, criterion, 'val'
            )
            
            # Build log data
            log_data = self._build_wandb_log_data(
                epoch, train_results, val_results, optimizer, 
                epoch_metrics, global_step
            )
            wandb.log(log_data, step=global_step)
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(
                    model, epoch, optimizer, scheduler, best_val_loss, 
                    best_val_weighted_f1, patience_counter, global_step, 
                    total_steps, False
                )
            
            # Update early stopping criteria
            best_val_weighted_f1, patience_counter, is_improvement = self._update_early_stopping(
                val_results['weighted_f1'], best_val_weighted_f1, patience_counter, epoch
            )
            
            # Save best model if improvement
            if is_improvement:
                self._save_checkpoint(
                    model, epoch, optimizer, scheduler, best_val_loss, 
                    best_val_weighted_f1, patience_counter, global_step, 
                    total_steps, True
                )
            
            # Early stopping
            if epoch >= self.config.min_epochs and patience_counter >= self.config.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

        # Final test evaluation
        test_results = self._evaluate_model(
            model, test_loader, criterion, 'test'
        )
        self._log_final_results(test_results, global_step)
        wandb.finish()
        
        return {
            "best_val_weighted_f1": best_val_weighted_f1,
            **test_results
        }
    
    # ========== TRAINING UTILITIES ==========
    
    def _build_wandb_init_args(self, model, optimizer, wandb_run_id, is_resume):
        """Build arguments for WandB initialization"""
        return {
            "project": "meld-emotion-recognition-new",
            "settings": wandb.Settings(init_timeout=360),
            "config": {
                "architecture": "MaskedSlowFast_R50",
                "num_classes": len(self.class_names),
                "class_names": self.class_names,
                "resumed": is_resume,
                "batch_size": self.config.batch_size,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "num_frames": self.config.num_frames,
                "crop_size": self.config.crop_size,
                "base_lr": self.config.base_lr,
                "max_lr": self.config.max_lr,
                "grad_clip": self.config.grad_clip,
                "accumulation_steps": self.config.accumulation_steps,
                "weight_decay": self.config.weight_decay,
                "resize_size": self.config.resize_size,
            },
            "resume": "allow" if wandb_run_id else None,
            "id": wandb_run_id if wandb_run_id else None
        }
    
    def _initialize_training_metrics(self):
        """Initialize metrics tracking dictionaries"""
        return {
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
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scheduler, epoch, global_step, epoch_metrics):
        """Train for a single epoch"""
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_preds = []
        all_labels = []
        epoch_grad_norms = []
        sample_predictions = []
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.max_epochs}')):
            global_step += 1
            slow, fast, slow_mask, fast_mask, labels = self._prepare_batch(batch)
            
            # Log sample predictions periodically
            if batch_idx % self.config.log_samples_freq == 0:
                sample_preds = self._log_sample_predictions(
                    model, slow, fast, slow_mask, fast_mask, labels, 
                    epoch, batch_idx
                )
                sample_predictions.extend(sample_preds)
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(slow, fast, slow_mask, fast_mask)
                loss = criterion(outputs, labels)
            
            # Handle NaN loss
            if not torch.isfinite(loss):
                print(f"⚠️ NaN loss detected at step {global_step}. Skipping batch.")
                optimizer.zero_grad()
                continue
                
            # Backpropagation
            self.scaler.scale(loss / self.config.accumulation_steps).backward()
            
            # Update metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Gradient accumulation and update
            if (batch_idx + 1) % self.config.accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                # Skip update if NaN gradients
                if self._has_nan_gradients(model):
                    print(f"⚠️ NaN gradients detected at step {global_step}. Skipping update.")
                    optimizer.zero_grad()
                    continue
                    
                # Gradient clipping and update
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                # Log gradient norms
                batch_grad_norms = self._log_gradient_norms(model, global_step)
                if batch_grad_norms:
                    epoch_grad_norms.extend(batch_grad_norms)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"learning_rate/step": current_lr}, step=global_step)
                
                optimizer.zero_grad()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Update epoch metrics
        epoch_metrics['train/loss'].append(avg_train_loss)
        epoch_metrics['train/macro_f1'].append(train_f1)
        epoch_metrics['train/weighted_f1'].append(train_weighted_f1)
        epoch_metrics['train/accuracy'].append(train_accuracy)
        epoch_metrics['grad_norm/epoch_mean'].append(np.mean(epoch_grad_norms) if epoch_grad_norms else 0)
        
        return {
            'global_step': global_step,
            'avg_train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'train_weighted_f1': train_weighted_f1,
            'all_labels': all_labels,
            'epoch_grad_norms': epoch_grad_norms,
            'sample_predictions': sample_predictions
        }
    
    # ========== EVALUATION METHODS ==========
    
    def _evaluate_model(self, model, data_loader, criterion, split='val'):
        """Evaluate model on given dataset split"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                slow, fast, slow_mask, fast_mask, labels = self._prepare_batch(batch)
                
                outputs = model(slow, fast, slow_mask, fast_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_file = self._plot_confusion_matrix(cm, self.class_names, split)
        
        # Classification report
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'classification_report_{split}.csv')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'cm': cm,
            'plot_file': plot_file,
            'all_labels': np.array(all_labels),
            'all_probs': np.array(all_probs),
            'report_df': report_df
        }
    
    # ========== HELPER METHODS ==========
    
    def _prepare_batch(self, batch):
        """Prepare batch data for model input"""
        return (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
            batch[4].to(self.device)
        )
    
    def _has_nan_gradients(self, model):
        """Check for NaN gradients"""
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False
    
    def _log_gradient_norms(self, model, global_step):
        """Log gradient norms for monitoring"""
        grad_norms = []
        nan_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    nan_count += 1
                    continue
                    
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                wandb.log({f"grad_norm/{name}": grad_norm}, step=global_step)
        
        wandb.log({"grad_norm/nan_count": nan_count}, step=global_step)
        
        if grad_norms:
            wandb.log({
                "grad_norm/mean": np.mean(grad_norms),
                "grad_norm/max": np.max(grad_norms),
                "grad_norm/min": np.min(grad_norms),
                "grad_norm/hist": wandb.Histogram(grad_norms) if grad_norms else None
            }, step=global_step)
        
        return grad_norms
    
    def _log_sample_predictions(self, model, slow, fast, slow_mask, fast_mask, labels, epoch, batch_idx):
        """Log sample predictions with visualizations"""
        sample_predictions = []
        
        with torch.no_grad():
            outputs_debug = model(slow, fast, slow_mask, fast_mask)
            _, preds_debug = torch.max(outputs_debug, 1)
            
            for i in range(min(3, len(slow))):  # First 3 samples
                # Create visualization of input frames
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                
                # Show first frame of slow pathway
                slow_frame = slow[i, :, 0].permute(1, 2, 0).cpu().numpy()
                slow_frame = np.clip((slow_frame * self.config.dataset_std + self.config.dataset_mean) * 255, 0, 255).astype(np.uint8)
                ax[0].imshow(slow_frame)
                ax[0].set_title(f"Slow Frame 0\nMask: {slow_mask[i, 0].item():.1f}")
                
                # Show last frame of slow pathway
                slow_frame = slow[i, :, -1].permute(1, 2, 0).cpu().numpy()
                slow_frame = np.clip((slow_frame * self.config.dataset_std + self.config.dataset_mean) * 255, 0, 255).astype(np.uint8)
                ax[1].imshow(slow_frame)
                ax[1].set_title(f"Slow Frame -1\nMask: {slow_mask[i, -1].item():.1f}")
                
                # Show first frame of fast pathway
                fast_frame = fast[i, :, 0].permute(1, 2, 0).cpu().numpy()
                fast_frame = np.clip((fast_frame * self.config.dataset_std + self.config.dataset_mean) * 255, 0, 255).astype(np.uint8)
                ax[2].imshow(fast_frame)
                ax[2].set_title(f"Fast Frame 0\nMask: {fast_mask[i, 0].item():.1f}")
                
                # Show last frame of fast pathway
                fast_frame = fast[i, :, -1].permute(1, 2, 0).cpu().numpy()
                fast_frame = np.clip((fast_frame * self.config.dataset_std + self.config.dataset_mean) * 255, 0, 255).astype(np.uint8)
                ax[3].imshow(fast_frame)
                ax[3].set_title(f"Fast Frame -1\nMask: {fast_mask[i, -1].item():.1f}")
                
                plt.tight_layout()
                plt.savefig(f"sample_{epoch}_{batch_idx}_{i}.png")
                plt.close()
                
                sample_predictions.append([
                    epoch,
                    batch_idx,
                    self.class_names[preds_debug[i].item()],
                    self.class_names[labels[i].item()],
                    wandb.Image(f"sample_{epoch}_{batch_idx}_{i}.png")
                ])
        
        return sample_predictions
    
    def _plot_confusion_matrix(self, cm, class_names, split):
        """Plot confusion matrix and save as image"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{split.capitalize()} Confusion Matrix')
        plot_file = f'confusion_matrix_{split}.png'
        plt.savefig(plot_file)
        plt.close()
        return plot_file
    
    def _build_wandb_log_data(self, epoch, train_results, val_results, optimizer, epoch_metrics, global_step):
        """Build comprehensive log data for W&B"""
        log_data = {
            "epoch": epoch,
            "train/loss": train_results['avg_train_loss'],
            "train/accuracy": train_results['train_accuracy'],
            "train/macro_f1": train_results['train_f1'],
            "train/weighted_f1": train_results['train_weighted_f1'],
            "val/loss": val_results['loss'],
            "val/accuracy": val_results['accuracy'],
            "val/macro_f1": val_results['macro_f1'],
            "val/weighted_f1": val_results['weighted_f1'],
            "learning_rate": optimizer.param_groups[0]['lr'],
            "val/confusion_matrix": wandb.Image(val_results['plot_file']),
            "train/class_distribution": wandb.Histogram(np.array(train_results['all_labels'])),
        }

        # Add ROC curve
        try:
            log_data["val/roc_curve"] = wandb.plot.roc_curve(
                val_results['all_labels'],
                val_results['all_probs'],
                labels=self.class_names
            )
        except Exception as e:
            print(f"Failed to log validation ROC: {str(e)}")
        
        # Add classification report metrics
        for metric in ['precision', 'recall', 'f1-score']:
            for i, class_name in enumerate(self.class_names):
                log_data[f"val/{class_name}_{metric}"] = val_results['report_df'].loc[class_name, metric]
        
        # Log sample predictions
        if train_results['sample_predictions']:
            log_data["train/sample_predictions"] = wandb.Table(
                columns=["Epoch", "Batch", "Predicted", "True", "Image"],
                data=train_results['sample_predictions']
            )
        
        # Log metric history curves
        for metric_name, values in epoch_metrics.items():
            if values:
                data = [[x, y] for x, y in enumerate(values)]
                table = wandb.Table(data=data, columns=["epoch", metric_name])
                log_data[f"curves/{metric_name}"] = wandb.plot.line(
                    table, 
                    "epoch", 
                    metric_name,
                    title=f"{metric_name} over Epochs"
                )
        
        return log_data
    
    def _update_early_stopping(self, current_val_wf1, best_val_weighted_f1, patience_counter, epoch):
        """Update early stopping criteria"""
        improvement = current_val_wf1 - best_val_weighted_f1
        new_best = best_val_weighted_f1
        new_patience = patience_counter
        is_improvement = False

        if improvement > self.config.no_improvement_threshold:
            new_best = current_val_wf1
            new_patience = 0
            is_improvement = True
            print(f"🌟 Significant improvement: +{improvement:.4f}")
        elif improvement > 0:  # Small improvement
            new_best = current_val_wf1
            new_patience = min(patience_counter, self.config.patience // 2)
            is_improvement = True
            print(f"💫 Minor improvement: +{improvement:.4f}, patience reduced to {new_patience}")
        else:
            new_patience += 1
            print(f"⏳ No improvement. Patience: {new_patience}/{self.config.patience}")
        
        return new_best, new_patience, is_improvement
    
    def _log_final_results(self, test_results, global_step):
        """Log final test results to W&B"""
        test_log_data = {
            "final_test/loss": test_results['loss'],
            "final_test/accuracy": test_results['accuracy'],
            "final_test/f1": test_results['macro_f1'],
            "final_test/weighted_f1": test_results['weighted_f1'],
            "final_test/confusion_matrix": wandb.Image(test_results['plot_file']),
        }
        
        # Test ROC logging
        try:
            test_log_data["test/roc_curve"] = wandb.plot.roc_curve(
                test_results['all_labels'],
                test_results['all_probs'],
                labels=self.class_names
            )
        except Exception as e:
            print(f"Failed to log test ROC: {str(e)}")
        
        # Log test classification report
        test_report_table = wandb.Table(dataframe=test_results['report_df'])
        test_log_data["final_test/classification_report"] = test_report_table
        
        # Log all test results
        wandb.log(test_log_data, step=global_step + 1)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _save_checkpoint(self, model, epoch, optimizer, scheduler, best_val_loss, best_val_weighted_f1, patience_counter, global_step, total_steps, is_best):
        """Save checkpoint to HuggingFace Hub"""
        save_dir = f"./model_checkpoint_epoch_{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)

        # Capture all critical states
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            "best_val_weighted_f1": best_val_weighted_f1,
            'patience_counter': patience_counter,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'grad_scaler_state_dict': self.scaler.state_dict(),
            'global_step': global_step,
            'scheduler_total_steps': total_steps,
            'scheduler_last_epoch': scheduler.last_epoch,
            'wandb_run_id': wandb.run.id if wandb.run else None
        }

        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)

        repo_name = "prakanda/hatsu-meld-emotion-recognition-new"
        
        # Ensure the repo exists
        self.hf_api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        
        try:
            self.hf_api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"checkpoint_epoch_{epoch+1}.pth",
                repo_id=repo_name,
                repo_type="model",
            )
            if is_best:
                self.hf_api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo="best_model.pth",
                    repo_id=repo_name,
                    repo_type="model",
                )
        except Exception as e:
            print(f"Error uploading: {str(e)}")
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)






if __name__ == "__main__":
    print("Starting Video Emotion Recognition Training")
    print("="*60)
    
    # Set up environment
    set_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup APIs
    hf_token, wandb_token, hf_api = setup_api_tokens()
    
    # Create sample dataset
    print("Creating sample dataset...")
    sample_dataset = create_sample_dataset()
    class_names = ["neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"]
    
    # Create dataloaders
    train_dataset = MELDDataset(sample_dataset, "train", train=True)
    val_dataset = MELDDataset(sample_dataset, "val", train=False)
    test_dataset = MELDDataset(sample_dataset, "test", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize model
    model = MaskedSlowFast(num_classes=len(class_names)).to(device)
    
    # Initialize loss function with class weights
    class_weights = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 2.0], device=device)
    criterion = FocalLoss(alpha=class_weights)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.base_lr, weight_decay=Config.weight_decay)
    total_steps = Config.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.max_lr, 
        total_steps=total_steps,
        epochs=Config.max_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Initialize trainer
    trainer = Trainer_SlowFast(Config, device, class_names, hf_api)
    
    # Start training
    results = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=total_steps
    )
    
    print("\nTraining completed successfully!")
    print(f"Best Validation Weighted F1: {results['best_val_weighted_f1']:.4f}")
    print(f"Test Weighted F1: {results['weighted_f1']:.4f}")