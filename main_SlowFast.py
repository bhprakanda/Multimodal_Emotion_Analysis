import os
import torch
import wandb
from torch.utils.data import DataLoader

from EmotionAnalysis.utils.common_utils import get_api_tokens, read_from_json, worker_init_fn
from EmotionAnalysis.components.trainer.model_trainer_SlowFast import Trainer_SlowFast
from EmotionAnalysis.components.evaluator.model_evaluation_SlowFast import Evaluator_SlowFast
from EmotionAnalysis.data.meld_dataset_SlowFast import MELDDataset
from EmotionAnalysis.models.architecture_MaskedSlowFast import MaskedSlowFast
from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.training.loss.FocalLoss import FocalLoss
from EmotionAnalysis.config.configuration import ConfigurationManager


def main():
    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_configuration()
    
    # Setup environment
    tokens = get_api_tokens()
    wandb.login(key=tokens["WANDB_API_KEY"])
    
    # Initialize with proper AMP settings
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    extraction_checkpoint = read_from_json(model_trainer_config.INPUT_DATA_PATH_SLOWFAST)
    metadata = extraction_checkpoint["metadata"]

    # Initialize datasets
    train_dataset = MELDDataset(model_trainer_config, metadata, "train")
    val_dataset = MELDDataset(model_trainer_config, metadata, "dev", train=False)
    test_dataset = MELDDataset(model_trainer_config, metadata, 'test', train=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedSlowFast(num_classes=7).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss()

    # Optimizer with weight decay and gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_trainer_config.BASE_LR_SLOWFAST,  # Reduced learning rate
        weight_decay=model_trainer_config.WEIGHT_DECAY_SLOWFAST  # Increased weight decay
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=model_trainer_config.PATIENCE_SLOWFAST, verbose=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_trainer_config.BATCH_SIZE_SLOWFAST,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=model_trainer_config.BATCH_SIZE_SLOWFAST)
    test_loader = DataLoader(test_dataset, batch_size=model_trainer_config.BATCH_SIZE_SLOWFAST)

    
    # Initialize trainer and evaluator
    trainer = Trainer_SlowFast(model_trainer_config, model, train_loader, val_loader, 
                     optimizer, scheduler, criterion, device, scaler)
    
    evaluator = Evaluator_SlowFast(model, test_loader, criterion, 
                         device, model_trainer_config.CLASS_NAME_SLOWFAST)
    
    # Training
    trainer.fit(model_trainer_config.MAX_EPOCHS_SLOWFAST)
    
    # Final evaluation
    results = evaluator.evaluate()
    wandb.log(results)

if __name__ == "__main__":
    main()