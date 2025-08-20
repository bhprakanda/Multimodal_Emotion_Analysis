import json
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader

from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.models.architecture_BiLSTM import MultimodalBiLSTM
from EmotionAnalysis.data.dialogue_dataset_BiLSTM import DialogueDataset
from EmotionAnalysis.components.trainer.model_trainer_BiLSTM import Trainer_BiLSTM
from EmotionAnalysis.data.dialogue_dataset_BiLSTM import dialogue_collate_fn
from EmotionAnalysis.training.loss.AdaptiveFocalLoss import AdaptiveFocalLoss
from EmotionAnalysis.components.evaluator.model_evaluation_BiLSTM import Evaluator_BiLSTM
from EmotionAnalysis.components.data_preparation.data_preprocessing import load_and_preprocess_data, compute_class_weights


def main():
    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_configuration()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB
    config = {
        "architecture": model_trainer_config.NAME_BILSTM,
        "max_epoch": model_trainer_config.MAX_EPOCH_BILSTM,
        "batch_size": model_trainer_config.BATCH_SIZE_BILSTM,
        "optimizer": "AdamW",
        "learning_rate": model_trainer_config.LEARNING_RATE_BILSTM,
        "gamma_base": model_trainer_config.GAMMA_BASE_BILSTM,
        "weight_decay": model_trainer_config.WEIGHT_DECAY_BILSTM,
        "hidden_size": model_trainer_config.HIDDEN_SIZE_BILSTM,
        "num_layers": model_trainer_config.NUM_LAYERS_BILSTM,
        "dropout": model_trainer_config.DROPOUT_BILSTM,
        "patience": model_trainer_config.PATIENCE_BILSTM,
        "modality_dims": model_trainer_config.MODALITY_DIMS_BILSTM
    }
    
    wandb.init(project="Multimodal_Features_BiLSTM", config=config.to_dict())
    
    # Load and preprocess data
    original_data = load_and_preprocess_data(model_trainer_config.DATA_PATH_BILSTM)
    
    # Create datasets
    train_dataset = DialogueDataset(
        original_data["train"], 
        minority_classes=model_trainer_config.MINORITY_CLASSES_BILSTM,
        oversample_factor=2
    )
    dev_dataset = DialogueDataset(original_data["dev"])
    test_dataset = DialogueDataset(original_data["test"])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_trainer_config.BATCH_SIZE_BILSTM, 
        shuffle=True, 
        collate_fn=dialogue_collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=model_trainer_config.BATCH_SIZE_BILSTM, 
        shuffle=False, 
        collate_fn=dialogue_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=model_trainer_config.BATCH_SIZE_BILSTM, 
        shuffle=False, 
        collate_fn=dialogue_collate_fn
    )
    
    # Initialize model
    model = MultimodalBiLSTM(
        input_dims=model_trainer_config.MODALITY_DIMS_BILSTM,
        hidden_size=model_trainer_config.HIDDEN_SIZE_BILSTM,
        num_layers=model_trainer_config.NUM_LAYERS_BILSTM,
        output_size=model_trainer_config.OUTPUT_SIZE_BILSTM,
        dropout=model_trainer_config.DROPOUT_BILSTM
    ).to(device)
    
    # Compute class weights
    temp_dataset = DialogueDataset(original_data["train"], minority_classes=None)
    all_labels = []
    for _, labels in temp_dataset:
        all_labels.extend(labels.numpy().tolist())
    _, class_counts = compute_class_weights(np.array(all_labels))
    
    criterion = AdaptiveFocalLoss(
        gamma_base=model_trainer_config.GAMMA_BASE_BILSTM,
        class_counts=model_trainer_config.CLASS_COUNTS_BiLSTM,
        smoothing=model_trainer_config.SMOOTHING_BILSTM,
        alpha=model_trainer_config.ALPHA_BILSTM
    ).to(device)
    
    # Train the model
    trainer = Trainer_BiLSTM(model, model_trainer_config, device)
    trained_model = trainer.train(train_loader, dev_loader, criterion)
    
    # Evaluate
    evaluator = Evaluator_BiLSTM(trained_model, device)
    evaluator.evaluate(test_loader)
    
    # Extract features
    datasets = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    original_data = trainer.extract_all_features(datasets, original_data)
    
    # Save enhanced data
    with open(model_trainer_config.OUTPUT_PATH_BILSTM, "w") as f:
        json.dump(original_data, f, indent=2)
    print(f"Saved enhanced data with BiLSTM features to {model_trainer_config.OUTPUT_PATH_BILSTM}")
    
    wandb.finish()

if __name__ == "__main__":
    main()