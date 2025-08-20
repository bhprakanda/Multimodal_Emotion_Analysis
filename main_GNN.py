import os
import torch
import subprocess
import numpy as np

from EmotionAnalysis.utils.visualizer import init_visualization
from EmotionAnalysis.utils.common_utils import set_seed
from EmotionAnalysis.data.datahandler_GNN import DataHandler
from EmotionAnalysis.components.evaluator.model_evaluation_GNN import Evaluator_GNN
from EmotionAnalysis.components.trainer.model_trainer_GNN import Trainer_GNN
from EmotionAnalysis.models.architecture_GNN import EnhancedEmotionGNN
from EmotionAnalysis.training.loss.EnhancedAdaptiveFocalLoss import EnhancedAdaptiveFocalLoss

from EmotionAnalysis.config.configuration import ConfigurationManager


def main():
    # Source the script "scripts/setup_environment.sh"
    subprocess.run(['source', 'setup_environment.sh'], shell=True)

    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_configuration()
    model_evaluation_config = config_manager.get_model_evaluation_config()
    plot_visualization_config = config_manager.get_plot_visualization_config()

    all_test_results = []
    
    data_handler = DataHandler(model_trainer_config)
    data_handler.load_data(model_trainer_config.TRAIN_DATA_GNN)
    
    model_trainer_config.NUM_SPEAKERS_GNN = len(data_handler.speaker_to_idx)
    
    for seed_idx, seed in enumerate(model_evaluation_config.SEEDS):
        print(f"\n\n{'='*80}\nENHANCED EMOTION CLASSIFICATION (Seed {seed_idx+1}/{len(model_evaluation_config.SEEDS)})\n{'='*80}")
        
        # Prepare datasets
        datasets = data_handler.prepare_datasets(seed)
        train_loader, dev_loader, test_loader = data_handler.create_loaders(datasets, seed)
        
        # Initialize model and loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(datasets['train'][0].x[0])
        model = EnhancedEmotionGNN(model_trainer_config, input_dim).to(device)
        criterion = EnhancedAdaptiveFocalLoss(
            data_handler.class_counts,
            model_trainer_config.MINORITY_CLASSES_GNN
        ).to(device)
        
        # Train
        trainer = Trainer_GNN(model_trainer_config, model, criterion, train_loader, dev_loader, device, seed)
        trainer.train(model_trainer_config.MAX_EPOCHS_GNN)
        
        # Evaluate
        model.load_state_dict(torch.load(model_trainer_config.MODEL_SAVE_PATH_GNN.format(seed)))
        evaluator = Evaluator_GNN(plot_visualization_config, model, test_loader, device)
        test_f1 = evaluator.evaluate(seed)
        all_test_results.append(test_f1)
    
    # Statistical reporting
    mean_f1 = np.mean(all_test_results)
    std_f1 = np.std(all_test_results)
    ci = 1.96 * std_f1 / np.sqrt(len(model_evaluation_config.SEEDS))
    
    print(f"\n\n{'='*80}\nFINAL STATISTICAL REPORT\n{'='*80}")
    print(f"Test F1 Scores: {all_test_results}")
    print(f"Mean Weighted F1: {mean_f1:.4f}")
    print(f"Standard Deviation: {std_f1:.4f}")
    print(f"95% Confidence Interval: ±{ci:.4f}")

if __name__ == "__main__":
    main()