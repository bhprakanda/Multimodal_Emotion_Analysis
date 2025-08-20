import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from EmotionAnalysis.config.configuration import PlotVisualizationConfig

def init_visualization(plot_visualization_config: PlotVisualizationConfig):
    plt.rcParams.update({
        'axes.facecolor': plot_visualization_config.BACKGROUND_COLOR,
        'figure.facecolor': plot_visualization_config.BACKGROUND_COLOR,
        'axes.edgecolor': plot_visualization_config.TEXT_COLOR,
        'axes.labelcolor': plot_visualization_config.LABEL_COLOR,
        'text.color': plot_visualization_config.TEXT_COLOR,
        'xtick.color': plot_visualization_config.TEXT_COLOR,
        'ytick.color': plot_visualization_config.TEXT_COLOR,
        'grid.color': '#4A4A4A',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlecolor': plot_visualization_config.TITLE_COLOR,
        'axes.labelsize': 12,
        'figure.dpi': 250,
        'figure.autolayout': True
    })


def plot_styled_confusion_matrix(plot_visualization_config: PlotVisualizationConfig, y_true, y_pred, class_names, seed):
    # Convert to numpy arrays for element-wise operations
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    """Create and display styled confusion matrix heatmap"""
    # Create subplots for both normalized and absolute counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=plot_visualization_config.BACKGROUND_COLOR)
    fig.suptitle(f'Confusion Matrices (Seed: {seed})', color=plot_visualization_config.TITLE_COLOR, fontsize=18, y=0.98)
    
    # Plot normalized confusion matrix (recall)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    
    ax1.set_facecolor(plot_visualization_config.BACKGROUND_COLOR)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sns.heatmap(
        cm_norm_df, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis',
        linewidths=0.5,
        annot_kws={'color': plot_visualization_config.TEXT_COLOR, 'size': 10},
        norm=norm,
        ax=ax1
    )
    ax1.set_title('Normalized by True Class (Recall)', color=plot_visualization_config.TITLE_COLOR, fontsize=14, pad=15)
    ax1.set_xlabel('Predicted Emotion', color=plot_visualization_config.LABEL_COLOR, fontsize=12, labelpad=10)
    ax1.set_ylabel('True Emotion', color=plot_visualization_config.LABEL_COLOR, fontsize=12, labelpad=10)
    ax1.tick_params(axis='both', colors=plot_visualization_config.TEXT_COLOR, labelsize=10)
    
    # Plot absolute counts confusion matrix
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_abs_df = pd.DataFrame(cm_abs, index=class_names, columns=class_names)
    
    ax2.set_facecolor(plot_visualization_config.BACKGROUND_COLOR)
    # Use logarithmic normalization for better visualization of large count ranges
    norm_abs = mpl.colors.LogNorm(vmin=1, vmax=cm_abs.max()) if cm_abs.max() > 0 else None
    sns.heatmap(
        cm_abs_df, 
        annot=True, 
        fmt='d',  # Integer format
        cmap='viridis',
        linewidths=0.5,
        annot_kws={'color': plot_visualization_config.TEXT_COLOR, 'size': 10},
        norm=norm_abs,
        ax=ax2
    )
    ax2.set_title('Absolute Sample Counts', color=plot_visualization_config.TITLE_COLOR, fontsize=14, pad=15)
    ax2.set_xlabel('Predicted Emotion', color=plot_visualization_config.LABEL_COLOR, fontsize=12, labelpad=10)
    ax2.set_ylabel('True Emotion', color=plot_visualization_config.LABEL_COLOR, fontsize=12, labelpad=10)
    ax2.tick_params(axis='both', colors=plot_visualization_config.TEXT_COLOR, labelsize=10)
    
    # Add colorbar for absolute counts
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(colors=plot_visualization_config.TEXT_COLOR)
    
    # Add statistics box
    total_samples = len(y_true_np)
    
    correct = np.sum(y_true_np == y_pred_np)

    accuracy = accuracy_score(y_true, y_pred)
    stats_text = (
        f"Total Samples: {total_samples}\n"
        f"Correct Predictions: {correct}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}"
    )
    
    fig.text(
        0.5, 0.05, 
        stats_text,
        ha='center', va='center',
        fontsize=12, color=plot_visualization_config.TEXT_COLOR,
        bbox=dict(facecolor='#3A3A3A', alpha=0.7, edgecolor='#1abc9c')
    )
    
    # Save and display
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make space for stats box
    save_path = f'confusion_matrix_seed_{seed}.png'
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")
    
    return save_path