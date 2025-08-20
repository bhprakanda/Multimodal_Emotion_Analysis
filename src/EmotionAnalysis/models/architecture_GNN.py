import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from EmotionAnalysis.entity import ModelTrainerConfig


class EnhancedEmotionGNN(nn.Module):
    def __init__(self, model_trainer_config: ModelTrainerConfig, input_dim):
        super().__init__()

        self.model_trainer_config = model_trainer_config        
        self.speaker_emb = nn.Embedding(self.model_trainer_config.NUM_SPEAKERS_GNN, 32)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.model_trainer_config.HIDDEN_DIM_GNN),
            nn.GELU(),
            nn.LayerNorm(self.model_trainer_config.HIDDEN_DIM_GNN)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.LayerNorm(16)
        )
        # Graph attention layers with residual connections
        self.conv1 = GATConv(
            self.model_trainer_config.HIDDEN_DIM_GNN + 32, self.model_trainer_config.HIDDEN_DIM_GNN * 2,
            heads=4, edge_dim=16, dropout=self.model_trainer_config.DROPOUT_GNN
        )
        self.res1 = nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN + 32, self.model_trainer_config.HIDDEN_DIM_GNN * 8)
        
        self.conv2 = GATConv(
            self.model_trainer_config.HIDDEN_DIM_GNN * 8, self.model_trainer_config.HIDDEN_DIM_GNN,
            heads=6, edge_dim=16, dropout=self.model_trainer_config.DROPOUT_GNN
        )
        self.res2 = nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN * 8, self.model_trainer_config.HIDDEN_DIM_GNN * 6)
        
        self.conv3 = GATConv(
            self.model_trainer_config.HIDDEN_DIM_GNN * 6, self.model_trainer_config.HIDDEN_DIM_GNN,
            heads=4, concat=False, edge_dim=16, dropout=self.model_trainer_config.HIDDEN_DIM_GNN
        )
        
        # Auxiliary classifier with deeper architecture
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN * 6, self.model_trainer_config.HIDDEN_DIM_GNN * 2),
            nn.GELU(),
            nn.Dropout(self.model_trainer_config.DROPOUT_GNN),
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN * 2, self.model_trainer_config.OUTPUT_DIM_GNN)
        )
        
        # Emotion context module with attention
        self.emotion_att = nn.Sequential(
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN, self.model_trainer_config.HIDDEN_DIM_GNN),
            nn.Tanh(),
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN, 1)
        )
        
        # Main classifier with LayerNorm
        self.main_classifier = nn.Sequential(
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN * 2, self.model_trainer_config.HIDDEN_DIM_GNN),
            nn.GELU(),
            nn.Dropout(self.model_trainer_config.DROPOUT_GNN),
            nn.LayerNorm(self.model_trainer_config.HIDDEN_DIM_GNN),
            nn.Linear(self.model_trainer_config.HIDDEN_DIM_GNN, self.model_trainer_config.OUTPUT_DIM_GNN)
        )
    
    def forward(self, data):
        # Project node features
        x = self.input_proj(data.x)
        
        # Add speaker embeddings
        spk_emb = self.speaker_emb(data.speaker_id)
        x_base = torch.cat([x, spk_emb], dim=1)
        x = x_base
        
        # Project edge features
        edge_attr = self.edge_proj(data.edge_attr)
        edge_index = data.edge_index
        
        # Apply edge dropout during training
        if self.training and self.edge_dropout > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
        
        # Graph convolutions with residual connections
        x1 = F.gelu(self.conv1(x, edge_index, edge_attr))
        x1 = x1 + self.res1(x)  # Residual connection
        
        x2 = F.gelu(self.conv2(x1, edge_index, edge_attr))
        x2 = x2 + self.res2(x1)  # Residual connection
        
        x3 = F.gelu(self.conv3(x2, edge_index, edge_attr))
        
        # Auxiliary output from intermediate layer
        aux_out = self.aux_classifier(x2)
        
        # Emotion context pooling
        att_scores = self.emotion_att(x3)
        att_weights = F.softmax(att_scores, dim=0)
        context = torch.sum(att_weights * x3, dim=0, keepdim=True)
        
        # Combine with node features
        context = context.expand(x3.size(0), -1)
        x_final = torch.cat([x3, context], dim=1)
        
        # Final classification
        main_out = self.main_classifier(x_final)
        
        return main_out, aux_out