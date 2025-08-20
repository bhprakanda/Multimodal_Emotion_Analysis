import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModalityProjector(nn.Module):
    def __init__(self, input_dims, proj_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(input_dims[0], proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim))
        
        self.audio_proj = nn.Sequential(
            nn.Linear(input_dims[1], 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim))
        
        self.video_proj = nn.Sequential(
            nn.Linear(input_dims[2], 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim))
        
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=-1))

    def forward(self, x):
        text = x[..., :self.text_proj[0].in_features]
        audio = x[..., self.text_proj[0].in_features:self.text_proj[0].in_features+self.audio_proj[0].in_features]
        video = x[..., self.text_proj[0].in_features+self.audio_proj[0].in_features:]
        
        text_proj = self.text_proj(text)
        audio_proj = self.audio_proj(audio)
        video_proj = self.video_proj(video)
        
        combined = torch.cat([text_proj, audio_proj, video_proj], dim=-1)
        gate_weights = self.gate(combined.mean(dim=1, keepdim=True))
        
        text_gated = text_proj * gate_weights[:, :, 0:1]
        audio_gated = audio_proj * gate_weights[:, :, 1:2]
        video_gated = video_proj * gate_weights[:, :, 2:3]
        
        return torch.cat([text_gated, audio_gated, video_gated], dim=-1), gate_weights

class MultimodalBiLSTM(nn.Module):
    def __init__(self, input_dims, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.projector = ModalityProjector(input_dims)
        self.input_norm = nn.LayerNorm(768)
        self.lstm = nn.LSTM(
            768, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.post_lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size)
        )
        
        self.contextual_features = None

    def forward(self, x, lengths):
        projected, gate_weights = self.projector(x)
        x = self.input_norm(projected)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        self.contextual_features = lstm_out.detach().clone()
        lstm_out = self.post_lstm_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out), gate_weights
    
    def get_contextual_features(self, normalize=True):
        if self.contextual_features is None:
            raise RuntimeError("No features available. Run forward pass first.")
        
        features = self.contextual_features
        if normalize:
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True)
            features = (features - mean) / (std + 1e-9)
        return features