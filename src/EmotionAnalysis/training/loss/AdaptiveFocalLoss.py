import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma_base=2.0, class_counts=None, smoothing=0.1, alpha=0.8):
        super().__init__()
        self.gamma_base = gamma_base
        self.smoothing = smoothing
        self.alpha = alpha
        
        if class_counts is not None:
            weights = 1.0 / np.power(class_counts, 0.5)
            weights = np.maximum(weights, self.alpha * np.max(weights))
            normalized_weights = weights / weights.sum() * len(class_counts)
            self.register_buffer('gamma_weights', torch.tensor(normalized_weights, dtype=torch.float32))
        else:
            self.register_buffer('gamma_weights', torch.ones(7))

    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        pt = torch.exp(-loss)
        gamma_weights = self.gamma_weights.to(targets.device)
        gamma = gamma_weights[targets] * self.gamma_base
        focal_loss = (1 - pt) ** gamma * loss
        return focal_loss.mean()