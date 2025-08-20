class EnhancedAdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma_base=3.0, class_counts=None, smoothing=0.15, 
                 alpha=0.8, beta=0.7, minority_classes=[2, 3, 5], penalty_factor=4.0):
        super().__init__()
        self.gamma_base = gamma_base
        self.smoothing = smoothing
        self.alpha = alpha
        self.beta = beta
        self.minority_classes = minority_classes
        self.penalty_factor = penalty_factor
        
        if class_counts is not None:
            weights = 1.0 / np.power(class_counts, 0.65)
            minority_mask = np.isin(np.arange(len(weights)), minority_classes)
            weights[minority_mask] *= (1 + self.beta)
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
        
        minority_mask = torch.isin(targets, torch.tensor(self.minority_classes).to(targets.device))
        minority_penalty = self.penalty_factor * minority_mask.float()
        focal_loss = focal_loss * (1 + minority_penalty)
        
        return focal_loss.mean()