import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50


class MaskedSlowFast(nn.Module):
    """Simplified Model Architecture"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = slowfast_r50(pretrained=True, progress=True)
        self.backbone.blocks = self.backbone.blocks[:-1]  # Remove classification head

        # Add batch normalization before classifier
        self.feature_bn = nn.BatchNorm1d(2304)
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Freeze initial layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Gradual unfreezing setup
        self.unfreeze_stages = {5: False, 4: False, 3: False}

    def unfreeze_layers(self, epoch):
        """Gradual layer unfreezing during training"""
        if epoch >= 3 and not self.unfreeze_stages[5]:
            self._unfreeze_stage(5)
        if epoch >= 6 and not self.unfreeze_stages[4]:
            self._unfreeze_stage(4)
        if epoch >= 9 and not self.unfreeze_stages[3]:
            self._unfreeze_stage(3)
            
    def _unfreeze_stage(self, stage):
        """Unfreeze a specific stage of the backbone"""
        for param in self.backbone.blocks[stage].parameters():
            param.requires_grad = True
        self.unfreeze_stages[stage] = True
        print(f"Unfroze stage {stage} layers")

    def forward(self, slow_input, fast_input, slow_mask, fast_mask):
        # Apply masks to zero out padded frames
        slow_input = slow_input * slow_mask[:, None, :, None, None]
        fast_input = fast_input * fast_mask[:, None, :, None, None]
        
        # Get features
        features = self.backbone([slow_input, fast_input])
        features = features.view(features.size(0), -1)

        # Apply feature batch normalization
        features = self.feature_bn(features)
        
        return self.classifier(features)