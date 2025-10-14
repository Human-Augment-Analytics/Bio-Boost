from typing import List
from .tnet import TemporalNet as TNet

import torch.nn as nn
import torch

import timm

class AnimalBoostNet(nn.Module):
    def __init__(self,
                 backbone_name: str,
                 backbone_weights: str,
                 tnet_input_size: int,
                 tnet_output_size: int,
                 tnet_hidden_sizes: List[int],
                 tnet_dropout: float = 0.5,
                 device: str = 'cpu'):
        
        self.device: torch.device = torch.device(device)
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=True,
            checkpoint_path=backbone_weights
        ).to(device=self.device)
        self.backbone.eval()

        self.head = TNet(
            input_size=tnet_input_size,
            output_size=tnet_output_size,
            hidden_sizes=tnet_hidden_sizes,
            dropout=tnet_dropout
        ).to(device=self.device)

    def forward(self, imgs: torch.Tensor, temp_features: torch.Tensor) -> torch.Tensor:
        img_logits: torch.Tensor = self.backbone(imgs)
        temp_logits: torch.Tensor = self.head(temp_features)

        logits: torch.Tensor = img_logits + temp_logits

        return logits