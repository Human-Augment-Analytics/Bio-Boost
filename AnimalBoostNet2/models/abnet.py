from typing import List

from .tnet import TemporalNet as TNet

import torch.nn as nn
import torch

import timm

class AnimalBoostNet(nn.Module):
    def __init__(self,
                 backbone_name: str,
                 tnet_input_size: int,
                 tnet_output_size: int,
                 tnet_hidden_sizes: List[int],
                 tnet_dropout: float = 0.5,
                 temperature: float = 10.0,
                 device: str = 'cpu'):
        
        super(AnimalBoostNet, self).__init__()
        self.device: torch.device = torch.device(device)
        
        self.backbone = timm.create_model(
            model_name=backbone_name,
            num_classes=tnet_output_size
        ).to(device=self.device)
        self.backbone.eval()

        self.head = TNet(
            input_size=tnet_input_size,
            output_size=tnet_output_size,
            hidden_sizes=tnet_hidden_sizes,
            dropout=tnet_dropout
        ).to(device=self.device)

        self.temperature = temperature

        self.alpha_m_logit = nn.Parameter(torch.tensor(2.944, dtype=torch.float32))
        self.alpha_f_logit = nn.Parameter(torch.tensor(2.944, dtype=torch.float32))

        # self.alpha = torch.tensor(0.5, requires_grad=True, dtype=torch.float32)

    def forward(self, imgs: torch.Tensor, temp_features: torch.Tensor) -> torch.Tensor:
        img_logits: torch.Tensor = self.backbone(imgs)
        temp_logits: torch.Tensor = self.head(temp_features)

        alpha_m: torch.Tensor = torch.sigmoid(self.alpha_m_logit)
        alpha_f: torch.Tensor = torch.sigmoid(self.alpha_f_logit)

        img_confs: torch.Tensor = img_logits.softmax(dim=1)
        f_conf: torch.Tensor = img_confs[:, 0]
        m_conf: torch.Tensor = img_confs[:, 1]

        m_gt_f: torch.Tensor = torch.sigmoid(self.temperature * (m_conf - f_conf))
        m_gt_alpha_m: torch.Tensor = torch.sigmoid(self.temperature * (m_conf - alpha_m))
        f_gt_alpha_f: torch.Tensor = torch.sigmoid(self.temperature * (f_conf - alpha_f))

        w_m: torch.Tensor = m_gt_f * m_gt_alpha_m
        w_f: torch.Tensor = (1 - m_gt_f) * f_gt_alpha_f

        w_img: torch.Tensor = (w_m + w_f).reshape(-1, 1)
        logits: torch.Tensor = w_img * img_logits + (1 - w_img) * temp_logits
        
        return logits