"""
FedUA-Net Model Architecture:
EfficientNetV2-S Feature Extractor with CBAM Attention and Decoupled Classification Heads.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from .cbam import CBAM, ChannelAttention, SpatialAttention


class SharedBackbone(nn.Module):
    """
    Shared convolutional feature extractor with optional attention mechanism.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        attention_type: str = "cbam",
        embed_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.attention_type = attention_type.lower() if attention_type else "none"

        if backbone_name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_v2_s(weights=weights)
            self.features = base.features
            in_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Attention layer
        if self.attention_type == "cbam":
            self.attention = CBAM(in_channels)
        elif self.attention_type == "spatial":
            self.attention = SpatialAttention()
        elif self.attention_type == "channel":
            self.attention = ChannelAttention(in_channels)
        elif self.attention_type == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, embed_dim),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        if self.attention_type == "spatial":
            feat = self.attention(feat) * feat
        elif self.attention_type == "channel":
            feat = self.attention(feat) * feat
        elif self.attention_type == "cbam":
            feat = self.attention(feat)
        emb = self.projector(self.pool(feat))
        return emb


class FedUANetClientModel(nn.Module):
    """
    Client-side model combining the shared backbone with a client-specific classification head.
    """
    def __init__(
        self,
        backbone: SharedBackbone,
        num_classes: int,
        embed_dim: int = 512
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CentralizedGlobalModel(nn.Module):
    """
    Centralized upper bound multi-task model with single shared backbone and 11-class global head.
    """
    def __init__(
        self,
        backbone: SharedBackbone,
        total_classes: int = 11,
        embed_dim: int = 512
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, total_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)
