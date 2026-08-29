"""
Unit tests for FedUA-Net Model Architectures.
"""

import torch
import pytest
from src.models.fedua_model import SharedBackbone, FedUANetClientModel, CentralizedGlobalModel


def test_shared_backbone_forward():
    backbone = SharedBackbone(
        backbone_name="efficientnet_v2_s",
        pretrained=False,
        attention_type="cbam",
        embed_dim=512,
        dropout=0.2
    )
    x = torch.randn(2, 3, 224, 224)
    emb = backbone(x)
    assert emb.shape == (2, 512), f"Expected embedding shape (2, 512), got {emb.shape}"


def test_client_model_forward_and_backward():
    backbone = SharedBackbone(
        backbone_name="efficientnet_v2_s",
        pretrained=False,
        attention_type="cbam",
        embed_dim=512
    )
    client_model = FedUANetClientModel(backbone=backbone, num_classes=4, embed_dim=512)
    x = torch.randn(2, 3, 224, 224)
    logits = client_model(x)
    assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"

    # Verify backprop
    loss = logits.sum()
    loss.backward()
    assert client_model.head.weight.grad is not None, "Gradients should flow back to classification head"


def test_centralized_model_forward():
    backbone = SharedBackbone(
        backbone_name="efficientnet_v2_s",
        pretrained=False,
        attention_type="none",
        embed_dim=512
    )
    centralized_model = CentralizedGlobalModel(backbone=backbone, total_classes=11, embed_dim=512)
    x = torch.randn(2, 3, 224, 224)
    logits = centralized_model(x)
    assert logits.shape == (2, 11), f"Expected logits shape (2, 11), got {logits.shape}"
