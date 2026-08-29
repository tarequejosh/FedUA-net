"""
Unit tests for CBAM (Convolutional Block Attention Module).
"""

import torch
import pytest
from src.models.cbam import ChannelAttention, SpatialAttention, CBAM


def test_channel_attention():
    x = torch.randn(2, 64, 14, 14)
    ca = ChannelAttention(in_planes=64, ratio=16)
    out = ca(x)
    assert out.shape == (2, 64, 1, 1), f"Expected shape (2, 64, 1, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Attention weights must be bounded in [0, 1]"


def test_spatial_attention():
    x = torch.randn(2, 64, 14, 14)
    sa = SpatialAttention(kernel_size=7)
    out = sa(x)
    assert out.shape == (2, 1, 14, 14), f"Expected shape (2, 1, 14, 14), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Attention weights must be bounded in [0, 1]"


def test_cbam_forward():
    x = torch.randn(4, 1280, 7, 7)
    cbam = CBAM(in_planes=1280, ratio=16, kernel_size=7)
    out = cbam(x)
    assert out.shape == x.shape, f"CBAM output shape {out.shape} should match input shape {x.shape}"
