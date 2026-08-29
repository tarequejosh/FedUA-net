"""
Unit tests for Federated Server Aggregation.
"""

import copy
import torch
import torch.nn as nn
import pytest
from src.models.fedua_model import SharedBackbone, FedUANetClientModel
from src.federated.aggregator import aggregate_weights, compute_aggregation_weights


def test_uniform_aggregation_weights():
    sample_sizes = [4855, 546, 14815]
    weights = compute_aggregation_weights(sample_sizes, agg_weight_type="uniform")
    assert len(weights) == 3
    assert all(abs(w - 1/3) < 1e-6 for w in weights), "Uniform weights must equal 1/K = 1/3"


def test_sample_proportional_weights():
    sample_sizes = [100, 200, 700]
    weights = compute_aggregation_weights(sample_sizes, agg_weight_type="sample_size")
    assert abs(weights[0] - 0.1) < 1e-6
    assert abs(weights[1] - 0.2) < 1e-6
    assert abs(weights[2] - 0.7) < 1e-6


def test_weight_aggregation_and_bn_isolation():
    backbone_global = SharedBackbone(backbone_name="efficientnet_v2_s", pretrained=False, attention_type="cbam")

    # Create 2 clients with different weights
    client1 = FedUANetClientModel(backbone=copy.deepcopy(backbone_global), num_classes=3)
    client2 = FedUANetClientModel(backbone=copy.deepcopy(backbone_global), num_classes=4)

    # Manually mutate projector weights
    with torch.no_grad():
        for p in client1.backbone.projector.parameters():
            p.fill_(1.0)
        for p in client2.backbone.projector.parameters():
            p.fill_(3.0)

    # Aggregate with equal weights (1/2, 1/2)
    aggregate_weights(backbone_global, [client1, client2], [0.5, 0.5], exclude_bn=True)

    for p in backbone_global.projector.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0)), "Aggregated parameter should equal mean (1.0 + 3.0)/2 = 2.0"
