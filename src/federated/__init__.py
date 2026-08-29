"""
Federated learning modules for FedUA-Net.
"""

from .aggregator import aggregate_weights, compute_aggregation_weights
from .client import FederatedClient, FocalLossWithSmoothing

__all__ = [
    "aggregate_weights",
    "compute_aggregation_weights",
    "FederatedClient",
    "FocalLossWithSmoothing",
]
