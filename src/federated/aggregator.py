"""
Federated Model Parameter Aggregation:
Uniform weighting (Eq. 8) and sample-proportional weighting.
"""

from typing import Dict, List
import copy
import torch
import torch.nn as nn


def aggregate_weights(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: List[float],
    exclude_bn: bool = True
) -> None:
    """
    Aggregates client model parameters into the global model backbone.
    Parameters matching BatchNorm layers ('bn', 'running_mean', 'running_var', 'num_batches_tracked')
    or local classification heads ('head') are excluded when exclude_bn is True.
    """
    total_weight = sum(client_weights)
    norm_weights = [w / total_weight for w in client_weights]

    global_dict = global_model.state_dict()
    updated_dict = copy.deepcopy(global_dict)

    for key in global_dict.keys():
        # Exclude client-specific heads
        if "head" in key:
            continue

        # Exclude local Batch Normalization layers
        if exclude_bn and any(bn_key in key.lower() for bn_key in ["bn", "running_mean", "running_var", "num_batches_tracked"]):
            continue

        # Check if parameter is floating point
        if not global_dict[key].is_floating_point():
            continue

        # Weighted average across participating clients
        weighted_param = torch.zeros_like(global_dict[key])
        for idx, client_model in enumerate(client_models):
            client_dict = client_model.state_dict()
            if key in client_dict:
                weighted_param += norm_weights[idx] * client_dict[key].to(global_dict[key].device)

        updated_dict[key] = weighted_param

    global_model.load_state_dict(updated_dict)


def compute_aggregation_weights(
    client_sample_sizes: List[int],
    agg_weight_type: str = "uniform"
) -> List[float]:
    """
    Computes aggregation weights across K clients.
    - 'uniform': wk = 1/K for all clients (Eq. 8 in paper)
    - 'sample_size': wk proportional to client dataset size Nk
    """
    num_clients = len(client_sample_sizes)
    if agg_weight_type.lower() == "uniform":
        return [1.0 / num_clients] * num_clients
    elif agg_weight_type.lower() in ["sample_size", "sample_weighted"]:
        total_samples = sum(client_sample_sizes)
        return [s / total_samples for s in client_sample_sizes]
    else:
        raise ValueError(f"Unknown aggregation weight type: {agg_weight_type}")
