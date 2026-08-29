"""
Client-side training and local evaluation logic.
"""

from typing import Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FocalLossWithSmoothing(nn.Module):
    """
    Balanced Focal Cross-Entropy loss with label smoothing.
    """
    def __init__(
        self,
        num_classes: int,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing target distribution
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        focal_weight = (1.0 - probs) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets].unsqueeze(1)
            loss = -alpha_t * focal_weight * smooth_targets * log_probs
        else:
            loss = -focal_weight * smooth_targets * log_probs

        return loss.sum(dim=-1).mean()


class FederatedClient:
    """
    Individual Hospital Node participating in cross-silo federated learning.
    """
    def __init__(
        self,
        client_id: int,
        name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.client_id = client_id
        self.name = name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        proximal_global_model: Optional[nn.Module] = None,
        mu: float = 0.0
    ) -> float:
        """
        Executes one local training epoch. Optional proximal penalty for FedProx.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            # FedProx proximal regularization term
            if proximal_global_model is not None and mu > 0.0:
                prox_loss = 0.0
                for w, w_t in zip(self.model.backbone.parameters(), proximal_global_model.parameters()):
                    prox_loss += ((w - w_t.to(self.device)) ** 2).sum()
                loss += (mu / 2.0) * prox_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            n_samples += len(targets)

        return total_loss / max(n_samples, 1)

    def evaluate(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs inference and extracts test logits, predicted probabilities, and ground truth labels.
        """
        self.model.eval()
        all_logits, all_probs, all_targets = [], [], []

        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = F.softmax(logits, dim=-1)

                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.numpy())

        return (
            np.concatenate(all_logits, axis=0),
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_targets, axis=0),
        )
