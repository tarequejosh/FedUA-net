"""
Validation-guided Temperature Scaling for Post-Hoc Calibration.
Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """
    Post-hoc Temperature Scaling module.
    Learns a single positive scalar T on validation logits to minimize NLL.
    """
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by temperature: z / T
        """
        temperature = self.temperature.clamp(min=1e-3)
        return logits / temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01
    ) -> float:
        """
        Fits temperature parameter on validation data using L-BFGS optimizer.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())

    @property
    def temp_value(self) -> float:
        return float(self.temperature.item())
