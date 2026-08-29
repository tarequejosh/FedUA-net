"""
Unit tests for Temperature Scaling module.
"""

import torch
import pytest
from src.uncertainty.temperature_scaling import TemperatureScaling


def test_temperature_scaling_initialization():
    ts = TemperatureScaling(init_temperature=1.5)
    assert abs(ts.temp_value - 1.5) < 1e-4
    x = torch.tensor([[2.0, 1.0]])
    scaled = ts(x)
    expected = x / 1.5
    assert torch.allclose(scaled, expected)


def test_temperature_scaling_fit():
    torch.manual_seed(42)
    # Synthetic overconfident logits
    val_logits = torch.randn(100, 4) * 5.0
    val_labels = torch.randint(0, 4, (100,))

    ts = TemperatureScaling()
    fitted_t = ts.fit(val_logits, val_labels, max_iter=20)
    assert fitted_t > 0.0, "Fitted temperature must be strictly positive"
