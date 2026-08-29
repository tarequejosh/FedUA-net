"""
Unit tests for Diagnostic and Calibration Metrics.
"""

import numpy as np
import pytest
from src.uncertainty.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_diagnostic_metrics,
    compute_risk_coverage_curve,
)


def test_ece_metric():
    # Perfectly calibrated case
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    labels = np.array([0, 0, 1, 1])
    ece = compute_expected_calibration_error(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_brier_score():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    brier = compute_multiclass_brier_score(probs, labels, num_classes=2)
    assert abs(brier - 0.0) < 1e-6, "Perfect predictions should have 0 Brier score"


def test_risk_coverage_curve():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 0, 1])
    curve = compute_risk_coverage_curve(probs, labels, coverage_thresholds=[0.5, 1.0])
    assert "acc_at_cov_50" in curve
    assert "aurc" in curve
    assert curve["acc_at_cov_50"] >= curve["acc_at_cov_100"]
