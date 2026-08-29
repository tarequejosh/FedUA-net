"""
Unit tests for Conformal Prediction (APS).
"""

import numpy as np
import pytest
from src.uncertainty.conformal import (
    compute_nonconformity_scores,
    calibrate_conformal_quantile,
    generate_prediction_sets,
    evaluate_conformal_coverage,
)


def test_conformal_pipeline():
    np.random.seed(42)
    n_samples = 200
    n_classes = 4

    # Generate synthetic softmax probabilities
    raw = np.random.exponential(scale=1.0, size=(n_samples, n_classes))
    probs = raw / np.sum(raw, axis=1, keepdims=True)
    labels = np.random.randint(0, n_classes, size=n_samples)

    # 1. Non-conformity scores
    scores = compute_nonconformity_scores(probs, labels)
    assert len(scores) == n_samples
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    # 2. Calibrate quantile at alpha = 0.10 (90% target coverage)
    q_hat = calibrate_conformal_quantile(scores[:100], alpha=0.10)
    assert 0.0 <= q_hat <= 1.0

    # 3. Generate prediction sets
    test_probs = probs[100:]
    test_labels = labels[100:]
    psets, sizes = generate_prediction_sets(test_probs, q_hat)
    assert len(psets) == len(test_probs)

    # 4. Evaluate coverage
    coverage, mean_size = evaluate_conformal_coverage(psets, test_labels)
    assert 0.0 <= coverage <= 1.0
    assert 1.0 <= mean_size <= n_classes
