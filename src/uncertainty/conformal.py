"""
Adaptive Prediction Sets (APS) Split-Conformal Prediction.
Romano et al., "Classification with Valid and Adaptive Coverage", NeurIPS 2020.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Computes APS non-conformity scores:
    s_i = sum of sorted probabilities down to and including the true label.
    """
    n_samples = len(labels)
    scores = np.zeros(n_samples, dtype=np.float32)

    # Sort probabilities descending
    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)

    for i in range(n_samples):
        true_label = labels[i]
        rank_idx = np.where(sorted_indices[i] == true_label)[0][0]
        scores[i] = np.sum(sorted_probs[i, : rank_idx + 1])

    return scores


def calibrate_conformal_quantile(
    cal_scores: np.ndarray,
    alpha: float = 0.10
) -> float:
    """
    Computes split-conformal quantile threshold q_hat with finite-sample correction:
    q_hat = Quantile({s_1, ..., s_n}; ceil((n+1)(1-alpha)) / n)
    """
    n = len(cal_scores)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(1.0, max(0.0, level))
    q_hat = float(np.quantile(cal_scores, level, method="higher"))
    return q_hat


def generate_prediction_sets(
    test_probs: np.ndarray,
    q_hat: float
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generates prediction sets for test samples given calibrated quantile q_hat.
    Returns:
        prediction_sets: list of class indices in the prediction set for each sample
        set_sizes: array of set size per sample
    """
    n_samples = len(test_probs)
    sorted_indices = np.argsort(-test_probs, axis=1)
    sorted_probs = np.take_along_axis(test_probs, sorted_indices, axis=1)
    cumulative_probs = np.cumsum(sorted_probs, axis=1)

    prediction_sets = []
    set_sizes = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        # find smallest k such that cumulative sum >= q_hat
        cutoff_mask = cumulative_probs[i] >= q_hat
        if np.any(cutoff_mask):
            k = np.argmax(cutoff_mask) + 1
        else:
            k = test_probs.shape[1]

        pset = sorted_indices[i, :k].tolist()
        prediction_sets.append(pset)
        set_sizes[i] = len(pset)

    return prediction_sets, set_sizes


def evaluate_conformal_coverage(
    prediction_sets: List[List[int]],
    test_labels: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluates empirical marginal coverage and average prediction set size.
    Returns:
        empirical_coverage: fraction of test samples where true label in prediction set
        mean_set_size: average number of classes in prediction set
    """
    n = len(test_labels)
    covered = sum(test_labels[i] in prediction_sets[i] for i in range(n))
    empirical_coverage = covered / n
    mean_set_size = float(np.mean([len(s) for s in prediction_sets]))
    return empirical_coverage, mean_set_size
