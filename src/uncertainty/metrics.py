"""
Diagnostic and Uncertainty Metric Computations:
Accuracy, Macro-F1, MCC, ECE (Expected Calibration Error), Brier Score, and Risk-Coverage.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, brier_score_loss


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins:
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int
) -> float:
    """
    Computes multi-class Brier score:
    BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{i,c} - y_{i,c})^2
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_diagnostic_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Computes standard diagnostic evaluation metrics.
    """
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(labels, preds))
    ece = compute_expected_calibration_error(probs, labels)
    brier = compute_multiclass_brier_score(probs, labels, num_classes)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "mcc": mcc,
        "ece": ece,
        "brier": brier,
    }


def compute_risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage_thresholds: List[float] = [0.50, 0.70, 0.80, 0.90, 0.95]
) -> Dict[str, float]:
    """
    Computes selective classification accuracy across coverage thresholds.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    sorted_order = np.argsort(-confidences)

    results = {}
    for cov in coverage_thresholds:
        k = int(np.ceil(cov * len(labels)))
        selected_idx = sorted_order[:k]
        cov_acc = float(np.mean(predictions[selected_idx] == labels[selected_idx]))
        results[f"acc_at_cov_{int(cov*100)}"] = cov_acc

    # Area Under Risk-Coverage Curve (AURC)
    sweep_covs = np.linspace(0.1, 1.0, 100)
    sweep_accs = []
    for cov in sweep_covs:
        k = max(1, int(np.ceil(cov * len(labels))))
        selected_idx = sorted_order[:k]
        sweep_accs.append(np.mean(predictions[selected_idx] == labels[selected_idx]))

    aurc = float(np.trapezoid(sweep_accs, sweep_covs)) if hasattr(np, 'trapezoid') else float(np.trapz(sweep_accs, sweep_covs))
    results["aurc"] = aurc
    return results
