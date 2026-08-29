"""
Uncertainty quantification and conformal prediction modules.
"""

from .temperature_scaling import TemperatureScaling
from .conformal import (
    compute_nonconformity_scores,
    calibrate_conformal_quantile,
    generate_prediction_sets,
    evaluate_conformal_coverage,
)
from .metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_diagnostic_metrics,
    compute_risk_coverage_curve,
)

__all__ = [
    "TemperatureScaling",
    "compute_nonconformity_scores",
    "calibrate_conformal_quantile",
    "generate_prediction_sets",
    "evaluate_conformal_coverage",
    "compute_expected_calibration_error",
    "compute_multiclass_brier_score",
    "compute_diagnostic_metrics",
    "compute_risk_coverage_curve",
]
