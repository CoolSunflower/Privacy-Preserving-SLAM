"""
Evaluation utilities for privacy-preserving SLAM.

Contains:
- metrics: SSIM-Sensitive, Re-ID Score, Excision Completeness
- baselines: Comparative methods (A through F)
"""

from .metrics import (
    PrivacyMetrics,
    compute_ssim_sensitive,
    compute_reidentification_score,
    compute_excision_completeness,
    compute_psnr,
    compute_ssim,
    evaluate_privacy_full,
)

from .baselines import (
    BaselineConfig,
    BaselineRunner,
    InputBlurringMode,
    UncertaintyOnlyMode,
    ComparisonEvaluator,
)

__all__ = [
    # Metrics
    "PrivacyMetrics",
    "compute_ssim_sensitive",
    "compute_reidentification_score",
    "compute_excision_completeness",
    "compute_psnr",
    "compute_ssim",
    "evaluate_privacy_full",
    # Baselines
    "BaselineConfig",
    "BaselineRunner",
    "InputBlurringMode",
    "UncertaintyOnlyMode",
    "ComparisonEvaluator",
]
