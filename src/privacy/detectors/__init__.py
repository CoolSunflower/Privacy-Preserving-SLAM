"""
Privacy detection backends.

Available detectors:
- YOLOPrivacyDetector: Fast YOLOv8-seg for real-time detection (~25ms/frame)
- GroundingSAMDetector: Open-vocabulary detection with pixel-perfect masks (~150ms/frame)
"""

from .base_detector import BasePrivacyDetector, PrivacyDetection, PrivacyMaskResult

__all__ = [
    "BasePrivacyDetector",
    "PrivacyDetection",
    "PrivacyMaskResult",
]
