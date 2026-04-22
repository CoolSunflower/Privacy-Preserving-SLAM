"""
Privacy-Preserving Gaussian SLAM Module

This module implements semantic Gaussian excision for privacy preservation in SLAM.
It detects sensitive content (faces, screens, documents) and removes corresponding
Gaussians from the 3D map without degrading tracking or map quality.

Key components:
- detectors/: Privacy detection backends (YOLO, Grounding DINO + SAM)
- privacy_manager.py: Main orchestrator for privacy processing
- gaussian_excision.py: Forward projection and Gaussian pruning
- region_filler.py: Depth inpainting and filler Gaussian creation
- evaluation/: Metrics and baseline implementations
"""

from .privacy_manager import PrivacyManager
from .gaussian_excision import GaussianExcisionManager
from .region_filler import RegionFiller

__all__ = [
    "PrivacyManager",
    "GaussianExcisionManager",
    "RegionFiller",
]
