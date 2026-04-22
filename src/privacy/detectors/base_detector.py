"""
Base class for privacy detectors.

Provides abstract interface for all privacy detection backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import time


@dataclass
class PrivacyDetection:
    """Single detection result for a private region."""
    mask: torch.Tensor           # Binary mask (H, W), float [0,1]
    confidence: float            # Detection confidence [0,1]
    category: str                # e.g., "face", "screen", "document", "person"
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) bounding box
    instance_id: int = -1        # For tracking same instance across frames


@dataclass
class PrivacyMaskResult:
    """Aggregated privacy mask for a frame."""
    combined_mask: torch.Tensor       # Union of all private regions (H, W)
    detections: List[PrivacyDetection] = field(default_factory=list)
    frame_idx: int = -1
    processing_time_ms: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return self.combined_mask.sum() > 0

    @property
    def coverage_ratio(self) -> float:
        """Fraction of image covered by privacy mask."""
        return (self.combined_mask > 0.5).float().mean().item()


class BasePrivacyDetector(ABC):
    """
    Abstract base class for privacy detectors.

    All privacy detectors must implement:
    - detect(): Run detection on an image
    - load_model(): Load detection model weights
    """

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize the privacy detector.

        Args:
            config: Configuration dictionary with detector-specific settings
            device: Device to run detection on
        """
        self.config = config
        self.device = device
        self.categories = config.get("categories", ["person", "face"])
        self.dilation_kernel_size = config.get("dilation_kernel", 7)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self._model_loaded = False

    @abstractmethod
    def detect(self, image: torch.Tensor) -> PrivacyMaskResult:
        """
        Detect private regions in an image.

        Args:
            image: RGB image tensor (3, H, W) or (H, W, 3), values in [0, 1]

        Returns:
            PrivacyMaskResult with combined mask and individual detections
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load detection model weights."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True

    def dilate_mask(self, mask: torch.Tensor, kernel_size: Optional[int] = None) -> torch.Tensor:
        """
        Dilate mask to ensure complete coverage of private regions.

        Args:
            mask: Binary mask (H, W) with values in [0, 1]
            kernel_size: Dilation kernel size (uses config default if None)

        Returns:
            Dilated mask (H, W)
        """
        if kernel_size is None:
            kernel_size = self.dilation_kernel_size

        if kernel_size <= 0:
            return mask

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

        # Ensure mask is 4D for conv2d
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)

        # Apply dilation via convolution
        dilated = F.conv2d(mask_4d, kernel, padding=padding)

        # Threshold to binary
        result = (dilated.squeeze() > 0).float()

        return result

    def preprocess_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for detection.

        Args:
            image: Input image tensor

        Returns:
            Tuple of (preprocessed image, original size (H, W))
        """
        # Handle different input formats
        if image.dim() == 3:
            if image.shape[0] == 3:  # (3, H, W)
                H, W = image.shape[1], image.shape[2]
            else:  # (H, W, 3)
                H, W = image.shape[0], image.shape[1]
                image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected 3D tensor, got {image.dim()}D")

        # Ensure on correct device
        if image.device != torch.device(self.device):
            image = image.to(self.device)

        # Ensure float [0, 1]
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        return image, (H, W)

    def create_empty_result(self, height: int, width: int, frame_idx: int = -1) -> PrivacyMaskResult:
        """Create an empty result when no detections are found."""
        return PrivacyMaskResult(
            combined_mask=torch.zeros(height, width, device=self.device),
            detections=[],
            frame_idx=frame_idx,
            processing_time_ms=0.0
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self._model_loaded})"
