"""
Privacy Manager - Main orchestrator for privacy-preserving Gaussian SLAM.

Coordinates detection, excision, and filling operations across different
modes (simultaneous, post-processing, hybrid).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time

from .detectors.base_detector import PrivacyMaskResult
from .gaussian_excision import GaussianExcisionManager, ExcisionStats
from .region_filler import RegionFiller, FillStats
from .mask_utils import resize_mask_to_features


@dataclass
class PrivacyState:
    """Tracks privacy state for a keyframe."""
    frame_idx: int
    video_idx: int
    mask_2d: torch.Tensor
    detections: list = field(default_factory=list)
    excised: bool = False
    filled: bool = False
    num_gaussians_affected: int = 0


@dataclass
class PrivacyProcessingStats:
    """Overall privacy processing statistics."""
    mode: str
    num_frames_processed: int
    num_keyframes_with_privacy: int
    total_private_pixels: int
    total_excised: int
    total_filled: int
    total_detection_time_ms: float
    total_excision_time_ms: float
    total_fill_time_ms: float

    def __str__(self):
        return (
            f"PrivacyStats(mode={self.mode}, "
            f"frames={self.num_frames_processed}, "
            f"excised={self.total_excised}, "
            f"filled={self.total_filled})"
        )


class PrivacyManager:
    """
    Main privacy management class for Gaussian SLAM.

    Supports three modes:
    1. simultaneous: Real-time detection during SLAM (YOLOv8)
    2. postprocess: Thorough detection after SLAM (Grounding DINO + SAM)
    3. hybrid: YOLOv8 during SLAM + Grounding DINO + SAM before export

    Usage:
        # Initialize
        privacy_mgr = PrivacyManager(config, device)

        # During SLAM (simultaneous/hybrid mode):
        privacy_mask = privacy_mgr.detect_realtime(image, frame_idx)
        if privacy_mask is not None:
            # Inject into uncertainty
            uncertainty = privacy_mgr.inject_privacy_uncertainty(uncertainty, privacy_mask)

        # After SLAM (postprocess/hybrid mode):
        num_excised, num_filled = privacy_mgr.postprocess_map(gaussians, cameras, dataset)
    """

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize the privacy manager.

        Args:
            config: Configuration dict with keys:
                - enable: Whether privacy processing is enabled
                - mode: "simultaneous", "postprocess", or "hybrid"
                - uncertainty_beta: High uncertainty value for privacy regions
                - yolo_config: YOLOv8 detector configuration
                - grounding_sam_config: Grounding DINO + SAM configuration
                - enable_excision: Whether to prune private Gaussians
                - enable_filling: Whether to fill excised regions
            device: Device for processing
        """
        self.config = config
        self.device = device
        self.enabled = config.get("enable", True)
        self.mode = config.get("mode", "hybrid")

        # Uncertainty injection parameters
        self.uncertainty_beta = config.get("uncertainty_beta", 100.0)

        # Detection components (lazy loaded)
        self.realtime_detector = None
        self.postprocess_detector = None

        # Excision and filling components
        self.excision_manager = GaussianExcisionManager(config)
        self.filler = RegionFiller(config) if config.get("enable_filling", True) else None

        # State tracking
        self.privacy_states: Dict[int, PrivacyState] = {}
        self.frame_masks: Dict[int, torch.Tensor] = {}

        # Initialize detectors based on mode
        if self.enabled:
            self._init_detectors()

    def _init_detectors(self):
        """Initialize detection backends based on mode."""
        if self.mode in ["simultaneous", "hybrid"]:
            detector_type = self.config.get("simultaneous_detector", "yolo")

            if detector_type == "yolo":
                from .detectors.yolo_detector import YOLOPrivacyDetector
                self.realtime_detector = YOLOPrivacyDetector(
                    self.config.get("yolo_config", {}),
                    self.device
                )
            elif detector_type == "grounding_dino":
                from .detectors.grounding_sam_detector import GroundingSAMDetector
                self.realtime_detector = GroundingSAMDetector(
                    self.config.get("grounding_sam_config", {}),
                    self.device
                )
            else:
                raise ValueError(f"Unknown simultaneous detector: {detector_type}")

            # Load model immediately for simultaneous mode
            self.realtime_detector.load_model()

        # Post-processing detector is lazy loaded

    def _ensure_postprocess_detector(self):
        """Lazy load post-processing detector."""
        if self.postprocess_detector is not None:
            return

        if self.mode in ["postprocess", "hybrid"]:
            from .detectors.grounding_sam_detector import GroundingSAMDetector
            self.postprocess_detector = GroundingSAMDetector(
                self.config.get("grounding_sam_config", {}),
                self.device
            )
            self.postprocess_detector.load_model()

    def detect_realtime(
        self,
        image: torch.Tensor,
        frame_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Real-time detection during SLAM.

        Args:
            image: RGB image (3, H, W)
            frame_idx: Current frame index

        Returns:
            Privacy mask (H, W) or None if no detections
        """
        if not self.enabled or self.realtime_detector is None:
            return None

        result = self.realtime_detector.detect(image)
        result.frame_idx = frame_idx

        if result.has_detections:
            self.frame_masks[frame_idx] = result.combined_mask
            return result.combined_mask

        return None

    def inject_privacy_uncertainty(
        self,
        uncertainty: torch.Tensor,
        privacy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject high uncertainty into private regions.

        This causes the mapping loss to effectively ignore these regions,
        preventing Gaussians from learning private appearance.

        Args:
            uncertainty: Current uncertainty map (H', W') from DINO features
            privacy_mask: Binary privacy mask (H, W) at image resolution

        Returns:
            Modified uncertainty with high values in private regions
        """
        # Resize mask to uncertainty resolution
        mask_resized = resize_mask_to_features(
            privacy_mask,
            tuple(uncertainty.shape[-2:])
        )

        # Inject high uncertainty using maximum (preserve existing high uncertainty)
        modified_uncertainty = torch.where(
            mask_resized > 0.5,
            torch.maximum(uncertainty, torch.full_like(uncertainty, self.uncertainty_beta)),
            uncertainty
        )

        return modified_uncertainty

    def get_privacy_mask_for_frame(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get stored privacy mask for a frame."""
        return self.frame_masks.get(frame_idx, None)

    def postprocess_map(
        self,
        gaussians,
        cameras: Dict,
        frame_reader=None,
    ) -> Tuple[int, int]:
        """
        Post-process the Gaussian map to remove private content.

        Args:
            gaussians: GaussianModel instance
            cameras: Dict of Camera objects (video_idx -> Camera)
            frame_reader: Optional dataset for loading images if needed

        Returns:
            Tuple of (num_excised, num_filled) Gaussians
        """
        if not self.enabled:
            return 0, 0

        # Ensure post-processing detector is loaded
        if self.mode in ["postprocess", "hybrid"]:
            self._ensure_postprocess_detector()

        total_excised = 0
        total_filled = 0

        # Collect all privacy masks
        all_privacy_masks = {}

        print(f"[PrivacyManager] Post-processing {len(cameras)} keyframes...")

        for video_idx, camera in cameras.items():
            frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

            # Try to get existing mask first (from simultaneous mode)
            mask = self.frame_masks.get(frame_idx, None)

            # Run post-processing detection if needed
            if mask is None and self.postprocess_detector is not None:
                image = camera.original_image
                result = self.postprocess_detector.detect(image)
                result.frame_idx = frame_idx

                if result.has_detections:
                    mask = result.combined_mask
                    self.frame_masks[frame_idx] = mask

                    # Record state
                    self.privacy_states[video_idx] = PrivacyState(
                        frame_idx=frame_idx,
                        video_idx=video_idx,
                        mask_2d=mask,
                        detections=result.detections,
                    )

            if mask is not None:
                all_privacy_masks[frame_idx] = mask

        if not all_privacy_masks:
            print("[PrivacyManager] No private regions detected")
            return 0, 0

        print(f"[PrivacyManager] Found privacy masks in {len(all_privacy_masks)} keyframes")

        # Perform excision
        if self.config.get("enable_excision", True):
            boundary_indices, excision_stats = self.excision_manager.identify_and_excise(
                gaussians=gaussians,
                privacy_masks=all_privacy_masks,
                cameras=cameras,
            )
            total_excised = excision_stats.num_excised
            print(f"[PrivacyManager] Excised {total_excised} Gaussians")

            # Mark states as excised
            for vid in self.privacy_states:
                self.privacy_states[vid].excised = True
                self.privacy_states[vid].num_gaussians_affected = excision_stats.num_private_gaussians

        # Perform filling
        if self.filler is not None and self.config.get("enable_filling", True):
            for video_idx, camera in cameras.items():
                frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

                if frame_idx not in all_privacy_masks:
                    continue

                mask = all_privacy_masks[frame_idx]
                num_filled = self.filler.fill_region(
                    gaussians=gaussians,
                    privacy_mask=mask,
                    camera=camera,
                )
                total_filled += num_filled

                if video_idx in self.privacy_states:
                    self.privacy_states[video_idx].filled = True

            print(f"[PrivacyManager] Filled with {total_filled} Gaussians")

        return total_excised, total_filled

    def sanitize_for_export(self, gaussians) -> int:
        """
        Final sanitization before PLY export.

        Removes any remaining tagged private Gaussians that weren't
        caught in post-processing.

        Args:
            gaussians: GaussianModel instance

        Returns:
            Number of Gaussians removed
        """
        stats = self.excision_manager.excise_tagged_gaussians(gaussians)
        return stats.num_excised

    def get_statistics(self) -> PrivacyProcessingStats:
        """Get overall privacy processing statistics."""
        total_private_pixels = sum(
            mask.sum().item() for mask in self.frame_masks.values()
        )

        return PrivacyProcessingStats(
            mode=self.mode,
            num_frames_processed=len(self.frame_masks),
            num_keyframes_with_privacy=len(self.privacy_states),
            total_private_pixels=int(total_private_pixels),
            total_excised=self.excision_manager.get_total_excised(),
            total_filled=self.filler.get_total_filled() if self.filler else 0,
            total_detection_time_ms=0.0,  # TODO: Track this
            total_excision_time_ms=0.0,
            total_fill_time_ms=0.0,
        )

    def get_state_summary(self) -> Dict:
        """Get summary of privacy states for logging."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "num_frames_with_masks": len(self.frame_masks),
            "num_keyframes_processed": len(self.privacy_states),
            "states": {
                vid: {
                    "frame_idx": state.frame_idx,
                    "excised": state.excised,
                    "filled": state.filled,
                    "num_gaussians": state.num_gaussians_affected,
                }
                for vid, state in self.privacy_states.items()
            }
        }


def create_privacy_manager(config: dict, device: str = "cuda:0") -> Optional[PrivacyManager]:
    """
    Factory function to create a PrivacyManager.

    Args:
        config: Full SLAM configuration
        device: Device for processing

    Returns:
        PrivacyManager instance or None if privacy not enabled
    """
    privacy_config = config.get("privacy", {})

    if not privacy_config.get("enable", False):
        return None

    return PrivacyManager(privacy_config, device)
