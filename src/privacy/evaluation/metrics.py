"""
Evaluation metrics for privacy-preserving Gaussian SLAM.

Includes:
- SSIM-Sensitive: Structural similarity in private regions
- Re-identification Score: Privacy protection effectiveness
- Excision Completeness: Coverage of private content removal
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PrivacyMetrics:
    """Collection of privacy evaluation metrics."""
    ate_rmse: float              # Trajectory accuracy (m)
    psnr_non_private: float      # PSNR in non-private regions (dB)
    ssim_non_private: float      # SSIM in non-private regions
    ssim_sensitive: float        # SSIM in private regions vs target
    reid_score: float            # Re-identification protection (0-1)
    excision_completeness: float # Fraction of private content removed
    false_excision_rate: float   # Non-private content incorrectly removed
    total_runtime_ms: float      # Total processing time

    def to_dict(self) -> Dict:
        return {
            "ate_rmse": self.ate_rmse,
            "psnr_non_private": self.psnr_non_private,
            "ssim_non_private": self.ssim_non_private,
            "ssim_sensitive": self.ssim_sensitive,
            "reid_score": self.reid_score,
            "excision_completeness": self.excision_completeness,
            "false_excision_rate": self.false_excision_rate,
            "total_runtime_ms": self.total_runtime_ms,
        }

    def __str__(self):
        return (
            f"PrivacyMetrics(\n"
            f"  ATE RMSE: {self.ate_rmse:.4f}m\n"
            f"  PSNR (non-priv): {self.psnr_non_private:.2f}dB\n"
            f"  SSIM (non-priv): {self.ssim_non_private:.4f}\n"
            f"  SSIM-Sensitive: {self.ssim_sensitive:.4f}\n"
            f"  Re-ID Score: {self.reid_score:.4f}\n"
            f"  Excision Complete: {self.excision_completeness:.2%}\n"
            f"  False Excision: {self.false_excision_rate:.2%}\n"
            f")"
        )


def compute_ssim_sensitive(
    rendered_excised: torch.Tensor,
    privacy_mask: torch.Tensor,
    target: str = "black",
    window_size: int = 11,
) -> float:
    """
    Compute SSIM in private regions against a target appearance.

    For complete excision, the private regions should match the target
    (black or neutral gray), resulting in HIGH SSIM with target.

    Args:
        rendered_excised: Rendered image after excision (3, H, W) or (H, W, 3)
        privacy_mask: Binary mask of private regions (H, W)
        target: Target appearance - "black" or "neutral_gray"
        window_size: SSIM window size

    Returns:
        SSIM value (0-1). Higher = better excision (matches target).
    """
    # Ensure CHW format
    if rendered_excised.dim() == 3 and rendered_excised.shape[-1] == 3:
        rendered_excised = rendered_excised.permute(2, 0, 1)

    C, H, W = rendered_excised.shape
    device = rendered_excised.device

    # Create target image
    if target == "black":
        target_img = torch.zeros_like(rendered_excised)
    elif target == "neutral_gray":
        target_img = torch.full_like(rendered_excised, 0.5)
    else:
        raise ValueError(f"Unknown target: {target}")

    # Expand mask for all channels
    mask = privacy_mask.unsqueeze(0).expand_as(rendered_excised)

    # Check if mask has enough pixels for meaningful SSIM
    num_private_pixels = mask[0].sum().item()
    if num_private_pixels < window_size * window_size:
        # Too few pixels for SSIM, use direct comparison
        if num_private_pixels == 0:
            return 1.0  # No private regions = perfect
        masked_excised = rendered_excised[mask].flatten()
        masked_target = target_img[mask].flatten()
        # Use L2 similarity: 1 - normalized_mse
        mse = F.mse_loss(masked_excised, masked_target).item()
        return max(0.0, 1.0 - mse)

    # Apply mask to images
    masked_excised = rendered_excised * mask
    masked_target = target_img * mask

    # Compute SSIM
    try:
        from pytorch_msssim import ssim
        ssim_value = ssim(
            masked_excised.unsqueeze(0),
            masked_target.unsqueeze(0),
            data_range=1.0,
            size_average=True
        )
        return ssim_value.item()
    except ImportError:
        # Fallback: structural comparison via correlation
        masked_excised_flat = masked_excised[mask].flatten()
        masked_target_flat = masked_target[mask].flatten()

        if masked_excised_flat.std() < 1e-6:
            # Constant value - check if matches target
            return 1.0 if torch.allclose(masked_excised_flat, masked_target_flat, atol=0.1) else 0.0

        # Pearson correlation as SSIM proxy
        corr = torch.corrcoef(torch.stack([
            masked_excised_flat,
            masked_target_flat
        ]))[0, 1]

        return max(0.0, corr.item())


def compute_reidentification_score(
    rendered_images: List[torch.Tensor],
    privacy_masks: List[torch.Tensor],
    detector=None,
    detection_threshold: float = 0.5,
) -> float:
    """
    Compute re-identification score.

    Re-ID Score = 1 - detection_rate
    A score of 1.0 means no private content is detectable.

    Args:
        rendered_images: List of rendered images after excision
        privacy_masks: List of original privacy masks (GT private regions)
        detector: Privacy detector to run on rendered images
        detection_threshold: Confidence threshold for detections

    Returns:
        Re-identification score (0-1). Higher = better privacy.
    """
    if len(rendered_images) == 0:
        return 1.0

    if detector is None:
        # Use YOLO detector by default
        try:
            from ..detectors.yolo_detector import YOLOPrivacyDetector
            detector = YOLOPrivacyDetector({
                "confidence_threshold": detection_threshold,
                "dilation_kernel": 0  # No dilation for evaluation
            })
            detector.load_model()
        except ImportError:
            print("[WARN] Cannot load detector for Re-ID score, returning 0.5")
            return 0.5

    total_detections = 0
    total_gt_regions = 0

    for img, gt_mask in zip(rendered_images, privacy_masks):
        # Skip if no GT private regions
        if gt_mask.sum() == 0:
            continue

        total_gt_regions += 1

        # Detect in rendered image
        result = detector.detect(img)

        # Check if any detections overlap with GT private regions
        if result.combined_mask.sum() > 0:
            # Compute overlap
            overlap = (result.combined_mask > 0.5) & (gt_mask > 0.5)
            if overlap.sum() > 0:
                total_detections += 1

    if total_gt_regions == 0:
        return 1.0

    detection_rate = total_detections / total_gt_regions
    reid_score = 1.0 - detection_rate

    return reid_score


def compute_excision_completeness(
    gaussians,
    gt_privacy_masks: Dict[int, torch.Tensor],
    cameras: Dict,
    threshold: float = 0.5,
) -> float:
    """
    Compute excision completeness.

    Measures what fraction of GT private pixels have no Gaussian coverage
    after excision.

    Args:
        gaussians: GaussianModel after excision
        gt_privacy_masks: Ground truth privacy masks per frame
        cameras: Camera objects per video_idx
        threshold: Projection threshold

    Returns:
        Completeness ratio (0-1). Higher = more complete excision.
    """
    from ..mask_utils import project_gaussians_to_image

    if len(gt_privacy_masks) == 0:
        return 1.0

    total_private_pixels = 0
    covered_private_pixels = 0

    for video_idx, camera in cameras.items():
        frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

        if frame_idx not in gt_privacy_masks:
            continue

        gt_mask = gt_privacy_masks[frame_idx]
        H, W = gt_mask.shape
        device = gt_mask.device

        # Project all Gaussians to this view
        means3D = gaussians.get_xyz
        u, v, valid = project_gaussians_to_image(means3D, camera)

        # Create coverage map
        coverage = torch.zeros(H, W, device=device)
        in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if in_bounds.any():
            u_int = u[in_bounds].long().clamp(0, W - 1)
            v_int = v[in_bounds].long().clamp(0, H - 1)
            coverage[v_int, u_int] = 1.0

        # Count private pixels and covered private pixels
        private_pixels = gt_mask > threshold
        num_private = private_pixels.sum().item()
        num_covered = (coverage > 0.5) & private_pixels
        num_covered = num_covered.sum().item()

        total_private_pixels += num_private
        covered_private_pixels += num_covered

    if total_private_pixels == 0:
        return 1.0

    # Completeness = fraction NOT covered
    completeness = 1.0 - (covered_private_pixels / total_private_pixels)

    return completeness


def compute_false_excision_rate(
    gaussians_before,
    gaussians_after,
    gt_privacy_masks: Dict[int, torch.Tensor],
    cameras: Dict,
) -> float:
    """
    Compute false excision rate.

    Measures what fraction of excised Gaussians were NOT in private regions.

    Args:
        gaussians_before: GaussianModel before excision (or count)
        gaussians_after: GaussianModel after excision
        gt_privacy_masks: Ground truth privacy masks
        cameras: Camera objects

    Returns:
        False excision rate (0-1). Lower = better precision.
    """
    # This requires tracking which specific Gaussians were removed
    # For now, provide a simplified estimate
    # TODO: Implement proper tracking

    return 0.0  # Placeholder


def compute_psnr(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute PSNR between rendered and ground truth images.

    Args:
        rendered: Rendered image (3, H, W) or (H, W, 3)
        gt: Ground truth image
        mask: Optional mask to restrict computation (H, W)

    Returns:
        PSNR in dB
    """
    # Ensure same format
    if rendered.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {rendered.shape} vs {gt.shape}")

    if mask is not None:
        # Expand mask
        if rendered.dim() == 3 and rendered.shape[0] == 3:
            mask = mask.unsqueeze(0).expand_as(rendered)
        else:
            mask = mask.unsqueeze(-1).expand_as(rendered)

        rendered = rendered[mask]
        gt = gt[mask]

    mse = F.mse_loss(rendered.float(), gt.float()).item()

    if mse < 1e-10:
        return 100.0  # Perfect match

    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def compute_ssim(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute SSIM between rendered and ground truth images.

    Args:
        rendered: Rendered image (C, H, W) or (H, W, C)
        gt: Ground truth image
        mask: Optional mask to restrict computation

    Returns:
        SSIM value (0-1)
    """
    try:
        from pytorch_msssim import ssim

        # Ensure CHW format
        if rendered.dim() == 3 and rendered.shape[-1] == 3:
            rendered = rendered.permute(2, 0, 1)
            gt = gt.permute(2, 0, 1)

        if mask is not None:
            mask_3d = mask.unsqueeze(0).expand_as(rendered)
            rendered = rendered * mask_3d
            gt = gt * mask_3d

        ssim_val = ssim(
            rendered.unsqueeze(0),
            gt.unsqueeze(0),
            data_range=1.0
        )
        return ssim_val.item()

    except ImportError:
        # Fallback to simple correlation
        if mask is not None:
            if rendered.dim() == 3 and rendered.shape[0] == 3:
                mask = mask.unsqueeze(0).expand_as(rendered)
            rendered = rendered[mask]
            gt = gt[mask]

        rendered_flat = rendered.flatten()
        gt_flat = gt.flatten()

        if rendered_flat.std() < 1e-6 or gt_flat.std() < 1e-6:
            return 0.0

        corr = torch.corrcoef(torch.stack([rendered_flat, gt_flat]))[0, 1]
        return max(0.0, corr.item())


def evaluate_privacy_full(
    gaussians,
    cameras: Dict,
    gt_privacy_masks: Dict[int, torch.Tensor],
    gt_images: Dict[int, torch.Tensor],
    render_fn,
    ate_rmse: float = 0.0,
    runtime_ms: float = 0.0,
) -> PrivacyMetrics:
    """
    Compute all privacy evaluation metrics.

    Args:
        gaussians: GaussianModel after privacy processing
        cameras: Camera objects
        gt_privacy_masks: Ground truth privacy masks
        gt_images: Ground truth images
        render_fn: Function to render Gaussians (viewpoint) -> image
        ate_rmse: Pre-computed trajectory accuracy
        runtime_ms: Total processing time

    Returns:
        PrivacyMetrics with all computed metrics
    """
    psnr_values = []
    ssim_values = []
    ssim_sensitive_values = []
    rendered_for_reid = []
    masks_for_reid = []

    for video_idx, camera in cameras.items():
        frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

        # Render
        render_result = render_fn(camera)
        if render_result is None:
            continue

        rendered = render_result["render"]  # (3, H, W)
        gt_image = gt_images.get(frame_idx, camera.original_image)
        privacy_mask = gt_privacy_masks.get(frame_idx, None)

        # Non-private region mask
        if privacy_mask is not None:
            non_private_mask = (privacy_mask <= 0.5).float()
        else:
            non_private_mask = torch.ones_like(rendered[0])

        # PSNR/SSIM on non-private regions
        psnr_val = compute_psnr(rendered, gt_image, non_private_mask)
        ssim_val = compute_ssim(rendered, gt_image, non_private_mask)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

        # SSIM-Sensitive on private regions
        if privacy_mask is not None and privacy_mask.sum() > 0:
            ssim_sens = compute_ssim_sensitive(rendered, privacy_mask, target="black")
            ssim_sensitive_values.append(ssim_sens)
            rendered_for_reid.append(rendered)
            masks_for_reid.append(privacy_mask)

    # Compute aggregated metrics
    psnr_non_private = np.mean(psnr_values) if psnr_values else 0.0
    ssim_non_private = np.mean(ssim_values) if ssim_values else 0.0
    ssim_sensitive = np.mean(ssim_sensitive_values) if ssim_sensitive_values else 1.0

    # Re-identification score
    reid_score = compute_reidentification_score(rendered_for_reid, masks_for_reid)

    # Excision completeness
    completeness = compute_excision_completeness(gaussians, gt_privacy_masks, cameras)

    return PrivacyMetrics(
        ate_rmse=ate_rmse,
        psnr_non_private=psnr_non_private,
        ssim_non_private=ssim_non_private,
        ssim_sensitive=ssim_sensitive,
        reid_score=reid_score,
        excision_completeness=completeness,
        false_excision_rate=0.0,  # TODO
        total_runtime_ms=runtime_ms,
    )
