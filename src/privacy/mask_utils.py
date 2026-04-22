"""
Mask utilities for privacy-preserving Gaussian SLAM.

Provides functions for projecting 2D privacy masks to 3D Gaussian space
using forward projection of Gaussian centers.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def project_gaussians_to_image(
    means3D: torch.Tensor,
    camera,
    depth_filter: bool = True,
    min_depth: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussian centers to 2D image coordinates.

    Args:
        means3D: Gaussian center positions (N, 3) in world coordinates
        camera: Camera object with R, T, intrinsics
        depth_filter: Whether to filter Gaussians behind camera
        min_depth: Minimum depth for valid projections

    Returns:
        u: Pixel x-coordinates (N,) - may be out of bounds
        v: Pixel y-coordinates (N,)
        valid: Boolean mask for valid projections (N,)
    """
    # Import here to avoid circular imports
    from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2

    N = means3D.shape[0]
    device = means3D.device

    # Get world-to-camera transform
    w2c = getWorld2View2(camera.R, camera.T)  # (4, 4)

    # Transform to camera space
    ones = torch.ones(N, 1, device=device)
    means_homo = torch.cat([means3D, ones], dim=1)  # (N, 4)
    means_cam = (w2c @ means_homo.T).T  # (N, 4)

    # Extract camera-space coordinates
    x_cam = means_cam[:, 0]
    y_cam = means_cam[:, 1]
    z_cam = means_cam[:, 2]

    # Check depth validity
    valid_depth = z_cam > min_depth

    # Project to image plane using pinhole model
    # u = fx * x/z + cx, v = fy * y/z + cy
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    # Avoid division by zero
    z_safe = torch.clamp(z_cam, min=min_depth)
    u = fx * x_cam / z_safe + cx
    v = fy * y_cam / z_safe + cy

    if depth_filter:
        valid = valid_depth
    else:
        valid = torch.ones(N, dtype=torch.bool, device=device)

    return u, v, valid


def project_mask_to_gaussians(
    mask: torch.Tensor,
    gaussians,
    camera,
    threshold: float = 0.5,
    depth_aware: bool = True,
    depth_buffer: float = 0.5,
) -> torch.Tensor:
    """
    Determine which Gaussians project into a 2D privacy mask.

    Uses forward projection: project each Gaussian center to 2D,
    then check if it falls within the mask region.

    Args:
        mask: Binary privacy mask (H, W) with values in [0, 1]
        gaussians: GaussianModel instance
        camera: Camera object with pose and intrinsics
        threshold: Mask threshold for considering a pixel private
        depth_aware: If True, also filter by depth consistency
        depth_buffer: Depth buffer in meters for depth-aware filtering

    Returns:
        private_mask: Boolean tensor (N,) where True = Gaussian is private
    """
    means3D = gaussians.get_xyz  # (N, 3)
    N = means3D.shape[0]
    H, W = mask.shape
    device = means3D.device

    # Project all Gaussians to image
    u, v, valid = project_gaussians_to_image(means3D, camera)

    # Check if projections are within image bounds
    in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Initialize private mask
    private_mask = torch.zeros(N, dtype=torch.bool, device=device)

    if not in_bounds.any():
        return private_mask

    # Get integer pixel coordinates for valid projections
    u_int = u[in_bounds].long().clamp(0, W - 1)
    v_int = v[in_bounds].long().clamp(0, H - 1)

    # Sample mask at projected locations
    # Note: mask is (H, W), so index as [v, u] (row, col)
    mask_vals = mask[v_int, u_int]
    is_in_private = mask_vals > threshold

    # Set private flag for Gaussians that project into mask
    private_mask[in_bounds] = is_in_private

    # Optional: depth-aware filtering
    if depth_aware and hasattr(camera, 'depth') and camera.depth is not None:
        # Get depths of private Gaussians from camera
        from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2
        w2c = getWorld2View2(camera.R, camera.T)
        means_homo = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=1)
        means_cam = (w2c @ means_homo.T).T
        gaussian_depths = means_cam[:, 2]

        # Get rendered/GT depth at private pixels
        depth_tensor = torch.from_numpy(camera.depth).to(device)
        private_pixel_depths = depth_tensor[v_int[is_in_private], u_int[is_in_private]]

        # Compute depth range with buffer
        if private_pixel_depths.numel() > 0:
            depth_min = private_pixel_depths.quantile(0.1).item() - depth_buffer
            depth_max = private_pixel_depths.quantile(0.9).item() + depth_buffer

            # Filter Gaussians that are within the depth range
            in_depth_range = (gaussian_depths >= depth_min) & (gaussian_depths <= depth_max)
            private_mask = private_mask & in_depth_range

    return private_mask


def compute_multi_view_privacy_mask(
    privacy_masks: dict,
    gaussians,
    cameras: dict,
    min_views: int = 1,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute privacy mask from multiple views using voting.

    A Gaussian is marked private if it appears in private regions
    in at least `min_views` keyframes.

    Args:
        privacy_masks: Dict mapping frame_idx -> privacy mask (H, W)
        gaussians: GaussianModel instance
        cameras: Dict mapping video_idx -> Camera object
        min_views: Minimum number of views for consensus
        threshold: Mask threshold for each view

    Returns:
        private_mask: Boolean tensor (N,) where True = Gaussian is private
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # Vote count per Gaussian
    vote_count = torch.zeros(N, device=device)

    for video_idx, camera in cameras.items():
        frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

        if frame_idx not in privacy_masks:
            continue

        mask = privacy_masks[frame_idx]
        if mask.device != device:
            mask = mask.to(device)

        # Get private Gaussians for this view
        view_private = project_mask_to_gaussians(
            mask, gaussians, camera, threshold=threshold
        )

        vote_count += view_private.float()

    # Apply voting threshold
    private_mask = vote_count >= min_views

    return private_mask


def get_boundary_gaussian_indices(
    private_mask: torch.Tensor,
    gaussians,
    cameras: dict,
    dilation_radius: int = 3,
) -> torch.Tensor:
    """
    Find Gaussians at the boundary of excised regions.

    These can be used as anchors for filling the excised region.

    Args:
        private_mask: Boolean tensor (N,) of private Gaussians
        gaussians: GaussianModel instance
        cameras: Dict of Camera objects
        dilation_radius: How far to look for boundary Gaussians

    Returns:
        boundary_indices: Indices of boundary Gaussians
    """
    N = private_mask.shape[0]
    device = private_mask.device

    # Get non-private Gaussians
    non_private_mask = ~private_mask

    # For each camera, find non-private Gaussians near private regions
    boundary_count = torch.zeros(N, device=device)

    for camera in cameras.values():
        # Create 2D mask of private Gaussian projections
        H, W = camera.image_height, camera.image_width
        private_projection = torch.zeros(H, W, device=device)

        means3D = gaussians.get_xyz
        u, v, valid = project_gaussians_to_image(means3D, camera)
        in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        private_in_view = private_mask & in_bounds

        if not private_in_view.any():
            continue

        # Mark private projections
        u_int = u[private_in_view].long().clamp(0, W - 1)
        v_int = v[private_in_view].long().clamp(0, H - 1)
        private_projection[v_int, u_int] = 1.0

        # Dilate private projection
        kernel_size = 2 * dilation_radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        dilated = F.conv2d(
            private_projection.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=dilation_radius
        ).squeeze() > 0

        # Find non-private Gaussians in dilated region
        non_private_in_view = non_private_mask & in_bounds
        if not non_private_in_view.any():
            continue

        u_np = u[non_private_in_view].long().clamp(0, W - 1)
        v_np = v[non_private_in_view].long().clamp(0, H - 1)

        # Check which non-private Gaussians are in boundary
        in_boundary = dilated[v_np, u_np]

        # Map back to full index
        non_private_indices = torch.where(non_private_in_view)[0]
        boundary_count[non_private_indices[in_boundary]] += 1

    # Return indices of Gaussians that appear in boundary of any view
    boundary_indices = torch.where(boundary_count > 0)[0]

    return boundary_indices


def resize_mask_to_features(
    mask: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Resize a 2D mask to match feature resolution (e.g., DINO features).

    Args:
        mask: Input mask (H, W)
        target_size: Target (H', W')
        mode: Interpolation mode

    Returns:
        Resized mask (H', W')
    """
    # Ensure 4D for interpolate
    mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(mask_4d, size=target_size, mode=mode, align_corners=False)
    return resized.squeeze()
