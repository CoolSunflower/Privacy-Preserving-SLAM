"""
Region Filler for privacy-preserving Gaussian SLAM.

Fills excised regions with neutral placeholder Gaussians using
depth inpainting and boundary interpolation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FillStats:
    """Statistics from a fill operation."""
    num_fill_points: int
    fill_region_size_pixels: int
    avg_fill_depth: float
    fill_time_ms: float

    def __str__(self):
        return (
            f"FillStats(points={self.num_fill_points}, "
            f"region_size={self.fill_region_size_pixels}, "
            f"avg_depth={self.avg_fill_depth:.2f}m)"
        )


class RegionFiller:
    """
    Fills excised regions with neutral placeholder Gaussians.

    Strategy:
    1. Inpaint depth using Navier-Stokes method
    2. Sample sparse points in the excised region
    3. Create low-opacity, neutral-color Gaussians
    4. Mark with special keyframe ID for separate handling
    """

    def __init__(self, config: dict):
        """
        Initialize the region filler.

        Args:
            config: Configuration dict with keys:
                - fill_opacity: Opacity for filler Gaussians (default: 0.15)
                - fill_color: RGB color [0,1] (default: [0.5, 0.5, 0.5] gray)
                - fill_density: Points per pixel (default: 0.02)
                - fill_point_size: Gaussian scale (default: 0.05)
                - inpaint_radius: OpenCV inpaint radius (default: 5)
                - fill_keyframe_id: Keyframe ID for fillers (default: -1)
        """
        self.config = config

        self.fill_opacity = config.get("fill_opacity", 0.15)
        self.fill_color = config.get("fill_color", [0.5, 0.5, 0.5])
        self.fill_density = config.get("fill_density", 0.02)
        self.point_size = config.get("fill_point_size", 0.05)
        self.inpaint_radius = config.get("inpaint_radius", 5)
        self.fill_keyframe_id = config.get("fill_keyframe_id", -1)

        self._fill_history = []

    def inpaint_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Inpaint depth map in masked regions.

        Args:
            depth: Depth map (H, W) as numpy array
            mask: Binary mask (H, W) where 1 = inpaint

        Returns:
            Inpainted depth map
        """
        # Ensure mask is uint8
        mask_uint8 = (mask > 0.5).astype(np.uint8)

        # Use Navier-Stokes inpainting for smooth results
        depth_inpainted = cv2.inpaint(
            depth.astype(np.float32),
            mask_uint8,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_NS
        )

        return depth_inpainted

    def sample_fill_points(
        self,
        mask: torch.Tensor,
        max_points: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample sparse points within the fill region.

        Args:
            mask: Binary mask (H, W) where 1 = fill region
            max_points: Maximum number of points to sample

        Returns:
            Pixel coordinates (N, 2) as [v, u] format
        """
        device = mask.device

        # Get all pixels in fill region
        fill_pixels = torch.nonzero(mask > 0.5)  # (N, 2) as [v, u]

        if len(fill_pixels) == 0:
            return torch.empty(0, 2, dtype=torch.long, device=device)

        # Calculate number of points to sample
        num_fill_pixels = len(fill_pixels)
        num_samples = int(num_fill_pixels * self.fill_density)

        if max_points is not None:
            num_samples = min(num_samples, max_points)

        num_samples = max(1, num_samples)  # At least 1 point

        # Random sampling
        if num_samples >= num_fill_pixels:
            return fill_pixels
        else:
            indices = torch.randperm(num_fill_pixels, device=device)[:num_samples]
            return fill_pixels[indices]

    def unproject_to_3d(
        self,
        pixel_coords: torch.Tensor,  # (N, 2) as [v, u]
        depth_map: np.ndarray,
        camera,
    ) -> np.ndarray:
        """
        Unproject 2D pixels to 3D world coordinates.

        Args:
            pixel_coords: Pixel coordinates (N, 2) as [v, u]
            depth_map: Depth map (H, W)
            camera: Camera object with intrinsics and pose

        Returns:
            3D points (N, 3) in world coordinates
        """
        if len(pixel_coords) == 0:
            return np.empty((0, 3))

        # Get pixel coordinates
        v_coords = pixel_coords[:, 0].cpu().numpy()
        u_coords = pixel_coords[:, 1].cpu().numpy()

        # Sample depths
        depths = depth_map[v_coords, u_coords]

        # Unproject to camera space
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy

        x_cam = (u_coords - cx) * depths / fx
        y_cam = (v_coords - cy) * depths / fy
        z_cam = depths

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to world coordinates
        from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2

        w2c = getWorld2View2(camera.R, camera.T).cpu().numpy()
        c2w = np.linalg.inv(w2c)

        # Add homogeneous coordinate
        ones = np.ones((points_cam.shape[0], 1))
        points_homo = np.concatenate([points_cam, ones], axis=1)

        # Transform
        points_world = (c2w @ points_homo.T).T[:, :3]

        return points_world

    def create_filler_gaussians(
        self,
        points_world: np.ndarray,
        gaussians,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create Gaussian parameters for fill points.

        Args:
            points_world: 3D points (N, 3)
            gaussians: GaussianModel for referencing SH degree etc.

        Returns:
            Tuple of (xyz, features, scales, rotations, opacities)
        """
        from thirdparty.gaussian_splatting.utils.sh_utils import RGB2SH
        from thirdparty.gaussian_splatting.utils.general_utils import inverse_sigmoid

        N = len(points_world)
        device = gaussians.get_xyz.device

        # Positions
        fused_point_cloud = torch.from_numpy(points_world).float().to(device)

        # Colors in SH space
        fill_rgb = torch.tensor(self.fill_color, device=device).float()
        fused_color = RGB2SH(fill_rgb.unsqueeze(0).expand(N, -1))

        # Features (SH coefficients)
        sh_degree = gaussians.max_sh_degree
        features = torch.zeros(
            (N, 3, (sh_degree + 1) ** 2),
            device=device
        ).float()
        features[:, :3, 0] = fused_color

        # Uniform scale
        if gaussians.isotropic:
            scales = torch.log(torch.full((N, 1), self.point_size, device=device))
        else:
            scales = torch.log(torch.full((N, 3), self.point_size, device=device))

        # Identity rotation
        rotations = torch.zeros((N, 4), device=device)
        rotations[:, 0] = 1.0  # w=1, x=y=z=0

        # Low opacity
        opacities = inverse_sigmoid(
            torch.full((N, 1), self.fill_opacity, device=device)
        )

        return fused_point_cloud, features, scales, rotations, opacities

    def fill_region(
        self,
        gaussians,
        privacy_mask: torch.Tensor,
        camera,
        keyframe_id: Optional[int] = None,
    ) -> int:
        """
        Fill excised region with neutral Gaussians.

        Args:
            gaussians: GaussianModel to add filler Gaussians to
            privacy_mask: Binary mask (H, W) of region to fill
            camera: Camera object with depth and pose
            keyframe_id: ID to assign to filler Gaussians

        Returns:
            Number of Gaussians added
        """
        import time
        start_time = time.time()

        if keyframe_id is None:
            keyframe_id = self.fill_keyframe_id

        # Get depth map
        depth = camera.depth
        if depth is None:
            print("[RegionFiller] Warning: No depth available, skipping fill")
            return 0

        # Convert mask to numpy for OpenCV
        mask_np = privacy_mask.cpu().numpy()

        # Inpaint depth
        depth_inpainted = self.inpaint_depth(depth, mask_np)

        # Sample fill points
        fill_pixels = self.sample_fill_points(privacy_mask)

        if len(fill_pixels) == 0:
            return 0

        # Unproject to 3D
        points_world = self.unproject_to_3d(fill_pixels, depth_inpainted, camera)

        # Filter invalid points
        valid_mask = ~np.any(np.isnan(points_world) | np.isinf(points_world), axis=1)
        points_world = points_world[valid_mask]

        if len(points_world) == 0:
            return 0

        # Create Gaussian parameters
        xyz, features, scales, rots, opacities = self.create_filler_gaussians(
            points_world, gaussians
        )

        # Add to Gaussian model
        gaussians.extend_from_pcd(
            xyz,
            features,
            scales,
            rots,
            opacities,
            keyframe_id
        )

        fill_time = (time.time() - start_time) * 1000

        # Record stats
        stats = FillStats(
            num_fill_points=len(points_world),
            fill_region_size_pixels=int(mask_np.sum()),
            avg_fill_depth=float(depth_inpainted[mask_np > 0.5].mean()) if mask_np.sum() > 0 else 0,
            fill_time_ms=fill_time
        )
        self._fill_history.append(stats)

        return len(points_world)

    def fill_from_boundary(
        self,
        gaussians,
        privacy_mask: torch.Tensor,
        boundary_indices: torch.Tensor,
        camera,
    ) -> int:
        """
        Fill region by interpolating from boundary Gaussians.

        Alternative to depth-based filling that uses nearby Gaussians
        for better color consistency.

        Args:
            gaussians: GaussianModel
            privacy_mask: Binary mask of fill region
            boundary_indices: Indices of boundary Gaussians
            camera: Camera object

        Returns:
            Number of Gaussians added
        """
        if len(boundary_indices) == 0:
            # Fall back to standard fill
            return self.fill_region(gaussians, privacy_mask, camera)

        # Get boundary Gaussian properties
        boundary_xyz = gaussians.get_xyz[boundary_indices]
        boundary_colors = gaussians._features_dc[boundary_indices]

        # Sample fill points
        fill_pixels = self.sample_fill_points(privacy_mask)

        if len(fill_pixels) == 0:
            return 0

        # Compute average properties from boundary
        avg_color = boundary_colors.mean(dim=0)
        avg_scale = gaussians._scaling[boundary_indices].mean(dim=0)

        # Get depth and unproject
        depth = camera.depth
        if depth is None:
            return 0

        depth_inpainted = self.inpaint_depth(depth, privacy_mask.cpu().numpy())
        points_world = self.unproject_to_3d(fill_pixels, depth_inpainted, camera)

        valid_mask = ~np.any(np.isnan(points_world) | np.isinf(points_world), axis=1)
        points_world = points_world[valid_mask]

        if len(points_world) == 0:
            return 0

        N = len(points_world)
        device = gaussians.get_xyz.device

        # Use averaged boundary properties
        from thirdparty.gaussian_splatting.utils.general_utils import inverse_sigmoid

        xyz = torch.from_numpy(points_world).float().to(device)
        features = avg_color.unsqueeze(0).expand(N, -1, -1).clone()
        scales = avg_scale.unsqueeze(0).expand(N, -1).clone()
        rots = torch.zeros((N, 4), device=device)
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(
            torch.full((N, 1), self.fill_opacity, device=device)
        )

        gaussians.extend_from_pcd(
            xyz, features, scales, rots, opacities,
            self.fill_keyframe_id
        )

        return N

    @property
    def fill_history(self):
        """Get history of fill operations."""
        return self._fill_history.copy()

    def get_total_filled(self) -> int:
        """Get total number of fill Gaussians created."""
        return sum(s.num_fill_points for s in self._fill_history)
