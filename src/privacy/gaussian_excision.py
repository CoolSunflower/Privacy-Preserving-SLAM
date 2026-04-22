"""
Gaussian Excision Manager for privacy-preserving SLAM.

Handles identification and removal of private Gaussians based on
2D privacy masks projected to 3D Gaussian space.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .mask_utils import (
    project_mask_to_gaussians,
    compute_multi_view_privacy_mask,
    get_boundary_gaussian_indices,
)


@dataclass
class ExcisionStats:
    """Statistics from a Gaussian excision operation."""
    num_total_gaussians: int
    num_private_gaussians: int
    num_excised: int
    num_views_processed: int
    excision_ratio: float  # num_excised / num_total

    def __str__(self):
        return (
            f"ExcisionStats(total={self.num_total_gaussians}, "
            f"private={self.num_private_gaussians}, "
            f"excised={self.num_excised}, "
            f"ratio={self.excision_ratio:.2%})"
        )


class GaussianExcisionManager:
    """
    Manages identification and removal of private Gaussians.

    Uses forward projection to identify which Gaussians project into
    privacy mask regions, then removes them from the model.
    """

    def __init__(self, config: dict):
        """
        Initialize the excision manager.

        Args:
            config: Configuration dict with keys:
                - multi_view_threshold: Min views for consensus (default: 2)
                - depth_aware: Use depth filtering (default: True)
                - depth_buffer: Depth tolerance in meters (default: 0.5)
                - excision_threshold: Mask threshold (default: 0.5)
        """
        self.config = config
        self.multi_view_threshold = config.get("multi_view_threshold", 2)
        self.depth_aware = config.get("depth_aware", True)
        self.depth_buffer = config.get("depth_buffer", 0.5)
        self.excision_threshold = config.get("excision_threshold", 0.5)

        # Track which Gaussians have been marked private
        self._private_flags: Optional[torch.Tensor] = None
        self._excision_history: List[ExcisionStats] = []

    def identify_private_gaussians(
        self,
        gaussians,
        privacy_masks: Dict[int, torch.Tensor],
        cameras: Dict,
        use_consensus: bool = True,
    ) -> torch.Tensor:
        """
        Identify Gaussians that should be excised.

        Args:
            gaussians: GaussianModel instance
            privacy_masks: Dict mapping frame_idx -> privacy mask (H, W)
            cameras: Dict mapping video_idx -> Camera object
            use_consensus: If True, require multi-view consensus

        Returns:
            Boolean mask (N,) where True = Gaussian should be excised
        """
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device

        if use_consensus and self.multi_view_threshold > 1:
            # Use multi-view voting
            private_mask = compute_multi_view_privacy_mask(
                privacy_masks=privacy_masks,
                gaussians=gaussians,
                cameras=cameras,
                min_views=self.multi_view_threshold,
                threshold=self.excision_threshold,
            )
        else:
            # Use single-view detection (union over all views)
            private_mask = torch.zeros(N, dtype=torch.bool, device=device)

            for video_idx, camera in cameras.items():
                frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx

                if frame_idx not in privacy_masks:
                    continue

                mask = privacy_masks[frame_idx]
                if mask.device != device:
                    mask = mask.to(device)

                view_private = project_mask_to_gaussians(
                    mask=mask,
                    gaussians=gaussians,
                    camera=camera,
                    threshold=self.excision_threshold,
                    depth_aware=self.depth_aware,
                    depth_buffer=self.depth_buffer,
                )

                # Union: mark as private if seen in ANY view
                private_mask = private_mask | view_private

        return private_mask

    def excise_gaussians(
        self,
        gaussians,
        private_mask: torch.Tensor,
    ) -> ExcisionStats:
        """
        Remove private Gaussians from the model.

        Args:
            gaussians: GaussianModel instance (modified in place)
            private_mask: Boolean mask (N,) where True = prune

        Returns:
            ExcisionStats with operation statistics
        """
        num_total = gaussians.get_xyz.shape[0]
        num_private = private_mask.sum().item()

        stats = ExcisionStats(
            num_total_gaussians=num_total,
            num_private_gaussians=num_private,
            num_excised=0,
            num_views_processed=0,
            excision_ratio=0.0
        )

        if num_private == 0:
            return stats

        # Prune the private Gaussians
        gaussians.prune_points(private_mask)

        stats.num_excised = num_private
        stats.excision_ratio = num_private / num_total if num_total > 0 else 0.0

        self._excision_history.append(stats)

        return stats

    def identify_and_excise(
        self,
        gaussians,
        privacy_masks: Dict[int, torch.Tensor],
        cameras: Dict,
    ) -> Tuple[torch.Tensor, ExcisionStats]:
        """
        Combined identification and excision in one call.

        Args:
            gaussians: GaussianModel instance
            privacy_masks: Privacy masks per frame
            cameras: Camera objects per video_idx

        Returns:
            Tuple of (boundary_indices, stats)
        """
        # Identify private Gaussians
        private_mask = self.identify_private_gaussians(
            gaussians=gaussians,
            privacy_masks=privacy_masks,
            cameras=cameras,
            use_consensus=True,
        )

        # Get boundary Gaussians before excision (for filling)
        if private_mask.any():
            boundary_indices = get_boundary_gaussian_indices(
                private_mask=private_mask,
                gaussians=gaussians,
                cameras=cameras,
            )
        else:
            boundary_indices = torch.tensor([], dtype=torch.long)

        # Excise
        stats = self.excise_gaussians(gaussians, private_mask)
        stats.num_views_processed = len(cameras)

        return boundary_indices, stats

    def tag_gaussians_for_deferred_excision(
        self,
        gaussians,
        privacy_mask: torch.Tensor,
    ) -> None:
        """
        Tag Gaussians as private without immediately excising.

        Useful for simultaneous mode where we want to track private
        Gaussians but defer actual pruning to post-processing.

        Args:
            gaussians: GaussianModel instance
            privacy_mask: Boolean mask (N,) of private Gaussians
        """
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device

        if self._private_flags is None or len(self._private_flags) != N:
            self._private_flags = torch.zeros(N, dtype=torch.bool, device=device)

        # Union: once tagged, stays tagged
        self._private_flags = self._private_flags | privacy_mask

    def get_tagged_private_mask(self) -> Optional[torch.Tensor]:
        """Get the current tagged private Gaussians mask."""
        return self._private_flags

    def excise_tagged_gaussians(self, gaussians) -> ExcisionStats:
        """
        Excise all previously tagged private Gaussians.

        Called at the end of SLAM session to perform deferred excision.
        """
        if self._private_flags is None:
            return ExcisionStats(
                num_total_gaussians=gaussians.get_xyz.shape[0],
                num_private_gaussians=0,
                num_excised=0,
                num_views_processed=0,
                excision_ratio=0.0
            )

        # Ensure mask matches current Gaussian count
        N = gaussians.get_xyz.shape[0]
        if len(self._private_flags) != N:
            # Model has changed size - flag mismatch
            print(f"[WARN] Private flag size mismatch: {len(self._private_flags)} vs {N}")
            self._private_flags = None
            return ExcisionStats(
                num_total_gaussians=N,
                num_private_gaussians=0,
                num_excised=0,
                num_views_processed=0,
                excision_ratio=0.0
            )

        stats = self.excise_gaussians(gaussians, self._private_flags)
        self._private_flags = None  # Reset after excision

        return stats

    def update_tags_after_densification(self, new_count: int, added_count: int) -> None:
        """
        Update private flags after Gaussian densification.

        When new Gaussians are added, they inherit False (non-private) status.
        This should be called after densify_and_prune or extend_from_pcd.

        Args:
            new_count: New total number of Gaussians
            added_count: Number of Gaussians added
        """
        if self._private_flags is None:
            return

        device = self._private_flags.device

        # Pad with False for new Gaussians
        new_flags = torch.zeros(added_count, dtype=torch.bool, device=device)
        self._private_flags = torch.cat([self._private_flags, new_flags])

        # Verify size
        if len(self._private_flags) != new_count:
            print(f"[WARN] Flag update mismatch: {len(self._private_flags)} vs {new_count}")
            self._private_flags = None

    @property
    def excision_history(self) -> List[ExcisionStats]:
        """Get history of excision operations."""
        return self._excision_history.copy()

    def get_total_excised(self) -> int:
        """Get total number of Gaussians excised across all operations."""
        return sum(s.num_excised for s in self._excision_history)
