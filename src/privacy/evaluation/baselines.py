"""
Baseline methods for privacy-preserving SLAM comparison.

Implements comparative methods for ablation studies:
- A: No Privacy (baseline)
- B: Input Blurring
- C: Uncertainty-Only
- D: Post-Process Only
- E: Excision-Only (no filling)
- F: Ours Full (hybrid + filling)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Configuration for a baseline method."""
    name: str
    description: str
    privacy_enabled: bool
    mode: str  # "none", "simultaneous", "postprocess", "hybrid", "input_blur"
    enable_excision: bool
    enable_filling: bool
    blur_radius: int = 21

    def to_dict(self) -> dict:
        return {
            "privacy": {
                "enable": self.privacy_enabled,
                "mode": self.mode,
                "enable_excision": self.enable_excision,
                "enable_filling": self.enable_filling,
            }
        }


class BaselineRunner:
    """
    Manages running all comparison methods as full SLAM pipelines.

    All methods produce comparable outputs for fair evaluation.
    """

    # Define all baseline configurations
    BASELINES = {
        "A_no_privacy": BaselineConfig(
            name="A: No Privacy",
            description="Baseline - no privacy processing. Upper bound for quality.",
            privacy_enabled=False,
            mode="none",
            enable_excision=False,
            enable_filling=False,
        ),
        "B_input_blur": BaselineConfig(
            name="B: Input Blurring",
            description="Blur private regions in input images before SLAM. Expected to degrade tracking.",
            privacy_enabled=True,
            mode="input_blur",
            enable_excision=False,
            enable_filling=False,
            blur_radius=21,
        ),
        "C_uncertainty_only": BaselineConfig(
            name="C: Uncertainty-Only",
            description="Inject high uncertainty for private regions, no Gaussian pruning.",
            privacy_enabled=True,
            mode="simultaneous",
            enable_excision=False,
            enable_filling=False,
        ),
        "D_postprocess_only": BaselineConfig(
            name="D: Post-Process Only",
            description="No runtime processing, excision only after SLAM completes.",
            privacy_enabled=True,
            mode="postprocess",
            enable_excision=True,
            enable_filling=False,
        ),
        "E_excision_only": BaselineConfig(
            name="E: Excision-Only",
            description="Full excision (simultaneous + post-process) but no filling.",
            privacy_enabled=True,
            mode="hybrid",
            enable_excision=True,
            enable_filling=False,
        ),
        "F_ours_full": BaselineConfig(
            name="F: Ours Full",
            description="Complete pipeline: simultaneous + post-process + filling.",
            privacy_enabled=True,
            mode="hybrid",
            enable_excision=True,
            enable_filling=True,
        ),
    }

    @classmethod
    def get_all_configs(cls) -> Dict[str, BaselineConfig]:
        """Get all baseline configurations."""
        return cls.BASELINES.copy()

    @classmethod
    def get_config(cls, method_id: str) -> BaselineConfig:
        """Get configuration for a specific method."""
        if method_id not in cls.BASELINES:
            raise ValueError(f"Unknown baseline: {method_id}. "
                           f"Available: {list(cls.BASELINES.keys())}")
        return cls.BASELINES[method_id]

    @classmethod
    def create_slam_config(cls, base_config: dict, method_id: str) -> dict:
        """
        Create a full SLAM configuration with baseline settings.

        Args:
            base_config: Base SLAM configuration
            method_id: Baseline method ID (e.g., "A_no_privacy")

        Returns:
            Modified configuration dict
        """
        import copy
        config = copy.deepcopy(base_config)
        baseline = cls.get_config(method_id)

        # Inject baseline privacy settings
        config["privacy"] = baseline.to_dict()["privacy"]

        # Add method metadata
        config["privacy"]["method_id"] = method_id
        config["privacy"]["method_name"] = baseline.name

        return config


class InputBlurringMode:
    """
    Method B: Blur private regions BEFORE they enter the SLAM pipeline.

    This is expected to degrade tracking significantly because:
    1. DROID-SLAM correlation volume relies on sharp features
    2. Blurring removes texture information needed for tracking

    This baseline demonstrates why naive input modification doesn't work.
    """

    def __init__(
        self,
        detector,
        blur_radius: int = 21,
        blur_sigma: float = 0.0,  # 0 = auto
    ):
        """
        Initialize input blurring mode.

        Args:
            detector: Privacy detector for identifying regions to blur
            blur_radius: Gaussian blur kernel size (must be odd)
            blur_sigma: Blur sigma (0 = compute from radius)
        """
        self.detector = detector
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.blur_sigma = blur_sigma

    def process_frame(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process a frame by blurring private regions.

        Args:
            image: RGB image (3, H, W) with values in [0, 1]

        Returns:
            Processed image with blurred private regions
        """
        # Detect private regions
        result = self.detector.detect(image)

        if not result.has_detections:
            return image

        mask = result.combined_mask

        # Convert to numpy for OpenCV
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            img_np,
            (self.blur_radius, self.blur_radius),
            self.blur_sigma
        )

        # Composite: use blurred for private regions, original elsewhere
        mask_np = mask.cpu().numpy()
        mask_3d = np.stack([mask_np] * 3, axis=-1)

        result_np = np.where(mask_3d > 0.5, blurred, img_np)

        # Convert back to tensor
        result_tensor = torch.from_numpy(result_np).permute(2, 0, 1).float() / 255.0
        return result_tensor.to(image.device)


class UncertaintyOnlyMode:
    """
    Method C: Only inject uncertainty, no Gaussian pruning.

    This demonstrates the limitation of uncertainty-based approaches
    without actual 3D excision:
    - Gaussians created before detection remain
    - Re-observation from new viewpoints can still optimize private content
    """

    def __init__(self, uncertainty_beta: float = 100.0):
        """
        Initialize uncertainty-only mode.

        Args:
            uncertainty_beta: High uncertainty value to inject
        """
        self.beta = uncertainty_beta

    def inject_uncertainty(
        self,
        uncertainty: torch.Tensor,
        privacy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject high uncertainty into private regions.

        Args:
            uncertainty: Current uncertainty map (H', W')
            privacy_mask: Privacy mask (H, W)

        Returns:
            Modified uncertainty map
        """
        # Resize mask to uncertainty resolution
        mask_resized = F.interpolate(
            privacy_mask.unsqueeze(0).unsqueeze(0),
            size=uncertainty.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Inject high uncertainty
        modified = torch.where(
            mask_resized > 0.5,
            torch.full_like(uncertainty, self.beta),
            uncertainty
        )

        return modified


class ComparisonEvaluator:
    """
    Runs evaluation across all baseline methods and produces comparison tables.
    """

    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize comparison evaluator.

        Args:
            methods: List of method IDs to evaluate. None = all methods.
        """
        if methods is None:
            methods = list(BaselineRunner.BASELINES.keys())

        self.methods = methods
        self.results: Dict[str, dict] = {}

    def add_result(self, method_id: str, metrics: dict) -> None:
        """Add evaluation results for a method."""
        self.results[method_id] = metrics

    def get_comparison_table(self) -> str:
        """
        Generate a comparison table in markdown format.

        Returns:
            Markdown table string
        """
        if not self.results:
            return "No results available."

        # Header
        headers = ["Method", "ATE (m)", "PSNR", "SSIM", "Re-ID", "Excision %", "Runtime (ms)"]
        rows = [headers]

        # Data rows
        for method_id in self.methods:
            if method_id not in self.results:
                continue

            m = self.results[method_id]
            config = BaselineRunner.get_config(method_id)

            row = [
                config.name,
                f"{m.get('ate_rmse', 0):.4f}",
                f"{m.get('psnr_non_private', 0):.2f}",
                f"{m.get('ssim_non_private', 0):.4f}",
                f"{m.get('reid_score', 0):.2f}",
                f"{m.get('excision_completeness', 0)*100:.1f}%",
                f"{m.get('total_runtime_ms', 0):.0f}",
            ]
            rows.append(row)

        # Format as markdown table
        col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

        lines = []
        for i, row in enumerate(rows):
            line = "| " + " | ".join(
                str(cell).ljust(col_widths[j]) for j, cell in enumerate(row)
            ) + " |"
            lines.append(line)

            if i == 0:  # Header separator
                sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
                lines.append(sep)

        return "\n".join(lines)

    def save_csv(self, path: str) -> None:
        """Save results to CSV file."""
        import csv

        if not self.results:
            return

        # Collect all unique keys
        all_keys = set()
        for m in self.results.values():
            all_keys.update(m.keys())

        fieldnames = ["method_id", "method_name"] + sorted(all_keys)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for method_id in self.methods:
                if method_id not in self.results:
                    continue

                config = BaselineRunner.get_config(method_id)
                row = {
                    "method_id": method_id,
                    "method_name": config.name,
                    **self.results[method_id]
                }
                writer.writerow(row)

        print(f"[ComparisonEvaluator] Results saved to {path}")

    def print_summary(self) -> None:
        """Print summary table to console."""
        print("\n" + "=" * 80)
        print("Privacy-Preserving SLAM Comparison Results")
        print("=" * 80 + "\n")
        print(self.get_comparison_table())
        print("\n" + "=" * 80)
