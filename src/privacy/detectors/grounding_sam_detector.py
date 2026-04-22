"""
Grounding DINO + SAM detector for open-vocabulary privacy detection.

This detector provides high-quality, pixel-perfect masks for privacy regions
using text-based prompts. Suitable for post-processing mode.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, List

from .base_detector import BasePrivacyDetector, PrivacyMaskResult, PrivacyDetection


class GroundingSAMDetector(BasePrivacyDetector):
    """
    Open-vocabulary detection with pixel-perfect masks using Grounding DINO + SAM.

    Detection time: ~150-200ms per frame
    Memory: ~2.5-3GB (can be reduced by using SAM-ViT-B)

    Features:
    - Text-based prompts for flexible category detection
    - Pixel-perfect masks via Segment Anything Model
    - Best quality for post-processing mode
    """

    DEFAULT_PROMPTS = [
        "human face",
        "person",
        "computer screen",
        "monitor display",
        "document",
        "paper with text",
        "license plate",
        "credit card",
        "id card",
        "whiteboard with writing",
    ]

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize Grounding DINO + SAM detector.

        Args:
            config: Configuration dict with keys:
                - text_prompts: List of text prompts for detection
                - box_threshold: Detection box confidence threshold
                - text_threshold: Text-image matching threshold
                - sam_model: SAM model variant ("vit_h", "vit_l", "vit_b")
                - grounding_dino_config: Path to Grounding DINO config
                - grounding_dino_weights: Path to Grounding DINO weights
                - sam_checkpoint: Path to SAM checkpoint
            device: Device to run inference on
        """
        super().__init__(config, device)

        self.text_prompts = config.get("text_prompts", self.DEFAULT_PROMPTS)
        self.box_threshold = config.get("box_threshold", 0.3)
        self.text_threshold = config.get("text_threshold", 0.25)
        self.sam_model_type = config.get("sam_model", "vit_h")

        # Paths to model weights (user needs to provide these)
        self.gdino_config = config.get(
            "grounding_dino_config",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        self.gdino_weights = config.get(
            "grounding_dino_weights",
            "weights/groundingdino_swint_ogc.pth"
        )
        self.sam_checkpoint = config.get(
            "sam_checkpoint",
            f"weights/sam_vit_h_4b8939.pth"
        )

        self.grounding_dino = None
        self.sam_predictor = None

    def load_model(self) -> None:
        """Load Grounding DINO and SAM models."""
        import warnings
        warnings.filterwarnings("ignore")

        # Try to import Grounding DINO
        try:
            from groundingdino.util.inference import load_model as load_gdino_model
        except ImportError:
            raise ImportError(
                "GroundingDINO not found. Install from: "
                "https://github.com/IDEA-Research/GroundingDINO"
            )

        # Try to import SAM
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not found. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        # Load Grounding DINO
        print(f"[GroundingSAMDetector] Loading Grounding DINO from {self.gdino_weights}")
        self.grounding_dino = load_gdino_model(self.gdino_config, self.gdino_weights)
        self.grounding_dino.to(self.device)
        self.grounding_dino.eval()

        # Load SAM
        print(f"[GroundingSAMDetector] Loading SAM {self.sam_model_type} from {self.sam_checkpoint}")
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

        self._model_loaded = True
        print(f"[GroundingSAMDetector] Models loaded on {self.device}")
        print(f"[GroundingSAMDetector] Text prompts: {self.text_prompts}")

    def detect(self, image: torch.Tensor) -> PrivacyMaskResult:
        """
        Detect private regions using Grounding DINO + SAM.

        Args:
            image: RGB image tensor (3, H, W) or (H, W, 3), values in [0, 1]

        Returns:
            PrivacyMaskResult with combined mask and individual detections
        """
        self.ensure_loaded()

        start_time = time.time()

        from groundingdino.util.inference import predict as gdino_predict

        # Preprocess image
        image, (H, W) = self.preprocess_image(image)

        # Convert for Grounding DINO (expects CHW tensor [0,1])
        img_for_gdino = image.cpu()

        # Convert for SAM (expects HWC uint8)
        img_for_sam = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Set SAM image
        self.sam_predictor.set_image(img_for_sam)

        # Build caption from prompts
        caption = " . ".join(self.text_prompts)

        # Run Grounding DINO
        boxes, logits, phrases = gdino_predict(
            model=self.grounding_dino,
            image=img_for_gdino,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )

        # Initialize outputs
        combined_mask = torch.zeros(H, W, device=self.device)
        detections = []

        if len(boxes) > 0:
            # Denormalize boxes to absolute coordinates
            boxes_abs = boxes.clone()
            boxes_abs[:, [0, 2]] *= W  # x coords
            boxes_abs[:, [1, 3]] *= H  # y coords

            for i, (box, logit, phrase) in enumerate(zip(boxes_abs, logits, phrases)):
                # Get SAM mask for this box
                box_np = box.cpu().numpy()
                masks, scores, _ = self.sam_predictor.predict(
                    box=box_np,
                    multimask_output=True
                )

                # Use highest scoring mask
                best_mask_idx = scores.argmax()
                mask = torch.from_numpy(masks[best_mask_idx]).float().to(self.device)

                # Create detection object
                detection = PrivacyDetection(
                    mask=mask,
                    confidence=logit.item(),
                    category=phrase,
                    bbox=tuple(box_np.astype(int)),
                    instance_id=i
                )
                detections.append(detection)

                # Update combined mask
                combined_mask = torch.maximum(combined_mask, mask)

        # Dilate for safety margin
        if combined_mask.sum() > 0:
            combined_mask = self.dilate_mask(combined_mask)

        processing_time = (time.time() - start_time) * 1000

        return PrivacyMaskResult(
            combined_mask=combined_mask,
            detections=detections,
            frame_idx=-1,
            processing_time_ms=processing_time
        )

    def detect_with_boxes(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        categories: Optional[List[str]] = None,
    ) -> PrivacyMaskResult:
        """
        Get SAM masks for pre-defined boxes (e.g., from YOLO).

        Args:
            image: RGB image tensor
            boxes: Pre-detected boxes (N, 4) in xyxy format
            categories: Optional category names for each box

        Returns:
            PrivacyMaskResult with SAM-refined masks
        """
        self.ensure_loaded()

        start_time = time.time()

        # Preprocess
        image, (H, W) = self.preprocess_image(image)
        img_for_sam = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        self.sam_predictor.set_image(img_for_sam)

        if categories is None:
            categories = ["unknown"] * len(boxes)

        combined_mask = torch.zeros(H, W, device=self.device)
        detections = []

        for i, (box, cat) in enumerate(zip(boxes, categories)):
            box_np = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
            masks, scores, _ = self.sam_predictor.predict(
                box=box_np,
                multimask_output=True
            )

            best_mask_idx = scores.argmax()
            mask = torch.from_numpy(masks[best_mask_idx]).float().to(self.device)

            detection = PrivacyDetection(
                mask=mask,
                confidence=scores[best_mask_idx],
                category=cat,
                bbox=tuple(box_np.astype(int)),
                instance_id=i
            )
            detections.append(detection)
            combined_mask = torch.maximum(combined_mask, mask)

        if combined_mask.sum() > 0:
            combined_mask = self.dilate_mask(combined_mask)

        processing_time = (time.time() - start_time) * 1000

        return PrivacyMaskResult(
            combined_mask=combined_mask,
            detections=detections,
            frame_idx=-1,
            processing_time_ms=processing_time
        )


class GroundedSAM2Detector(BasePrivacyDetector):
    """
    Grounded SAM2 detector - integrated pipeline for video.

    SAM2 provides better temporal consistency for video sequences.
    This is the preferred option for 2024+ implementations.
    """

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize Grounded SAM2 detector.

        Args:
            config: Configuration with model paths and prompts
            device: Device for inference
        """
        super().__init__(config, device)
        self.text_prompts = config.get("text_prompts", GroundingSAMDetector.DEFAULT_PROMPTS)
        self.box_threshold = config.get("box_threshold", 0.3)

        self.model = None
        self._implementation_note = (
            "SAM2 integration requires huggingface transformers >= 4.40 "
            "and the grounded-sam-2 package. This is a placeholder."
        )

    def load_model(self) -> None:
        """Load Grounded SAM2 model."""
        # Placeholder for SAM2 integration
        # When SAM2 becomes widely available, implement here
        raise NotImplementedError(self._implementation_note)

    def detect(self, image: torch.Tensor) -> PrivacyMaskResult:
        """Detect private regions - placeholder for SAM2."""
        raise NotImplementedError(self._implementation_note)
