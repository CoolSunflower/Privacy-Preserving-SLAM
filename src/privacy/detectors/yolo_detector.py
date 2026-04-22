"""
YOLOv8+FastSAM-based privacy detector for real-time detection.

This detector provides fast (~25-50ms) privacy detection suitable for
simultaneous mode during SLAM. Uses YOLOv8-seg for segmentation masks.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, List

from .base_detector import BasePrivacyDetector, PrivacyMaskResult, PrivacyDetection


class YOLOPrivacyDetector(BasePrivacyDetector):
    """
    Fast YOLOv8-seg detector for real-time privacy detection.

    Uses COCO-pretrained YOLOv8 segmentation model.
    Detection time: ~25-50ms per frame depending on model size.
    Memory: ~0.5-1.5GB depending on model size.

    Supported privacy classes from COCO:
    - person (class 0)
    """

    # COCO class IDs for privacy-sensitive categories
    PRIVACY_CLASS_IDS = {
        0: "person",      # COCO class 0
        # Other COCO classes that might be privacy-sensitive:
        62: "tv",        # TV/monitor
        # 63: "laptop",
        # 64: "mouse",
        # 65: "remote",
        # 66: "keyboard",
        # 67: "cell phone",
    }

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize YOLO privacy detector.

        Args:
            config: Configuration dict with keys:
                - model: YOLOv8 model name ("yolov8n-seg", "yolov8s-seg", etc.)
                - confidence_threshold: Detection confidence threshold
                - iou_threshold: NMS IOU threshold
                - dilation_kernel: Mask dilation kernel size
                - privacy_classes: List of class names to detect (default: ["person"])
            device: Device to run inference on
        """
        super().__init__(config, device)

        self.model_name = config.get("model", "yolov8n-seg")
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.privacy_classes = config.get("privacy_classes", ["person"])

        # Build mapping from class names to IDs
        self.target_class_ids = {}
        for cls_id, cls_name in self.PRIVACY_CLASS_IDS.items():
            if cls_name in self.privacy_classes:
                self.target_class_ids[cls_id] = cls_name

        self.model = None

    def load_model(self) -> None:
        """Load YOLOv8 segmentation model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self._model_loaded = True

        print(f"[YOLOPrivacyDetector] Loaded {self.model_name} on {self.device}")

    def detect(self, image: torch.Tensor) -> PrivacyMaskResult:
        """
        Detect private regions using YOLOv8 segmentation.

        Args:
            image: RGB image tensor (3, H, W) or (H, W, 3), values in [0, 1]

        Returns:
            PrivacyMaskResult with combined mask and individual detections
        """
        self.ensure_loaded()

        start_time = time.time()

        # Preprocess image
        image, (H, W) = self.preprocess_image(image)

        # Convert to numpy for YOLO (expects HWC uint8)
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Run YOLO inference
        results = self.model(
            img_np,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )[0]

        # Initialize combined mask
        combined_mask = torch.zeros(H, W, device=self.device)
        detections = []

        # Process results
        if results.masks is not None and len(results.masks) > 0:
            for i, (mask_data, box, cls, conf) in enumerate(zip(
                results.masks.data,
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf
            )):
                cls_id = int(cls.item())

                # Skip if not a privacy class
                if cls_id not in self.target_class_ids:
                    continue

                # Resize mask to image size (convert uint8 to float32 for bilinear interpolation)
                mask_resized = F.interpolate(
                    mask_data.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                # Create detection object
                detection = PrivacyDetection(
                    mask=mask_resized.to(self.device),
                    confidence=conf.item(),
                    category=self.target_class_ids[cls_id],
                    bbox=tuple(box.cpu().numpy().astype(int)),
                    instance_id=i
                )
                detections.append(detection)

                # Update combined mask (union)
                combined_mask = torch.maximum(combined_mask, mask_resized.to(self.device))

        # Dilate combined mask for safety margin
        if combined_mask.sum() > 0:
            combined_mask = self.dilate_mask(combined_mask)

        processing_time = (time.time() - start_time) * 1000

        return PrivacyMaskResult(
            combined_mask=combined_mask,
            detections=detections,
            frame_idx=-1,  # Set by caller
            processing_time_ms=processing_time
        )

    def detect_batch(
        self,
        images: List[torch.Tensor],
        frame_indices: Optional[List[int]] = None,
    ) -> List[PrivacyMaskResult]:
        """
        Detect private regions in a batch of images.

        Args:
            images: List of RGB image tensors
            frame_indices: Optional frame indices for tracking

        Returns:
            List of PrivacyMaskResult objects
        """
        self.ensure_loaded()

        if frame_indices is None:
            frame_indices = list(range(len(images)))

        results = []
        for img, idx in zip(images, frame_indices):
            result = self.detect(img)
            result.frame_idx = idx
            results.append(result)

        return results


class YOLOWorldPrivacyDetector(BasePrivacyDetector):
    """
    Open-vocabulary YOLO-World detector for flexible privacy detection.

    YOLO-World supports text-based prompts for custom categories.
    Slightly slower than standard YOLOv8 but more flexible.
    """

    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Initialize YOLO-World privacy detector.

        Args:
            config: Configuration dict with keys:
                - model: YOLO-World model name
                - text_prompts: List of text prompts for detection
                - confidence_threshold: Detection confidence threshold
            device: Device to run inference on
        """
        super().__init__(config, device)

        self.model_name = config.get("model", "yolov8s-worldv2")
        self.text_prompts = config.get("text_prompts", [
            "person",
            "human face",
            "computer screen",
        ])

        self.model = None

    def load_model(self) -> None:
        """Load YOLO-World model with custom classes."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        # Set custom classes
        self.model.set_classes(self.text_prompts)
        self._model_loaded = True

        print(f"[YOLOWorldPrivacyDetector] Loaded {self.model_name}")
        print(f"[YOLOWorldPrivacyDetector] Custom classes: {self.text_prompts}")

    def detect(self, image: torch.Tensor) -> PrivacyMaskResult:
        """
        Detect private regions using YOLO-World with text prompts.

        Args:
            image: RGB image tensor (3, H, W) or (H, W, 3)

        Returns:
            PrivacyMaskResult with combined mask and detections
        """
        self.ensure_loaded()

        start_time = time.time()

        # Preprocess
        image, (H, W) = self.preprocess_image(image)
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Run inference
        results = self.model(
            img_np,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )[0]

        combined_mask = torch.zeros(H, W, device=self.device)
        detections = []

        # Process detections
        if results.boxes is not None and len(results.boxes) > 0:
            for i, (box, cls, conf) in enumerate(zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf
            )):
                cls_id = int(cls.item())
                # YOLO-World uses the custom class order
                category = self.text_prompts[cls_id] if cls_id < len(self.text_prompts) else "unknown"

                # Create box mask (YOLO-World doesn't have segmentation)
                box_coords = box.cpu().numpy().astype(int)
                x1, y1, x2, y2 = box_coords
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                mask = torch.zeros(H, W, device=self.device)
                mask[y1:y2, x1:x2] = 1.0

                detection = PrivacyDetection(
                    mask=mask,
                    confidence=conf.item(),
                    category=category,
                    bbox=(x1, y1, x2, y2),
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
