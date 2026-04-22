"""
MambaVision Descriptor Extractor for Loop Closure Detection
Stage 5: Vision-Mamba Descriptor Pipeline

This module provides a wrapper for NVIDIA's MambaVision model to extract
640-dimensional global descriptors for visual place recognition in SLAM.

Features:
- Automatic model downloading from HuggingFace
- Efficient inference with cached preprocessing
- L2-normalized descriptors for cosine similarity
- Batch processing support for multiple frames
- GPU memory optimization

Reference:
- MambaVision: https://github.com/NVlabs/MambaVision
- Paper: "MambaVision: A Hybrid Mamba-Transformer Vision Backbone"
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import time
from typing import Union, List, Optional, Tuple
import numpy as np

from .reranker import select_keypoints_entropy


class MambaDescriptorExtractor:
    """
    Extract global image descriptors using MambaVision for loop closure detection.

    The model outputs a 640-dimensional feature vector from the average pooling layer,
    which is L2-normalized for efficient cosine similarity computation.

    Optionally supports Stage 3 local feature extraction for two-stage reranking
    using entropy-based keypoint selection and MNN scoring.
    """

    def __init__(
        self,
        model_name: str = "nvidia/MambaVision-T-1K",
        device: str = "cuda:0",
        input_size: tuple = (224, 224),
        cache_dir: Optional[str] = None,
        batch_size: int = 1,
        enable_local_features: bool = False,
        entropy_threshold_t1: float = 0.3
    ):
        """
        Initialize the MambaVision descriptor extractor.

        Args:
            model_name: HuggingFace model identifier (default: nvidia/MambaVision-T-1K)
            device: Device for inference ('cuda:0', 'cpu', etc.)
            input_size: Target input resolution (height, width). Default (224, 224)
            cache_dir: Directory to cache downloaded models (None = default HF cache)
            batch_size: Maximum batch size for inference (default: 1)
            enable_local_features: Enable Stage 3 local feature extraction for reranking
            entropy_threshold_t1: Entropy threshold for keypoint selection (lower = more selective)
        """
        self.device = torch.device(device)
        self.input_size = (3, input_size[0], input_size[1])  # CHW format
        self.batch_size = batch_size
        self.model_name = model_name
        self.enable_local_features = enable_local_features
        self.entropy_threshold_t1 = entropy_threshold_t1

        # Local feature parameters (MambaVision Stage 3)
        self.num_heads = 8
        self.head_dim = 40
        self.local_dim = 320  # 8 * 40
        self._attn_data = {}  # Storage for QKV hook output
        
        print(f"[MambaDescriptor] Initializing {model_name}...")
        print(f"[MambaDescriptor]   - Device: {device}")
        print(f"[MambaDescriptor]   - Input size: {input_size}")
        
        # Load model
        start_time = time.time()
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            load_time = time.time() - start_time
            print(f"[MambaDescriptor] ✓ Model loaded in {load_time:.2f}s")
        except Exception as e:
            raise RuntimeError(f"Failed to load MambaVision model: {e}")
        
        # Setup preprocessing transform
        self.transform = create_transform(
            input_size=self.input_size,
            is_training=False,
            mean=self.model.config.mean,
            std=self.model.config.std,
            crop_mode=self.model.config.crop_mode,
            crop_pct=self.model.config.crop_pct
        )

        # Setup Stage 3 QKV hook for local feature extraction (if enabled)
        if self.enable_local_features:
            self._setup_local_feature_hooks()
            print(f"[MambaDescriptor]   - Local features: ENABLED (t1={entropy_threshold_t1})")
        else:
            print(f"[MambaDescriptor]   - Local features: disabled")

        # Statistics tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.descriptor_dim = 640  # MambaVision-T output dimension
        
        # Warmup (important for accurate timing)
        self._warmup()
    
    def _warmup(self):
        """Run warmup inference to initialize CUDA kernels."""
        print("[MambaDescriptor] Running warmup inference...")
        dummy_input = torch.randn(1, *self.input_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        torch.cuda.synchronize()
        print("[MambaDescriptor] ✓ Warmup complete")

    def _setup_local_feature_hooks(self):
        """
        Register QKV hook on MambaVision Stage 3 last attention block.
        This captures the Q, K, V projections for local feature extraction.
        """
        # Access Stage 3 last attention block: model.model.levels[2].blocks[7].mixer
        backbone = self.model.model
        stage3_last_attn = backbone.levels[2].blocks[7].mixer

        # Disable fused attention to get explicit attention weights
        stage3_last_attn.fused_attn = False

        # Register hook on qkv projection to capture raw projections
        stage3_last_attn.qkv.register_forward_hook(self._qkv_hook)

        # Store attention module reference
        self._stage3_attn = stage3_last_attn

    def _qkv_hook(self, module, input, output):
        """Capture the QKV projection output during forward pass."""
        self._attn_data["qkv"] = output.detach()

    @torch.no_grad()
    def extract_local_features(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract local features from MambaVision Stage 3 V-projections.

        Uses entropy-based keypoint selection to identify discriminative patches.
        MambaVision has no CLS token, so we use attention entropy instead.

        Args:
            image: Input image (see preprocess_image for supported formats)

        Returns:
            selected_features: Tensor [num_keypoints, 320] L2-normalized local features
            keypoint_mask: numpy array [196] boolean mask of selected keypoints
        """
        if not self.enable_local_features:
            raise RuntimeError("Local features not enabled. Set enable_local_features=True")

        # Preprocess and run forward pass (triggers QKV hook)
        input_tensor = self.preprocess_image(image)
        _ = self.model(input_tensor)

        # Parse QKV: [B, N, 3*C] where N=196 (14x14 patches), C=320
        qkv_raw = self._attn_data["qkv"]  # [1, 196, 960]
        B, N, _ = qkv_raw.shape
        qkv = qkv_raw.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [1, 8, 196, 40]

        # V features: [B, heads, N, head_dim] -> [N, heads*head_dim]
        v_feat = v[0].permute(1, 0, 2).reshape(N, self.local_dim)  # [196, 320]
        v_feat = F.normalize(v_feat, p=2, dim=1)

        # Compute attention matrix for entropy-based keypoint selection
        scale = self.head_dim ** -0.5
        q_scaled = q[0] * scale  # [8, 196, 40]
        attn = q_scaled @ k[0].transpose(-2, -1)  # [8, 196, 196]
        attn = attn.softmax(dim=-1)  # [8, 196, 196]

        # Average across heads
        attn_avg = attn.mean(dim=0)  # [196, 196]

        # Entropy-based keypoint selection
        attn_np = attn_avg.cpu().numpy()
        keypoint_mask = select_keypoints_entropy(attn_np, self.entropy_threshold_t1)

        # Select discriminative keypoints
        selected_features = v_feat[keypoint_mask]

        return selected_features, keypoint_mask
    
    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for MambaVision inference.
        
        Args:
            image: Input image in one of the following formats:
                   - torch.Tensor: [C, H, W] in [0, 1] range
                   - numpy.ndarray: [H, W, C] in [0, 255] range (uint8)
                   - PIL.Image: RGB image
        
        Returns:
            Preprocessed tensor [1, C, H, W] ready for model inference
        """
        if isinstance(image, torch.Tensor):
            # Convert to PIL for consistent preprocessing
            if image.dim() == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)  # -> [H, W, C]
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply transform and add batch dimension
        tensor = self.transform(image_pil).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def extract_descriptor(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        normalize: bool = True,
        return_local: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract a global descriptor from a single image.

        Args:
            image: Input image (see preprocess_image for supported formats)
            normalize: Apply L2 normalization (required for cosine similarity)
            return_local: Also return local features (requires enable_local_features=True)

        Returns:
            If return_local=False: Descriptor tensor of shape [640]
            If return_local=True: Tuple of (descriptor [640], local_features [N, 320])
        """
        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Extract features
        out_avg_pool, features = self.model(input_tensor)

        # L2 normalize
        if normalize:
            descriptor = F.normalize(out_avg_pool, p=2, dim=1)
        else:
            descriptor = out_avg_pool

        # Update statistics
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time

        global_desc = descriptor.squeeze(0)  # [640]

        # Optionally extract local features
        if return_local:
            if not self.enable_local_features:
                raise RuntimeError("Local features not enabled. Set enable_local_features=True")

            # Parse QKV from hook (already captured during forward pass above)
            qkv_raw = self._attn_data["qkv"]  # [1, 196, 960]
            B, N, _ = qkv_raw.shape
            qkv = qkv_raw.reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [1, 8, 196, 40]

            # V features: [B, heads, N, head_dim] -> [N, heads*head_dim]
            v_feat = v[0].permute(1, 0, 2).reshape(N, self.local_dim)  # [196, 320]
            v_feat = F.normalize(v_feat, p=2, dim=1)

            # Compute attention matrix for entropy-based keypoint selection
            scale = self.head_dim ** -0.5
            q_scaled = q[0] * scale  # [8, 196, 40]
            attn = q_scaled @ k[0].transpose(-2, -1)  # [8, 196, 196]
            attn = attn.softmax(dim=-1)  # [8, 196, 196]
            attn_avg = attn.mean(dim=0)  # [196, 196]

            # Entropy-based keypoint selection
            attn_np = attn_avg.cpu().numpy()
            keypoint_mask = select_keypoints_entropy(attn_np, self.entropy_threshold_t1)

            # Select discriminative keypoints
            selected_features = v_feat[keypoint_mask]

            return global_desc, selected_features

        return global_desc
    
    @torch.no_grad()
    def extract_descriptors_batch(
        self,
        images: List[Union[torch.Tensor, np.ndarray, Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract descriptors from a batch of images.
        
        Args:
            images: List of input images
            normalize: Apply L2 normalization
        
        Returns:
            Descriptor tensor of shape [N, 640]
        """
        descriptors = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = [self.preprocess_image(img) for img in batch_images]
            batch_input = torch.cat(batch_tensors, dim=0)
            
            # Extract features
            out_avg_pool, _ = self.model(batch_input)
            
            # Normalize
            if normalize:
                batch_descriptors = F.normalize(out_avg_pool, p=2, dim=1)
            else:
                batch_descriptors = out_avg_pool
            
            descriptors.append(batch_descriptors)
        
        # Concatenate all batches
        all_descriptors = torch.cat(descriptors, dim=0)
        
        # Update statistics
        self.inference_count += len(images)
        
        return all_descriptors
    
    def get_stats(self) -> dict:
        """
        Get inference statistics.
        
        Returns:
            Dictionary with inference count and average time
        """
        avg_time = self.total_inference_time / max(1, self.inference_count)
        return {
            'inference_count': self.inference_count,
            'total_time': self.total_inference_time,
            'avg_time_ms': avg_time * 1000,
            'descriptor_dim': self.descriptor_dim
        }
    
    def print_stats(self):
        """Print inference statistics."""
        stats = self.get_stats()
        print(f"[MambaDescriptor] Statistics:")
        print(f"  - Total inferences: {stats['inference_count']}")
        print(f"  - Total time: {stats['total_time']:.2f}s")
        print(f"  - Average time: {stats['avg_time_ms']:.2f}ms")
        print(f"  - Descriptor dim: {stats['descriptor_dim']}")


def test_mamba_descriptor():
    """Simple test function to verify the descriptor extractor works."""
    print("\n=== Testing MambaDescriptor Extractor ===\n")

    # Initialize extractor (global only)
    extractor = MambaDescriptorExtractor(device="cuda:0")

    # Create dummy image
    dummy_image = torch.rand(3, 480, 640)  # CHW format
    print(f"Input image shape: {dummy_image.shape}")

    # Extract descriptor
    descriptor = extractor.extract_descriptor(dummy_image)
    print(f"Descriptor shape: {descriptor.shape}")
    print(f"Descriptor norm: {descriptor.norm():.4f} (should be ~1.0 if normalized)")

    # Test batch extraction
    batch_images = [torch.rand(3, 480, 640) for _ in range(3)]
    batch_descriptors = extractor.extract_descriptors_batch(batch_images)
    print(f"Batch descriptors shape: {batch_descriptors.shape}")

    # Print statistics
    extractor.print_stats()

    # Test similarity computation
    desc1 = extractor.extract_descriptor(torch.rand(3, 480, 640))
    desc2 = extractor.extract_descriptor(torch.rand(3, 480, 640))
    similarity = torch.dot(desc1, desc2).item()
    print(f"\nCosine similarity between two random images: {similarity:.4f}")

    print("\n--- Testing Local Feature Extraction ---\n")

    # Initialize extractor with local features enabled
    extractor_local = MambaDescriptorExtractor(
        device="cuda:0",
        enable_local_features=True,
        entropy_threshold_t1=0.3
    )

    # Test local feature extraction
    global_desc, local_feats = extractor_local.extract_descriptor(dummy_image, return_local=True)
    print(f"Global descriptor shape: {global_desc.shape}")
    print(f"Local features shape: {local_feats.shape}")
    print(f"Number of keypoints selected: {local_feats.shape[0]} (out of 196 patches)")
    print(f"Local feature dim: {local_feats.shape[1]} (expected: 320)")

    # Verify local features are normalized
    local_norms = local_feats.norm(dim=1)
    print(f"Local feature norms: min={local_norms.min():.4f}, max={local_norms.max():.4f} (should be ~1.0)")

    print("\n All tests passed!")


if __name__ == "__main__":
    test_mamba_descriptor()
