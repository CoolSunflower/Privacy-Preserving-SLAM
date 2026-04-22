from typing import Union, List, Tuple, Optional
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.dyn_uncertainty.median_filter import MedianPool2d


def resample_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resample a tensor to a target shape using specified interpolation mode.

    Args:
        tensor: Input tensor to resample, shape should be [H,W], no B and C
        target_shape: Desired output shape (height, width)
        interpolation_mode: Interpolation method ("bilinear" or "bicubic")

    Returns:
        Resampled tensor of shape target_shape
    """
    tensor = tensor.view((1, 1) + tensor.shape[:2])
    return (
        F.interpolate(tensor, size=target_shape, mode=interpolation_mode)
        .squeeze(0)
        .squeeze(0)
    )


"""Mapping loss function."""
# Constants
EPSILON = torch.finfo(torch.float32).eps
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2
SSIM_C3 = SSIM_C2 / 2
GAUSSIAN_SIGMA = 1.5
SSIM_MAX_CLIP = 0.98
DEPTH_MAX_CLIP = 5.0


def compute_bias_factor(x: float, s: float) -> float:
    """
    Compute bias factor for adaptive weighting.
    This is from Nerf-on-the-go

    Args:
        x: Input value
        s: Scaling factor

    Returns:
        Computed bias value
    """
    return x / (1 + (1 - x) * (1 / s - 2))


def generate_gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """
    Generate 1D Gaussian kernel.

    Args:
        window_size: Size of the window
        sigma: Standard deviation of Gaussian

    Returns:
        Normalized Gaussian kernel
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_2d_gaussian_window(window_size: int, num_channels: int) -> torch.Tensor:
    """
    Create 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the window
        num_channels: Number of channels in the input

    Returns:
        2D Gaussian window
    """
    _1D_window = generate_gaussian_kernel(window_size, GAUSSIAN_SIGMA).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(num_channels, 1, window_size, window_size).contiguous()
    )
    return window


def compute_ssim_components(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SSIM components.
    Not the same as the standard SSIM,
    see details in the head comments of compute_mapping_loss_components

    Args:
        img1: First input image
        img2: Second input image
        window_size: Size of Gaussian window

    Returns:
        Tuple of (luminance, contrast, structure) components
    """
    num_channels = img1.size(-3)
    window = create_2d_gaussian_window(window_size, num_channels)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, num_channels)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    num_channels: int,
    eps: float = EPSILON,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute individual SSIM components (luminance, contrast, structure).
    Not the same as the standard SSIM,
    see details in the head comments of compute_mapping_loss_components

    Args:
        img1, img2: Input images
        window: Gaussian window
        window_size: Window size
        num_channels: Number of channels
        eps: Small constant for numerical stability

    Returns:
        Tuple of (luminance, contrast, structure) components
    """
    # Handle single image case
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        unsqueeze_orig = True
    else:
        unsqueeze_orig = False

    # Compute means and variances
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=num_channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=num_channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=num_channels)
        - mu1_mu2
    )

    # Ensure valid values
    epsilon = torch.tensor([eps]).to(img1.device)
    sigma1_sq = torch.maximum(epsilon, sigma1_sq)
    sigma2_sq = torch.maximum(epsilon, sigma2_sq)
    sigma12 = torch.sign(sigma12) * torch.minimum(
        torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12)
    )

    # Compute SSIM components
    luminance = (2 * mu1_mu2 + SSIM_C1) / (mu1_sq + mu2_sq + SSIM_C1)
    contrast = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C2) / (
        sigma1_sq + sigma2_sq + SSIM_C2
    )
    structure = (sigma12 + SSIM_C3) / (
        torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C3
    )

    # Apply clipping
    contrast = torch.clamp(contrast, max=SSIM_MAX_CLIP)
    structure = torch.clamp(structure, max=SSIM_MAX_CLIP)

    if unsqueeze_orig:
        return (
            luminance.mean(1).squeeze(),
            contrast.mean(1).squeeze(),
            structure.mean(1).squeeze(),
        )
    return luminance.mean(1), contrast.mean(1), structure.mean(1)


def compute_mapping_loss_components(
    gt_img: torch.Tensor,
    rendered_img: torch.Tensor,
    ref_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    uncertainty: torch.Tensor,
    opacity: torch.Tensor,
    train_fraction: float,
    ssim_fraction: float,
    uncertainty_config: dict,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute essential components for uncertainty-aware mapping loss.

    This function computes four key components used in the final mapping loss:
    1. Uncertainty loss: Based on SSIM and depth differences, weighted by predicted uncertainty
    2. Resized uncertainty: Uncertainty values resampled to match image dimensions
    3. RGB L1 loss: Masked absolute differences between rendered and ground truth RGB values
    4. Depth L1 loss: Masked absolute differences between rendered and reference depth values

    Note: The SSIM loss here is not the same as the common one. Key differences include:
            1. We clip contrast and structure components to a maximum value
            2. the equation is modified to (1-c)(1-s)(1-l) according to nerf-on-the-go
    Args:
        gt_img: Ground truth RGB image [C,H,W]
        rendered_img: Rendered RGB image [C,H,W]
        ref_depth: Reference depth map [1,H,W] (from metric depth)
        rendered_depth: Rendered depth map [1,H,W]
        uncertainty: Model's uncertainty estimates [H',W'] (downsampled due to dino)
        opacity: Rendering opacity mask [1,H,W]
        train_fraction: Training progress (0-1) for adaptive weighting
        ssim_fraction: SSIM loss weight fraction
        uncertainty_config: Dictionary containing uncertainty estimation parameters
        mask: Optional visibility mask for loss computation [1,H,W]
    """
    # Initialize median pooling for SSIM
    median_filter = MedianPool2d(
        kernel_size=uncertainty_config["ssim_median_filter_size"],
        stride=1,
        padding=0,
        same=True,
    )
    _, h, w = gt_img.shape

    # Compute RGB L1 loss with masking
    rgb_l1_loss = torch.abs(rendered_img * mask - gt_img * mask)

    # Compute depth loss with adaptive thresholding
    median_depth = ref_depth.median()
    depth_threshold = min(10 * median_depth, 50)
    depth_mask = ((ref_depth > 0.01) & (ref_depth < depth_threshold)).view(
        *rendered_depth.shape
    )
    depth_l1_loss = (
        torch.abs(rendered_depth * depth_mask - ref_depth * depth_mask)
    )

    # Process uncertainty values
    processed_uncertainty = torch.clip(uncertainty, min=0.1) + 1e-3
    resized_uncertainty = resample_tensor_to_shape(
        processed_uncertainty.detach(), (h, w)
    )
    # 0.8 is ssim_anneal, this number is directly taken from nerf-on-the-go
    data_rate = 1 + 1 * compute_bias_factor(train_fraction, 0.8)
    resized_uncertainty = (resized_uncertainty - 0.1) * data_rate + 0.1

    # Process opacity
    resized_opacity = opacity.detach().view((h, w))
    small_opacity = resample_tensor_to_shape(resized_opacity, uncertainty.shape)

    # Compute SSIM-based loss
    # 0.8 is ssim_anneal, this number is directly taken from nerf-on-the-go
    ssim_weight = 100 + 900 * compute_bias_factor(ssim_fraction, 0.8)
    luminance, contrast, structure = compute_ssim_components(
        gt_img, rendered_img, window_size=uncertainty_config["ssim_window_size"]
    )
    ssim_loss = torch.clip(
        resized_opacity
        * ssim_weight
        * (1 - luminance)
        * (1 - structure)
        * (1 - contrast),
        max=5.0,
    )

    # Process SSIM loss for uncertainty computation
    small_ssim_loss = resample_tensor_to_shape(ssim_loss.detach(), uncertainty.shape)
    filtered_ssim_loss = (
        median_filter(small_ssim_loss.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    )

    # Process depth loss for uncertainty computation
    small_depth_loss = resample_tensor_to_shape(
        torch.clip(depth_l1_loss.squeeze(), max=5.0).detach(),
        uncertainty.shape,
        "bicubic",
    )
    small_depth = resample_tensor_to_shape(
        ref_depth.squeeze().detach(), uncertainty.shape, "bicubic"
    )
    # do not penalize far away pixels
    small_depth_loss[small_depth > depth_threshold] = 0.0

    # Compute final uncertainty loss
    uncertainty_loss = (
        filtered_ssim_loss / processed_uncertainty ** 2
        + 0.5 * torch.log(processed_uncertainty)
        + uncertainty_config["uncer_depth_mult"]
        * small_depth_loss
        / processed_uncertainty ** 2
    )
    uncertainty_loss[
        small_opacity < uncertainty_config["opacity_th_for_uncer_loss"]
    ] = 0

    return uncertainty_loss, resized_uncertainty, rgb_l1_loss, depth_l1_loss

"""Regularization loss for DINO model based on feature similarity."""
# Constants
TOP_K_FEATURES = 128
SIMILARITY_THRESHOLD = 0.75
# EPSILON = torch.finfo(torch.float32).eps ## this is defined above


def compute_dino_regularization_loss(
    uncertainty_buffer: Union[torch.Tensor, List[torch.Tensor]],
    feature_buffer: Union[torch.Tensor, List[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute DINO regularization loss based on uncertainty and feature buffers.
    Implementation based on equations (2) & (3) in NeRF-on-the-Go paper.

    Args:
        uncertainty_buffer: Tensor or list of tensors containing uncertainty values
        feature_buffer: Tensor or list of tensors containing feature vectors

    Returns:
        torch.Tensor: Mean uncertainty variance across similar features
    """
    # Convert lists to tensors if needed
    uncertainty = _ensure_tensor(uncertainty_buffer)
    features = _ensure_tensor(feature_buffer)

    # Reshape and normalize features
    feature_dim = features.shape[-1]
    uncertainty_flat = uncertainty.view(-1, 1)
    features_flat = features.contiguous().view(-1, feature_dim)
    features_normalized = F.normalize(features_flat, p=2, dim=-1)

    # Validate shape of inputs
    if uncertainty_flat.shape[0] != features_normalized.shape[0]:
        raise ValueError(
            "Uncertainty and feature buffers must have same number of samples"
            + f"but got {uncertainty_flat.shape[0]} and {features_normalized.shape[0]}"
        )

    # Compute feature similarity matrix
    similarity_matrix = torch.matmul(features_normalized, features_normalized.T)

    # Find top-k similar features above threshold
    k = min(TOP_K_FEATURES, similarity_matrix.shape[-1])
    top_similarities, top_indices = torch.topk(similarity_matrix, k=k, dim=-1)
    similarity_mask = (top_similarities > SIMILARITY_THRESHOLD).float()

    # Compute uncertainty statistics
    neighbor_uncertainties = uncertainty_flat[top_indices] * similarity_mask.unsqueeze(
        -1
    )
    uncertainty_sums = torch.sum(neighbor_uncertainties, dim=1)
    valid_neighbor_counts = torch.sum(similarity_mask, dim=-1, keepdim=True) + EPSILON

    # Calculate mean and variance
    uncertainty_means = uncertainty_sums / valid_neighbor_counts
    squared_differences = (
        neighbor_uncertainties - uncertainty_means.unsqueeze(-1)
    ) ** 2 * similarity_mask.unsqueeze(-1)
    uncertainty_variances = (
        torch.sum(squared_differences, dim=1) / valid_neighbor_counts
    )

    return torch.mean(uncertainty_variances)


def _ensure_tensor(buffer: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """Convert list of tensors to single tensor if needed."""
    return torch.stack(buffer) if isinstance(buffer, list) else buffer


def compute_uncertainty_regularization_loss(
    uncertainty: torch.Tensor,
    opacity: Optional[torch.Tensor] = None,
    regularization_config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Compute regularization loss to prevent uncertainty network from predicting 
    uniformly high uncertainty across the entire map.
    
    This addresses a critical failure mode where the network learns a trivial solution:
    predict everything as uncertain → losses weighted by 1/u² decrease → network happy!
    
    The solution uses multiple complementary regularizations:
    
    1. **Mean Penalty**: Directly penalize high average uncertainty
       - Encourages network to keep mean uncertainty low
       - Stronger than just log(u) regularizer
    
    2. **Variance Encouragement**: Reward diversity in predictions
       - High variance = some high, some low (good discrimination)
       - Low variance = everything similar (bad - trivial solution)
    
    3. **Sparsity Prior**: Encourage most pixels to have low uncertainty
       - Based on assumption: most of scene is static
       - Uses L1 penalty on uncertainty values
    
    4. **Entropy Penalty**: Prevent uniform distributions
       - Measures how "flat" the uncertainty distribution is
       - High entropy = uniform = bad
    
    Args:
        uncertainty: Predicted uncertainty map [H, W] or [H', W'] (at any resolution)
            - Will be automatically resized to match opacity shape if needed
            - Should be before any processing/clipping for accurate regularization
        opacity: Optional opacity mask [H, W] to only regularize visible regions
            - If provided and shape differs from uncertainty, uncertainty will be resized
        regularization_config: Configuration dict with keys:
            - enabled: bool, master switch (default: True)
            - mean_penalty_weight: float, penalty for high mean (default: 0.5)
            - variance_weight: float, reward for high variance (default: -0.2, negative = reward)
            - sparsity_weight: float, L1 penalty on uncertainty (default: 0.3)
            - entropy_weight: float, penalty for flat distribution (default: 0.1)
            - target_mean: float, desired mean uncertainty (default: 0.5)
            - opacity_threshold: float, min opacity to consider (default: 0.5)
    
    Returns:
        regularization_loss: Scalar tensor (0 if disabled)
    
    Mathematical Formulation:
    
    Given uncertainty predictions u ∈ [H, W] and valid mask M:
    
    1. Mean penalty: 
       L_mean = λ_mean * max(0, mean(u[M]) - u_target)²
       Only penalize if mean exceeds target
    
    2. Variance reward (negative weight = reward):
       L_var = λ_var * var(u[M])
       Higher variance → lower loss (because λ_var < 0)
    
    3. Sparsity penalty:
       L_sparse = λ_sparse * mean(|u[M]|)
       Encourages small absolute values
    
    4. Entropy penalty:
       First normalize: p = softmax(u[M])
       Then: L_entropy = λ_entropy * (-Σ p·log(p))
       Low entropy = peaky distribution = good
    
    Total: L_reg = L_mean + L_var + L_sparse + L_entropy
    
    Example Configuration:
        uncertainty_regularization:
            enabled: True
            mean_penalty_weight: 1.0      # Strong penalty for high mean
            variance_weight: -0.3         # Reward diversity (negative!)
            sparsity_weight: 0.5          # Encourage low values
            entropy_weight: 0.2           # Discourage uniform distribution
            target_mean: 0.5              # Desired mean uncertainty
    """
    if regularization_config is None:
        regularization_config = {}
    
    if not regularization_config.get("enabled", True):
        return torch.tensor(0.0, device=uncertainty.device)
    
    # Get config parameters with defaults
    mean_penalty_weight = regularization_config.get("mean_penalty_weight", 0.5)
    variance_weight = regularization_config.get("variance_weight", -0.2)  # Negative = reward
    sparsity_weight = regularization_config.get("sparsity_weight", 0.3)
    entropy_weight = regularization_config.get("entropy_weight", 0.1)
    target_mean = regularization_config.get("target_mean", 0.5)
    opacity_threshold = regularization_config.get("opacity_threshold", 0.5)
    
    # Handle shape mismatch: uncertainty may be at feature resolution, opacity at render resolution
    if opacity is not None:
        if opacity.dim() == 3:
            opacity = opacity.squeeze(0)
        
        # Resize uncertainty to match opacity shape if needed
        if uncertainty.shape != opacity.shape:
            target_h, target_w = opacity.shape[-2], opacity.shape[-1]
            uncertainty = resample_tensor_to_shape(uncertainty, (target_h, target_w))
        
        valid_mask = opacity > opacity_threshold
    else:
        valid_mask = torch.ones_like(uncertainty, dtype=torch.bool)
    
    # Extract valid uncertainty values
    valid_uncertainty = uncertainty[valid_mask]
    
    if valid_uncertainty.numel() < 10:  # Not enough pixels
        return torch.tensor(0.0, device=uncertainty.device)
    
    total_loss = 0.0
    
    # 1. Mean Penalty: Penalize if mean exceeds target
    mean_uncertainty = valid_uncertainty.mean()
    if mean_uncertainty > target_mean:
        mean_penalty = mean_penalty_weight * (mean_uncertainty - (target_mean/2)) ** 2
        total_loss = total_loss + mean_penalty

        # 4. Entropy Penalty: Discourage flat/uniform distributions
        # Normalize to probability distribution and compute entropy
        if entropy_weight > 0:
            # Clip to avoid numerical issues
            u_clipped = torch.clamp(valid_uncertainty, min=1e-6, max=10.0)
            # Normalize to sum to 1 (treat as probability)
            u_normalized = u_clipped / (u_clipped.sum() + EPSILON)
            # Compute entropy: H = -Σ p·log(p)
            entropy = -(u_normalized * torch.log(u_normalized + EPSILON)).sum()
            # Penalize high entropy (flat distribution)
            entropy_penalty = entropy_weight * entropy
            total_loss = total_loss + entropy_penalty

        # 2. Variance Reward: Encourage diversity (negative weight = reward)
        # Higher variance → more discrimination between static/dynamic
        variance_uncertainty = valid_uncertainty.var()
        variance_term = variance_weight * variance_uncertainty
        total_loss = total_loss + variance_term
    
    # 3. Sparsity Penalty: L1 regularization on uncertainty values
    # Encourages most pixels to have low uncertainty
    # sparsity_penalty = sparsity_weight * valid_uncertainty.abs().mean()
    # total_loss = total_loss + sparsity_penalty    
    
    return total_loss


def compute_depth_uncertainty_correlation_penalty(
    rendered_depth: torch.Tensor,
    uncertainty: torch.Tensor,
    correlation_config: dict,
    opacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute correlation penalty loss to discourage high uncertainty for far-away objects.
    
    This loss penalizes the uncertainty network when it predicts high uncertainty for pixels
    with large depth values. The intuition is that far-away static objects should not be
    uncertain - uncertainty should primarily appear for close dynamic objects or occlusion
    boundaries.
    
    The loss uses an adaptive threshold based on depth statistics to determine when to
    apply the penalty, avoiding hard-coded thresholds that may not generalize across scenes.
    
    Args:
        rendered_depth: Rendered depth map [1, H, W] or [H, W]
        uncertainty: Predicted uncertainty map [H', W'] (typically downsampled)
        correlation_config: Dictionary containing:
            - enabled: bool, whether to enable this loss
            - correlation_threshold: float, Pearson correlation above which to apply penalty
            - depth_percentile: float, percentile of depth to use as "far" threshold (e.g., 75)
            - min_depth_threshold: float, minimum absolute depth to consider (e.g., 3.0m)
            - penalty_strength: float, how strongly to penalize (e.g., 0.5-2.0)
        opacity: Optional opacity mask [1, H, W], to only consider visible regions
    
    Returns:
        correlation_penalty: Scalar loss value (0 if disabled or correlation below threshold)
    
    Example config:
        correlation_penalty:
            enabled: True
            correlation_threshold: 0.3  # Only penalize if correlation > 0.3
            depth_percentile: 75.0      # Top 25% of depth values are "far"
            min_depth_threshold: 3.0    # Minimum 3m to be considered "far"
            penalty_strength: 1.0       # Loss multiplier
    """
    if not correlation_config.get("enabled", False):
        return torch.tensor(0.0, device=uncertainty.device)
    
    # Squeeze depth to [H, W] if needed
    if rendered_depth.dim() == 3:
        rendered_depth = rendered_depth.squeeze(0)
    
    # Resize uncertainty to match depth shape
    target_shape = rendered_depth.shape[-2:]  # Get (H, W) tuple
    uncertainty_resized = resample_tensor_to_shape(
        uncertainty, target_shape, interpolation_mode="bilinear"
    )
    
    # Create valid mask (combine opacity and depth validity)
    valid_mask = rendered_depth > 0.01  # Valid depth
    if opacity is not None:
        if opacity.dim() == 3:
            opacity = opacity.squeeze(0)
        opacity_threshold = correlation_config.get("opacity_threshold", 0.5)
        valid_mask = valid_mask & (opacity > opacity_threshold)
    
    # Extract valid pixels
    valid_depth = rendered_depth[valid_mask]
    valid_uncertainty = uncertainty_resized[valid_mask]
    
    if valid_depth.numel() < 100:  # Not enough valid pixels
        return torch.tensor(0.0, device=uncertainty.device)
    
    # Compute Pearson correlation coefficient
    depth_mean = valid_depth.mean()
    uncertainty_mean = valid_uncertainty.mean()
    
    depth_centered = valid_depth - depth_mean
    uncertainty_centered = valid_uncertainty - uncertainty_mean
    
    covariance = (depth_centered * uncertainty_centered).mean()
    depth_std = depth_centered.std() + EPSILON
    uncertainty_std = uncertainty_centered.std() + EPSILON
    
    correlation = covariance / (depth_std * uncertainty_std)
    
    # Only apply penalty if correlation exceeds threshold
    correlation_threshold = correlation_config.get("correlation_threshold", 0.3)
    if correlation < correlation_threshold:
        return torch.tensor(0.0, device=uncertainty.device)
    
    # Determine "far" depth threshold adaptively
    depth_percentile = correlation_config.get("depth_percentile", 75.0)
    min_depth_threshold = correlation_config.get("min_depth_threshold", 3.0)
    
    # Compute adaptive threshold as max of percentile and minimum
    percentile_threshold = torch.quantile(valid_depth, depth_percentile / 100.0)
    far_depth_threshold = max(percentile_threshold.item(), min_depth_threshold)
    
    # Identify "far" pixels
    far_mask = valid_depth > far_depth_threshold
    
    if far_mask.sum() < 10:  # Not enough far pixels
        return torch.tensor(0.0, device=uncertainty.device)
    
    # Penalize high uncertainty at far depths
    # Loss = mean(uncertainty^2) for pixels beyond far_depth_threshold
    # We use squared uncertainty to penalize high values more strongly
    far_uncertainty = valid_uncertainty[far_mask]
    penalty_strength = correlation_config.get("penalty_strength", 1.0)
    
    # The penalty is the mean squared uncertainty of far pixels
    # Squared because we want to strongly discourage high uncertainty values
    penalty_loss = penalty_strength * (far_uncertainty ** 2).mean()
    
    return penalty_loss
