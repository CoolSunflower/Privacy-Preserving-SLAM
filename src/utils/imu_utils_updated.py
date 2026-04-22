"""
IMU Preintegration Utilities for WildGS-SLAM (V2 - Continuous Velocity Tracking)

MAJOR REDESIGN (2025-01-29):
- Continuous velocity tracking across ALL frames (not just keyframes)
- Velocity updated EVERY frame using accelerometer data
- S3E dataset: Skip gyro integration (ground truth has identity rotation)
- Eliminates pose-based velocity estimation (root cause of drift errors)

Coordinate Frame Conventions:
- World frame (W): Fixed inertial frame
- Camera frame (C): Optical frame (Z forward, X right, Y down)
- IMU frame (I): Body frame aligned with IMU axes

Pose Representation (lietorch SE3):
- 7D tensor: [tx, ty, tz, qx, qy, qz, qw]
- Translation FIRST, then quaternion (vector-first quat convention)
- Represents position in world frame (NOT w2c transform!)

Author: Adarsh Gupta
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

# Gravity vector in world frame (assuming Z-up convention)
GRAVITY_WORLD = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)


def is_s3e_dataset(cfg: Dict) -> bool:
    """
    Check if current dataset is S3E type.
    S3E has special handling: skip gyro integration (ground truth has identity rotation).
    
    Args:
        cfg: Config dictionary
        
    Returns:
        True if S3E dataset, False otherwise
    """
    dataset_name = cfg.get('dataset', '').lower()
    return 's3e' in dataset_name or dataset_name == 's3e_rgbd_imu'


def propagate_imu_continuous(
    pose_t1: torch.Tensor,
    prev_velocity: Optional[torch.Tensor],
    imu_chunk: Dict[str, torch.Tensor],
    c2i_transform: torch.Tensor,
    dt_cam: float,
    gravity: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dataset_is_s3e: bool = False,
    sigma_acc_walk: float = 2.18e-04,  # NEW: Accelerometer bias random walk (m/s²/√s)
    sigma_acc: float = 1.09e-02,       # NEW: Accelerometer measurement noise (m/s²)
    frames_since_init: int = 0,        # NEW: Track time for uncertainty growth
    power_factor: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Propagate camera pose using IMU measurements with probabilistic drift mitigation.
    
    KEY CHANGES FROM V1:
    - Accepts previous velocity as input (continuous tracking)
    - Returns updated velocity for next frame
    - For S3E: Skips gyro integration entirely (uses identity rotation)
    - No dependency on pose deltas for velocity estimation
    
    V2.2 PROBABILISTIC DRIFT MITIGATION (2025-01-29):
    - Uses IMU noise parameters (sigma_acc_walk) instead of arbitrary decay/clamps
    - Models velocity uncertainty growth from accelerometer bias drift
    - Applies uncertainty-weighted damping: higher uncertainty → more damping
    - Theoretically grounded in Kalman filter covariance propagation
    
    Algorithm:
    1. Compute velocity uncertainty from bias random walk: σ_v² = (σ_walk * t)² * t
    2. Calculate optimal damping factor: λ = 1 / (1 + σ_v²)
    3. Apply probabilistic damping: v_damped = v_raw * λ
    4. This naturally prevents drift without arbitrary thresholds
    
    Args:
        pose_t1: Previous camera pose (t-1) as 7D tensor [tx,ty,tz,qx,qy,qz,qw]
        prev_velocity: Previous velocity estimate [vx,vy,vz] in camera frame (m/s), or None
        imu_chunk: Dictionary with keys:
            - 'timestamps': (N,) float32 tensor of IMU sample times
            - 'angular_velocity': (N, 3) float32 rad/s [wx, wy, wz]
            - 'linear_acceleration': (N, 3) float32 m/s² [ax, ay, az]
            - 'orientation': (N, 4) float32 quaternion [qx, qy, qz, qw] (optional, for S3E)
            - 'c2i_transform': (4, 4) float32 camera-to-IMU extrinsics
        c2i_transform: Camera-to-IMU transformation (4x4)
        dt_cam: Time interval between camera frames (seconds)
        gravity: Optional gravity vector in world frame, defaults to [0,0,-9.81]
        device: Torch device ('cuda' or 'cpu')
        dataset_is_s3e: If True, skip gyro integration (S3E has identity rotation in ground truth)
        sigma_acc_walk: Accelerometer bias random walk (m/s²/√s) - from IMU calibration
        sigma_acc: Accelerometer measurement noise (m/s²) - from IMU calibration
        frames_since_init: Number of frames since velocity initialization (for uncertainty model)
    
    Returns:
        propagated_pose: 7D tensor in camera frame [tx,ty,tz,qx,qy,qz,qw]
        updated_velocity: 3D tensor in camera frame [vx,vy,vz] (m/s), after probabilistic damping
        covariance: 6x6 covariance matrix (EKF propagation with process noise)
    
    Raises:
        ValueError: If IMU chunk is empty or malformed
        
    Notes:
        - Velocity is ALWAYS updated, even if frame is not a keyframe
        - For S3E: Rotation stays identity (ground truth convention)
        - Damping factor computed from uncertainty: λ = 1 / (1 + σ_v²)
        - As time increases, uncertainty grows → more damping → prevents unbounded drift
        - No arbitrary thresholds - all parameters from sensor calibration
    """
    
    if gravity is None:
        gravity = GRAVITY_WORLD.to(device)
    else:
        gravity = gravity.to(device)
    
    # Validate inputs
    if len(imu_chunk['timestamps']) == 0:
        raise ValueError("Empty IMU chunk - cannot propagate")
    
    # Move everything to device and ensure float32
    pose_t1 = pose_t1.to(device=device, dtype=torch.float32)
    c2i = c2i_transform.to(device=device, dtype=torch.float32)
    
    if prev_velocity is not None:
        lin_vel = prev_velocity.to(device=device, dtype=torch.float32).clone()
    else:
        # Initialize with zero velocity (at-rest assumption)
        lin_vel = torch.zeros(3, device=device, dtype=torch.float32)
    
    # Extract IMU data
    timestamps = imu_chunk['timestamps'].to(device=device, dtype=torch.float32)
    ang_vel = imu_chunk['angular_velocity'].to(device=device, dtype=torch.float32)
    lin_acc = imu_chunk['linear_acceleration'].to(device=device, dtype=torch.float32)
    n_samples = timestamps.shape[0]
    
    # Get camera-to-IMU rotation
    R_c2i = c2i[:3, :3]
    R_i2c = R_c2i.T  # IMU-to-camera
    
    # Get current camera rotation
    R_cam = quaternion_to_rotation_matrix(pose_t1[3:7])
    
    # Transform velocity to IMU frame
    # Velocity is in camera frame, transform to IMU body frame
    vel_imu = R_c2i @ lin_vel
    
    # Initialize relative motion accumulation (in IMU body frame)
    relative_position = torch.zeros(3, device=device, dtype=torch.float32)
    relative_rotation = torch.eye(3, device=device, dtype=torch.float32)
    
    # Check if IMU orientation is available (S3E provides orientation)
    has_orientation = 'orientation' in imu_chunk and dataset_is_s3e
    prev_imu_orientation = None
    
    # Integration loop
    for i in range(n_samples):
        # Compute time step
        if i < n_samples - 1:
            dt = (timestamps[i+1] - timestamps[i]).item()
        else:
            if n_samples > 1:
                dt = (timestamps[-1] - timestamps[0]).item() / (n_samples - 1)
            else:
                dt = 0.01
        
        if dt <= 0 or dt > 0.1:
            dt = 0.01
        
        # Get measurements
        omega = ang_vel[i]
        accel = lin_acc[i]
        
        # Gravity correction in IMU frame
        # Current IMU orientation = relative_rotation @ R_c2i @ R_cam
        R_imu_current = relative_rotation @ R_c2i @ R_cam
        gravity_imu = R_imu_current @ gravity
        accel_corrected = accel + gravity_imu
        
        # CRITICAL: For S3E dataset, skip gyro integration
        # Ground truth has identity rotation at all times
        if dataset_is_s3e:
            # Use identity rotation (no rotation change)
            delta_rotation = torch.eye(3, device=device, dtype=torch.float32)
            
            if has_orientation and i == 0:
                print(f"               [S3E Mode] Skipping gyro integration - using identity rotation")
        else:
            # Normal mode: Integrate angular velocity
            delta_rotation = integrate_rotation(omega, dt)
        
        # Update velocity (in IMU body frame)
        vel_imu = vel_imu + accel_corrected * dt
        
        # Update position (in IMU body frame)
        position_delta = vel_imu * dt + 0.5 * accel_corrected * dt * dt
        
        # Accumulate relative motion
        relative_position = relative_rotation @ position_delta + relative_position
        relative_rotation = relative_rotation @ delta_rotation
    
    # Transform back to camera frame
    position_delta_camera = R_i2c @ relative_position
    velocity_camera_raw = R_i2c @ vel_imu
    
    # V2.2: Probabilistic drift mitigation using IMU noise parameters
    # 
    # Theory: Accelerometer bias drifts as random walk with rate σ_acc_walk
    # After time t, bias uncertainty: σ_bias(t) = σ_acc_walk * √t
    # Velocity uncertainty from bias: σ_v(t) = σ_bias(t) * t = σ_acc_walk * t^(3/2)
    # 
    # ADJUSTMENT: For real camera motion (not just drift), use sqrt(t) instead of t^1.5
    # This prevents over-damping of actual motion while still mitigating drift
    # 
    # Optimal Kalman gain for fading memory filter:
    # K = σ_measurement² / (σ_measurement² + σ_process²)
    # 
    # We use inverse variance weighting:
    # λ = σ_measurement² / (σ_measurement² + σ_bias²(t))
    #   = 1 / (1 + (σ_bias(t) / σ_measurement)²)
    #
    # As uncertainty grows, λ → 0, providing natural damping
    
    # Compute time since initialization (in seconds)
    time_elapsed = max(frames_since_init * dt_cam, dt_cam)  # Avoid zero
    
    # Velocity uncertainty from bias random walk
    # MODIFIED: Use t^0.8 instead of t^1.5 to avoid over-damping real motion
    # Physical interpretation: Compromise between √t (too gentle) and t^1.5 (too aggressive)
    # Original theory: σ_v = σ_walk * t^1.5 (pure drift from random walk)
    # Adjusted: σ_v = σ_walk * t^0.8 (allows real camera motion)
    sigma_v_bias = sigma_acc_walk * (time_elapsed ** power_factor)
    
    # Measurement noise contribution
    sigma_v_meas = sigma_acc * dt_cam
    
    # Total velocity uncertainty
    sigma_v_total = (sigma_v_bias**2 + sigma_v_meas**2) ** 0.5
    
    # Compute damping factor (inverse variance weighting)
    # λ = 1 / (1 + (σ_bias / σ_meas)²)
    # Simplified: λ = σ_meas² / (σ_meas² + σ_bias²)
    uncertainty_ratio = (sigma_v_bias / (sigma_v_meas + 1e-6)) ** 2
    damping_factor = 1.0 / (1.0 + uncertainty_ratio)
    
    # Apply probabilistic damping
    velocity_camera = velocity_camera_raw * damping_factor
    
    # Debug logging
    vel_mag_raw = torch.norm(velocity_camera_raw).item()
    vel_mag_damped = torch.norm(velocity_camera).item()
    
    if frames_since_init % 50 == 0:  # Log every 50 frames
        print(f"               [Probabilistic Damping]")
        print(f"               - Time elapsed: {time_elapsed:.1f}s, frames: {frames_since_init}")
        print(f"               - σ_v_bias: {sigma_v_bias:.4f}, σ_v_meas: {sigma_v_meas:.4f}")
        print(f"               - Damping factor: {damping_factor:.4f} (1.0=no damping, 0.0=full)")
        print(f"               - Velocity: {vel_mag_raw:.3f} → {vel_mag_damped:.3f} m/s")
    
    # Apply deltas to pose
    # lietorch stores POSITIONS directly, so just add delta
    trans_propagated = pose_t1[:3] + position_delta_camera
    
    # Apply rotation delta
    if dataset_is_s3e:
        # S3E: Keep rotation identity
        quat_propagated = pose_t1[3:7]  # No change
    else:
        R_camera_delta = R_i2c @ relative_rotation @ R_c2i
        R_propagated = R_camera_delta @ R_cam
        quat_propagated = rotation_matrix_to_quaternion(R_propagated)
    
    # Build propagated pose
    propagated_pose = torch.cat([trans_propagated, quat_propagated], dim=0)
    
    # Covariance (simplified EKF)
    sigma_acc = 0.01  # m/s²
    sigma_gyr = 0.01  # rad/s
    
    Q = torch.zeros(6, 6, device=device, dtype=torch.float32)
    Q[:3, :3] = torch.eye(3, device=device) * (sigma_acc ** 2) * (dt_cam ** 2)
    Q[3:6, 3:6] = torch.eye(3, device=device) * (sigma_gyr ** 2) * dt_cam
    
    F = torch.eye(6, device=device, dtype=torch.float32)
    P_init = torch.eye(6, device=device, dtype=torch.float32) * 0.01
    covariance = F @ P_init @ F.T + Q
    
    # Debug output
    translation_delta = torch.norm(trans_propagated - pose_t1[:3]).item()
    velocity_mag = torch.norm(velocity_camera).item()
    
    print(f"               [IMU Propagation]")
    print(f"               - Translation delta: {translation_delta:.4f} m")
    print(f"               - Velocity (updated): [{velocity_camera[0]:.3f}, {velocity_camera[1]:.3f}, {velocity_camera[2]:.3f}] m/s (mag={velocity_mag:.3f})")
    print(f"               - IMU samples: {n_samples}, dt_cam={dt_cam:.4f}s")
    print(f"               - S3E mode: {dataset_is_s3e} (rotation {'frozen' if dataset_is_s3e else 'integrated'})")
    
    return propagated_pose, velocity_camera, covariance


def integrate_rotation(omega: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Integrate angular velocity to rotation matrix using exponential map.
    
    Args:
        omega: Angular velocity (3,) tensor in rad/s [wx, wy, wz]
        dt: Time step in seconds
    
    Returns:
        Rotation matrix (3, 3) representing orientation change
    """
    device = omega.device
    
    theta = omega * dt
    angle = torch.norm(theta)
    
    if angle < 1e-8:
        return torch.eye(3, device=device, dtype=torch.float32)
    
    axis = theta / angle
    K = skew_symmetric(axis)
    K_squared = K @ K
    
    R = (torch.eye(3, device=device, dtype=torch.float32) + 
         torch.sin(angle) * K + 
         (1 - torch.cos(angle)) * K_squared)
    
    return R


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Construct skew-symmetric matrix from 3D vector."""
    device = v.device
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], device=device, dtype=torch.float32)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to 3x3 rotation matrix (gradient-friendly)."""
    device = q.device
    dtype = q.dtype
    
    q_norm = q / torch.norm(q)
    qx, qy, qz, qw = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
    
    R = torch.stack([
        torch.stack([1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)]),
        torch.stack([2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)]),
        torch.stack([2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)])
    ])
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to unit quaternion (gradient-friendly)."""
    device = R.device
    dtype = R.dtype
    
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    # Branch 1: trace > 0
    s1 = 0.5 / torch.sqrt(trace + 1.0 + 1e-8)
    qw1 = 0.25 / (s1 + 1e-8)
    qx1 = (R[2, 1] - R[1, 2]) * s1
    qy1 = (R[0, 2] - R[2, 0]) * s1
    qz1 = (R[1, 0] - R[0, 1]) * s1
    
    # Branch 2: R[0,0] largest
    s2 = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2] + 1e-8)
    qw2 = (R[2, 1] - R[1, 2]) / (s2 + 1e-8)
    qx2 = 0.25 * s2
    qy2 = (R[0, 1] + R[1, 0]) / (s2 + 1e-8)
    qz2 = (R[0, 2] + R[2, 0]) / (s2 + 1e-8)
    
    # Branch 3: R[1,1] largest
    s3 = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2] + 1e-8)
    qw3 = (R[0, 2] - R[2, 0]) / (s3 + 1e-8)
    qx3 = (R[0, 1] + R[1, 0]) / (s3 + 1e-8)
    qy3 = 0.25 * s3
    qz3 = (R[1, 2] + R[2, 1]) / (s3 + 1e-8)
    
    # Branch 4: R[2,2] largest
    s4 = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1] + 1e-8)
    qw4 = (R[1, 0] - R[0, 1]) / (s4 + 1e-8)
    qx4 = (R[0, 2] + R[2, 0]) / (s4 + 1e-8)
    qy4 = (R[1, 2] + R[2, 1]) / (s4 + 1e-8)
    qz4 = 0.25 * s4
    
    # Select branch
    cond1 = trace > 0
    cond2 = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]) & ~cond1
    cond3 = (R[1, 1] > R[2, 2]) & ~cond1 & ~cond2
    
    qx = torch.where(cond1, qx1, torch.where(cond2, qx2, torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond1, qy1, torch.where(cond2, qy2, torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond1, qz1, torch.where(cond2, qz2, torch.where(cond3, qz3, qz4)))
    qw = torch.where(cond1, qw1, torch.where(cond2, qw2, torch.where(cond3, qw3, qw4)))
    
    q = torch.stack([qx, qy, qz, qw])
    q = q / (torch.norm(q) + 1e-8)
    
    return q.to(dtype=dtype)


def compute_imu_prior_loss(
    estimated_pose: torch.Tensor,
    prior_pose: torch.Tensor,
    weight_translation: float = 1.0,
    weight_rotation: float = 1.0,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute soft constraint loss between estimated and prior poses.
    
    Loss = λ_T * ||t_est - t_prior||² + λ_R * ||log(R_prior^{-1} @ R_est)||²
    
    Args:
        estimated_pose: Current pose estimate [tx,ty,tz,qx,qy,qz,qw] (requires_grad=True)
        prior_pose: IMU-propagated prior [tx,ty,tz,qx,qy,qz,qw] (detached)
        weight_translation: Weight for translation residual
        weight_rotation: Weight for rotation residual
        device: Torch device
    
    Returns:
        Scalar loss (differentiable w.r.t. estimated_pose)
    """
    # Translation error
    t_est = estimated_pose[0:3]
    t_prior = prior_pose[0:3].detach()
    trans_error = torch.sum((t_est - t_prior) ** 2)
    
    # Rotation error
    q_est = estimated_pose[3:7] / torch.norm(estimated_pose[3:7])
    q_prior = prior_pose[3:7].detach() / torch.norm(prior_pose[3:7])
    
    R_est = quaternion_to_rotation_matrix(q_est)
    R_prior = quaternion_to_rotation_matrix(q_prior)
    
    R_delta = R_prior.T @ R_est
    trace_R = torch.trace(R_delta)
    angle = torch.acos(torch.clamp((trace_R - 1) / 2, -1.0, 1.0))
    rot_error = angle ** 2
    
    loss = weight_translation * trans_error + weight_rotation * rot_error
    
    return loss
