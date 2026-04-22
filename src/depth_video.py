import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F

from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.utils.dyn_uncertainty import mapping_utils as map_utils

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer, uncer_network=None):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # To save gpu ram, we put images to cpu as it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device='cpu', dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_mask_up = torch.ones(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.printer = printer
        
        # IMU-related buffers (Stage 2: Shared-State Plumbing)
        self.imu_enabled = cfg.get('imu', {}).get('enabled', False)
        self.use_in_ba = False
        if self.imu_enabled:
            # Check if use_in_ba is set 
            self.use_in_ba = cfg.get('imu', {}).get('use_in_ba', False)
        if self.imu_enabled:
            # IMU data storage: variable-length slicing per frame
            # Max IMU samples: assume 10x camera rate * buffer size (e.g., 100Hz IMU, 10Hz camera)
            max_imu_samples = buffer * 50  # Conservative estimate
            
            # Per-frame IMU indexing
            self.imu_offsets = torch.zeros(buffer, device=self.device, dtype=torch.long).share_memory_()
            self.imu_counts = torch.zeros(buffer, device=self.device, dtype=torch.long).share_memory_()
            
            # Global IMU data buffer: [N_samples, 7] = [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            self.imu_data = torch.zeros(max_imu_samples, 7, device=self.device, dtype=torch.float).share_memory_()
            self.imu_data_counter = Value('i', 0)  # Track next free slot in imu_data
            
            # IMU prior storage: propagated poses from IMU preintegration
            self.imu_prior_poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()  # quat + trans
            self.imu_prior_cov = torch.zeros(buffer, 6, 6, device=self.device, dtype=torch.float).share_memory_()  # 6-DOF covariance
            
            # PERSISTENT VELOCITY TRACKING (NEW APPROACH)
            # Maintain velocity estimate across ALL frames (not just keyframes)
            # Updated EVERY frame using accelerometer data, provides continuous velocity tracking
            # Format: [vx, vy, vz] in camera frame (m/s)
            self.imu_velocities = torch.zeros(buffer, 3, device=self.device, dtype=torch.float).share_memory_()
            
            # Track when velocity was last updated (for sanity checking)
            self.imu_velocity_timestamps = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
            
            # Velocity state management
            self.current_velocity_idx = Value('i', -1)  # Index of most recent velocity update
            self.velocity_initialized = Value('i', 0)  # 0=not initialized, 1=initialized
            
            # Camera-to-IMU transform (4x4 SE(3) matrix, shared across all frames)
            self.c2i_transform = torch.eye(4, device=self.device, dtype=torch.float).share_memory_()
            
            self.printer.print(f"IMU buffers allocated: {max_imu_samples} samples, {buffer} frames", FontColor.INFO)
            self.printer.print(f"Velocity tracking: ENABLED (continuous update across all frames)", FontColor.INFO)
        else:
            self.imu_offsets = None
            self.imu_counts = None
            self.imu_data = None
            self.imu_data_counter = None
            self.imu_prior_poses = None
            self.imu_prior_cov = None
            self.imu_velocities = None
            self.imu_velocity_timestamps = None
            self.current_velocity_idx = None
            self.velocity_initialized = None
            self.c2i_transform = None
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.uncer_network = uncer_network
        if self.uncertainty_aware:
            n_features = self.cfg["mapping"]["uncertainty_params"]['feature_dim']
            
            # This check is to ensure the size of self.dino_feats
            if self.cfg["mono_prior"]["feature_extractor"] not in ["dinov2_reg_small_fine", "dinov2_small_fine","dinov2_vits14", "dinov2_vits14_reg"]:
                raise ValueError("You are using a new feature extractor, make sure the downsample factor is 14")
            
            # The followings are in cpu to save memory
            self.dino_feats = torch.zeros(buffer, ht//14, wd//14, n_features, device='cpu', dtype=torch.float).share_memory_()
            self.dino_feats_resize = torch.zeros(buffer, n_features, ht//self.down_scale, wd//self.down_scale, device='cpu', dtype=torch.float).share_memory_()
            self.uncertainties_inv = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        else:
            self.dino_feats = None
            self.dino_feats_resize = None
        
        # MambaVision descriptor storage (Stage 5: Vision-Mamba Descriptor Pipeline)
        self.use_mamba_descriptors = cfg.get('loop_closure', {}).get('use_mamba_descriptors', False)
        if self.use_mamba_descriptors:
            self.descriptor_dim = 640  # MambaVision-T output dimension
            
            # Descriptor storage: [buffer, 640] in CPU to save GPU memory
            # Will be copied to GPU for FAISS operations
            self.descriptors = torch.zeros(buffer, self.descriptor_dim, device='cpu', dtype=torch.float32).share_memory_()
            
            # Validity flags: Track which frames have descriptors
            self.descriptor_valid = torch.zeros(buffer, device='cpu', dtype=torch.bool).share_memory_()
            
            # FAISS index management (initialized lazily in main process)
            self.faiss_index = None
            self.faiss_frame_ids = []  # Map FAISS index -> frame index
            
            # Loop closure configuration (stored as attributes for factor_graph access)
            self.loop_topk = cfg.get('loop_closure', {}).get('topk', 20)
            self.loop_min_temporal_dist = cfg.get('loop_closure', {}).get('min_temporal_distance', 30)
            self.loop_descriptor_threshold = cfg.get('loop_closure', {}).get('descriptor_threshold', 0.7)
            
            self.printer.print(f"MambaVision descriptor buffers allocated: {buffer} frames × {self.descriptor_dim}D", FontColor.INFO)
            self.printer.print(f"  - FAISS top-K: {self.loop_topk}", FontColor.INFO)
            self.printer.print(f"  - Descriptor threshold: {self.loop_descriptor_threshold:.2f}", FontColor.INFO)
            self.printer.print(f"  - Min temporal distance: {self.loop_min_temporal_dist} frames", FontColor.INFO)

            # Local feature storage for two-stage reranking (MambaVision Stage 3 V-features)
            self.use_local_reranking = cfg.get('loop_closure', {}).get('enable_reranking', False)
            if self.use_local_reranking:
                self.local_feature_dim = 320  # MambaVision Stage 3: 8 heads × 40 dim
                self.max_keypoints = cfg.get('loop_closure', {}).get('max_keypoints_per_frame', 100)

                # Local feature storage: [buffer, max_kp, 320] in CPU
                self.local_features = torch.zeros(
                    buffer, self.max_keypoints, self.local_feature_dim,
                    device='cpu', dtype=torch.float32
                ).share_memory_()

                # Track actual keypoint count per frame
                self.local_feature_counts = torch.zeros(buffer, device='cpu', dtype=torch.int32).share_memory_()

                # Validity mask for local features
                self.local_valid = torch.zeros(buffer, device='cpu', dtype=torch.bool).share_memory_()

                # Reranking parameters
                self.rerank_t2 = cfg.get('loop_closure', {}).get('rerank_threshold_t2', 0.05)
                self.rerank_global_weight = cfg.get('loop_closure', {}).get('rerank_global_weight', 0.5)

                mem_mb = (buffer * self.max_keypoints * self.local_feature_dim * 4) / (1024 * 1024)
                self.printer.print(f"  - Local reranking: ENABLED ({mem_mb:.1f} MB)", FontColor.INFO)
                self.printer.print(f"    - MNN threshold (t2): {self.rerank_t2}", FontColor.INFO)
                self.printer.print(f"    - Global weight: {self.rerank_global_weight}", FontColor.INFO)
                self.printer.print(f"    - Max keypoints/frame: {self.max_keypoints}", FontColor.INFO)
            else:
                self.use_local_reranking = False
                self.local_features = None
                self.local_feature_counts = None
                self.local_valid = None
        else:
            self.descriptors = None
            self.descriptor_valid = None
            self.faiss_index = None
            self.faiss_frame_ids = []
            # Local reranking also disabled when descriptors are disabled
            self.use_local_reranking = False
            self.local_features = None
            self.local_feature_counts = None
            self.local_valid = None

    def get_lock(self):
        return self.counter.get_lock()

    def set_c2i_transform(self, c2i_matrix):
        """
        Store camera-to-IMU transformation matrix (Stage 2: IMU Integration)
        
        Args:
            c2i_matrix (torch.Tensor): 4x4 SE(3) transformation matrix from camera to IMU frame
        """
        if not self.imu_enabled:
            raise RuntimeError("Cannot set c2i_transform when IMU is not enabled")
        
        if not isinstance(c2i_matrix, torch.Tensor):
            c2i_matrix = torch.tensor(c2i_matrix, dtype=torch.float, device=self.device)
        else:
            c2i_matrix = c2i_matrix.to(self.device)
        
        # Validate SE(3) properties
        assert c2i_matrix.shape == (4, 4), f"Expected 4x4 matrix, got {c2i_matrix.shape}"
        
        # Check rotation matrix properties (orthogonality and determinant = 1)
        R = c2i_matrix[:3, :3]
        det = torch.det(R)
        orthogonal = torch.allclose(R @ R.T, torch.eye(3, device=self.device), atol=1e-5)
        
        if not orthogonal or not torch.allclose(det, torch.tensor(1.0, device=self.device), atol=1e-5):
            self.printer.print(f"WARNING: c2i_transform may not be a valid SE(3) matrix", FontColor.WARNING)
            self.printer.print(f"  Determinant: {det.item():.6f}, Orthogonal: {orthogonal}", FontColor.WARNING)
        
        with self.get_lock():
            self.c2i_transform[:] = c2i_matrix
        
        self.printer.print(f"Camera-to-IMU transform stored successfully", FontColor.INFO)

    def get_imu_chunk(self, frame_idx):
        """
        Retrieve IMU measurements for a specific frame (Stage 2: IMU Integration)
        
        Args:
            frame_idx (int): Frame index
        
        Returns:
            dict: IMU data with keys:
                - 'timestamps': [N] timestamps
                - 'angular_velocity': [N, 3] gyroscope readings (rad/s)
                - 'linear_acceleration': [N, 3] accelerometer readings (m/s²)
                - 'c2i_transform': [4, 4] camera-to-IMU transformation
            Returns None if IMU not enabled or no data for this frame
        """
        if not self.imu_enabled:
            return None
        
        with self.get_lock():
            # Handle negative indexing
            if frame_idx < 0:
                frame_idx = self.counter.value + frame_idx
            
            # Boundary check
            if frame_idx < 0 or frame_idx >= self.counter.value:
                return None
            
            count = self.imu_counts[frame_idx].item()
            if count == 0:
                return None
            
            offset = self.imu_offsets[frame_idx].item()
            
            # Extract slice from global buffer
            imu_slice = self.imu_data[offset:offset+count].clone()  # [N, 7]
            
            return {
                'timestamps': imu_slice[:, 0],
                'linear_acceleration': imu_slice[:, 1:4],
                'angular_velocity': imu_slice[:, 4:7],
                'c2i_transform': self.c2i_transform.clone()
            }

    def set_imu_prior(self, frame_idx, pose, covariance=None):
        """
        Store propagated IMU pose prior for a frame (Stage 2: IMU Integration)
        
        Args:
            frame_idx (int): Frame index
            pose (torch.Tensor): 7D pose (quaternion + translation) or 4x4 SE(3) matrix
            covariance (torch.Tensor, optional): 6x6 pose covariance matrix. If None, uses identity.
        """
        if not self.imu_enabled:
            raise RuntimeError("Cannot set IMU prior when IMU is not enabled")
        
        # Convert pose to 7D quaternion + translation format if needed
        if pose.shape == (4, 4):
            # Convert SE(3) matrix to quaternion + translation
            from lietorch import SE3
            se3 = SE3(pose.unsqueeze(0))
            pose_7d = se3.data.squeeze()  # [7] quat + trans
        elif pose.shape == (7,):
            pose_7d = pose
        else:
            raise ValueError(f"Invalid pose shape: {pose.shape}. Expected (7,) or (4, 4)")
        
        # Default covariance if not provided
        if covariance is None:
            covariance = torch.eye(6, dtype=torch.float, device=self.device)
        
        assert covariance.shape == (6, 6), f"Expected 6x6 covariance, got {covariance.shape}"
        
        # Bound check against imu_prior buffers to avoid silent out-of-range writes
        buffer_size = self.imu_prior_poses.shape[0]
        if frame_idx < 0 or frame_idx >= buffer_size:
            raise IndexError(f"set_imu_prior: frame_idx {frame_idx} out of bounds [0, {buffer_size})")

        with self.get_lock():
            self.imu_prior_poses[frame_idx] = pose_7d.to(self.device)
            self.imu_prior_cov[frame_idx] = covariance.to(self.device)

    def get_imu_prior(self, frame_idx):
        """
        Retrieve propagated IMU pose prior for a frame (Stage 2: IMU Integration)
        
        Args:
            frame_idx (int): Frame index
        
        Returns:
            tuple: (pose, covariance) where:
                - pose: [7] quaternion + translation
                - covariance: [6, 6] pose covariance
            Returns (None, None) if IMU not enabled or frame has no prior
        """
        if not self.imu_enabled:
            return None, None
        
        with self.get_lock():
            # Handle negative indexing (relative to current counter)
            if frame_idx < 0:
                frame_idx = self.counter.value + frame_idx

            # Boundary check against IMU prior buffer size (allow priors reserved for upcoming frames)
            buffer_size = self.imu_prior_poses.shape[0]
            if frame_idx < 0 or frame_idx >= buffer_size:
                return None, None
            
            # CRITICAL: Detach prior to prevent gradients flowing back through IMU propagation
            pose = self.imu_prior_poses[frame_idx].clone().detach()
            cov = self.imu_prior_cov[frame_idx].clone().detach()
            
            # Check if prior has been set (non-zero pose)
            if torch.allclose(pose, torch.zeros_like(pose)):
                return None, None
            
            return pose, cov

    def get_descriptor(self, frame_idx):
        """
        Retrieve descriptor for a frame.

        Args:
            frame_idx (int): Frame index

        Returns:
            torch.Tensor or None: Descriptor [640] if available, else None
        """
        if not self.use_mamba_descriptors:
            return None

        if not self.descriptor_valid[frame_idx]:
            return None
        
        return self.descriptors[frame_idx].clone()

    def query_loop_candidates(self, frame_idx, topk=10, min_temporal_distance=20, max_candidates=50):
        """
        Query FAISS index for loop closure candidates (Stage 6).
        Args:
            frame_idx (int): Query frame index
            topk (int): Number of nearest neighbors to retrieve
            min_temporal_distance (int): Minimum frame separation (avoid local matches)
            max_candidates (int): Maximum candidates to return
        
        Returns:
            List[tuple]: List of (candidate_frame_idx, similarity_score) sorted by similarity
        """
        if not self.use_mamba_descriptors or not self.descriptor_valid[frame_idx]:
            return []

        if self.faiss_index is None or len(self.faiss_frame_ids) == 0:
            return []
        
        # Get query descriptor
        query_desc = self.descriptors[frame_idx].numpy().reshape(1, -1).astype(np.float32)
        
        # Search FAISS index (returns similarities and indices)
        similarities, indices = self.faiss_index.search(query_desc, min(topk, len(self.faiss_frame_ids)))
        
        # Convert FAISS indices to frame indices and filter
        candidates = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_frame_ids):
                continue
            
            candidate_frame = self.faiss_frame_ids[idx]
            
            # Filter out temporal neighbors
            if abs(candidate_frame - frame_idx) < min_temporal_distance:
                continue
            
            candidates.append((candidate_frame, float(sim)))
            
            if len(candidates) >= max_candidates:
                break
        
        return candidates

    def set_descriptor(self, frame_idx, descriptor):
        """
        Store MambaVision descriptor for a keyframe (Stage 5: Descriptor Pipeline)
        Args:
            frame_idx (int): Frame index
            descriptor (torch.Tensor): 640-D descriptor tensor (L2-normalized)
        """

        if not self.use_mamba_descriptors:
            return
        
        if not isinstance(descriptor, torch.Tensor):
            descriptor = torch.tensor(descriptor, dtype=torch.float32)
        
        descriptor = descriptor.cpu()  # Store in CPU to save GPU memory
        
        assert descriptor.shape[-1] == self.descriptor_dim, \
            f"Expected descriptor dim {self.descriptor_dim}, got {descriptor.shape[-1]}"
        
        with self.get_lock():
            self.descriptors[frame_idx] = descriptor.flatten()
            self.descriptor_valid[frame_idx] = True

    def set_local_features(self, frame_idx: int, features: 'torch.Tensor'):
        """
        Store local features for a keyframe (Stage 2 reranking).

        Args:
            frame_idx (int): Frame index
            features (torch.Tensor): [num_keypoints, 320] local feature tensor
        """
        if not self.use_local_reranking:
            return

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        features = features.cpu()  # Store in CPU

        num_kp = features.shape[0]
        if num_kp > self.max_keypoints:
            # Truncate if too many keypoints
            features = features[:self.max_keypoints]
            num_kp = self.max_keypoints

        with self.get_lock():
            self.local_features[frame_idx, :num_kp] = features
            self.local_feature_counts[frame_idx] = num_kp
            self.local_valid[frame_idx] = True

    def get_local_features(self, frame_idx: int) -> 'torch.Tensor':
        """
        Retrieve local features for a keyframe.

        Args:
            frame_idx (int): Frame index

        Returns:
            torch.Tensor: [num_keypoints, 320] local features, or empty tensor if not valid
        """
        if not self.use_local_reranking or not self.local_valid[frame_idx]:
            return torch.empty(0, self.local_feature_dim if hasattr(self, 'local_feature_dim') else 320)

        num_kp = int(self.local_feature_counts[frame_idx].item())
        return self.local_features[frame_idx, :num_kp]

    def get_local_features_batch(self, frame_indices: list) -> list:
        """
        Retrieve local features for multiple frames (for batch reranking).

        Args:
            frame_indices (list): List of frame indices

        Returns:
            list: List of numpy arrays, one per frame [num_keypoints, 320]
        """
        import numpy as np

        if not self.use_local_reranking:
            return [np.empty((0, 320), dtype=np.float32) for _ in frame_indices]

        result = []
        for idx in frame_indices:
            if self.local_valid[idx]:
                num_kp = int(self.local_feature_counts[idx].item())
                feats = self.local_features[idx, :num_kp].numpy()
            else:
                feats = np.empty((0, self.local_feature_dim), dtype=np.float32)
            result.append(feats)

        return result


    def get_velocity(self, frame_idx: int):
        """
        Retrieve velocity estimate for a specific frame.
        
        Returns:
            torch.Tensor: 3D velocity [vx, vy, vz] in camera frame (m/s), or None if not available
        """
        if not self.imu_enabled:
            return None
        
        buffer_size = self.imu_velocities.shape[0]

        if 0 <= frame_idx < buffer_size:
            velocity = self.imu_velocities[frame_idx].clone().detach()
            timestamp = self.imu_velocity_timestamps[frame_idx].item()
            
            # Check if velocity was ever set (timestamp > 0)
            if timestamp > 0:
                return velocity
            else:
                return None
        else:
            return None
    
    def set_velocity(self, frame_idx: int, velocity: torch.Tensor, timestamp: float):
        """
        Store velocity estimate for a specific frame.
        
        Args:
            frame_idx: Frame index
            velocity: 3D velocity [vx, vy, vz] in camera frame (m/s)
            timestamp: Timestamp when velocity was computed
        """
        if not self.imu_enabled:
            raise RuntimeError("Cannot set velocity when IMU is not enabled")
        
        buffer_size = self.imu_velocities.shape[0]
        
        if frame_idx < 0 or frame_idx >= buffer_size:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {buffer_size})")
        
        self.imu_velocities[frame_idx] = velocity.to(self.device)
        self.imu_velocity_timestamps[frame_idx] = timestamp
        
        # Update current velocity index
        with self.current_velocity_idx.get_lock():
            self.current_velocity_idx.value = frame_idx
        
        # Mark velocity as initialized
        if self.velocity_initialized.value == 0:
            with self.velocity_initialized.get_lock():
                self.velocity_initialized.value = 1

    def initialize_faiss_index(self, device='cuda:0'):
        """
        Initialize FAISS index for fast descriptor search (Stage 5).
        Called lazily when first descriptor is added.
        
        Args:
            device: Device for FAISS operations ('cuda:0' or 'cpu')
        """
        if not self.use_mamba_descriptors:
            return
        
        if self.faiss_index is not None:
            return  # Already initialized
        
        try:
            import faiss
        except ImportError:
            raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        # This is the simplest and most accurate for small-to-medium databases (<10k keyframes)
        self.faiss_index = faiss.IndexFlatIP(self.descriptor_dim)
        
        # Move to GPU if available and requested
        if 'cuda' in device and faiss.get_num_gpus() > 0:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, self.faiss_index)
            self.printer.print(f"FAISS index initialized on GPU {gpu_id}", FontColor.INFO)
        else:
            self.printer.print(f"FAISS index initialized on CPU", FontColor.INFO)
        
        self.faiss_frame_ids = []

    def add_descriptor_to_index(self, frame_idx):
        """
        Add a frame's descriptor to the FAISS index (Stage 5).
        
        Args:
            frame_idx (int): Frame index with stored descriptor
        
        Returns:
            bool: True if added successfully, False otherwise
        """
        if not self.use_mamba_descriptors or not self.descriptor_valid[frame_idx]:
            return False
        
        # Lazy initialization
        if self.faiss_index is None:
            self.initialize_faiss_index(device=str(self.device))
        
        descriptor = self.descriptors[frame_idx].numpy().reshape(1, -1)
        
        # Verify normalization (FAISS IndexFlatIP requires normalized vectors for cosine similarity)
        norm = np.linalg.norm(descriptor)
        if not np.isclose(norm, 1.0, atol=1e-4):
            self.printer.print(f"WARNING: Descriptor for frame {frame_idx} not L2-normalized (norm={norm:.4f})", 
                             FontColor.WARNING)
            # Normalize it
            descriptor = descriptor / (norm + 1e-8)
        
        self.faiss_index.add(descriptor.astype(np.float32))
        self.faiss_frame_ids.append(frame_idx)
        
        return True

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1].cpu()

        if item[2] is not None:
            self.poses[index] = item[2]
            # CRITICAL: Initialize visual_poses to match poses (prevents NaN)
            if self.imu_enabled and hasattr(self, 'visual_poses'):
                self.visual_poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            mono_depth = item[4][self.slice_h,self.slice_w]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.mono_disps_up[index] = torch.where(item[4]>0, 1.0/item[4], 0)
            # self.disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.dino_feats[index] = item[9].cpu()

            if len(item[9].shape) == 3:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(2,0,1).unsqueeze(0),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear').squeeze()[:,self.slice_h,self.slice_w].cpu()
            else:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(0,3,1,2),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear')[:,:,self.slice_h,self.slice_w].cpu()

        # Store IMU data (Stage 2: IMU Integration)
        if len(item) > 10 and item[10] is not None and self.imu_enabled:
            imu_dict = item[10]
            
            # Extract IMU data: timestamps, angular_velocity, linear_acceleration
            timestamps = imu_dict['timestamps']  # [N]
            ang_vel = imu_dict['angular_velocity']  # [N, 3]
            lin_acc = imu_dict['linear_acceleration']  # [N, 3]
            
            n_samples = timestamps.shape[0]
            
            # Get current IMU data counter value
            start_idx = self.imu_data_counter.value
            end_idx = start_idx + n_samples
            
            # Check buffer capacity
            if end_idx > self.imu_data.shape[0]:
                self.printer.print(f"WARNING: IMU buffer overflow at frame {index}. Increase max_imu_samples.", FontColor.WARNING)
                n_samples = max(0, self.imu_data.shape[0] - start_idx)
                end_idx = start_idx + n_samples
            
            if n_samples > 0:
                # Pack IMU data: [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                imu_samples = torch.zeros(n_samples, 7, dtype=torch.float, device=self.device)
                imu_samples[:, 0] = timestamps[:n_samples].to(self.device)
                imu_samples[:, 1:4] = lin_acc[:n_samples].to(self.device)  # accel
                imu_samples[:, 4:7] = ang_vel[:n_samples].to(self.device)  # gyro
                
                # Store in global buffer
                self.imu_data[start_idx:end_idx] = imu_samples
                
                # Update per-frame indexing
                self.imu_offsets[index] = start_idx
                self.imu_counts[index] = n_samples
                
                # Update global counter
                self.imu_data_counter.value = end_idx

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_images_with_mask(self, images, pixel_positions, masks=None):
        """ 
            Project images/depths from the input pixel positions using bilinear interpolation.
            This function will automatically return the mask where the given pixel positions are out of the images
        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing the images/depths.
            pixel_positions (torch.Tensor): A tensor of shape [B, H, W, 2] containing float 
                                            pixel positions for interpolation. Note that [:,:,:,0]
                                            is width and [:,:,:,1] is height.
            masks (torch.Tensor, optional): A boolean tensor of shape [B, H, W]. If provided, 
                                            specifies valid pixels. Default is None, which 
                                            results in all pixels being valid at the begining.
        
        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] containing the projected images/depths, 
                        where invalid pixels are set to 0.
            torch.Tensor: The combined mask that filters out invalid positions and applies
                      the original mask.
        """
        B, C, H, W = images.shape
        device = images.device

        # If masks are not provided, create a mask of all ones (True) with the same shape as the images
        if masks is None:
            masks = torch.ones(B, H, W, dtype=torch.bool, device=device)
        
        # Normalize pixel positions to range [-1, 1]
        grid = pixel_positions.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] / (H - 1)) - 1.0

        projected_image = F.grid_sample(images, grid, mode='bilinear', align_corners=True)

        # Mask out invalid positions where x or y are out of bounds and combine it with the initial mask
        valid_mask = (pixel_positions[..., 0] >= 0) & (pixel_positions[..., 0] < W) & \
                    (pixel_positions[..., 1] >= 0) & (pixel_positions[..., 1] < H)
        valid_mask &= masks

        # Apply the combined mask: set to 0 where combined mask is False
        projected_image = projected_image.permute(0, 2, 3, 1)  # conver to [B, H, W, C]
        projected_image = projected_image * valid_mask.unsqueeze(-1)
        
        return projected_image.permute(0, 3, 1, 2), valid_mask  # Return to [B, C, H, W]

    @torch.no_grad()
    def filter_high_err_mono_depth(self, idx, ii, jj):
        nb_frame = self.cfg['tracking']['nb_ref_frame_metric_depth_filtering']

        jj = jj[ii==idx]
        for j in torch.arange(idx-1, max(0,idx-nb_frame)-1, -1):
            if jj.shape[0] >= nb_frame:
                break
            if j not in jj:
                torch.cat((jj, j.unsqueeze(0).to(jj.device)))

        ii = torch.tensor(idx).repeat(jj.shape[0])

        # all frames share the same intrinsics
        X0, _ = pops.iproj(self.mono_disps_up[jj].unsqueeze(0), 
                      self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, 
                      jacobian=False)
        Gs = lietorch.SE3(self.poses[None])
        Gji = Gs[:,ii] * Gs[:,jj].inv()
        X1, _ = pops.actp(Gji, X0, jacobian=False)
        x1, _ = pops.proj(X1, self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, jacobian=False, return_depth=True)

        i_disp = self.mono_disps_up[idx]
        accurate_count = torch.zeros_like(i_disp)
        inaccurate_count = torch.zeros_like(i_disp)

        x1_rounded = torch.round(x1[..., :2]).long()
        # x1 is the 3d poisition (x,y,z)
        # projected point is valid only if its inside the image range and the depth is greater than 0
        valid_mask = (x1_rounded[..., 1] >= 0) & (x1_rounded[..., 1] < x1.shape[2]) & \
                    (x1_rounded[..., 0] >= 0) & (x1_rounded[..., 0] < x1.shape[3]) & (x1[...,2]>0)
        
        i_dino = F.interpolate(self.dino_feats[idx].permute(2,0,1).unsqueeze(0),
                                self.disps_up.shape[-2:], 
                                mode='bilinear').to(self.device).squeeze()
        for j_id in range(jj.shape[0]):
            projected_j_to_i = x1[0, j_id]
            x_coords, y_coords = x1_rounded[0, j_id, ..., 0], x1_rounded[0, j_id, ..., 1]
            
            # Select valid coordinates and their Dino features
            j_dino = F.interpolate(self.dino_feats[jj[j_id]].permute(2,0,1).unsqueeze(0),
                                    self.disps_up.shape[-2:], 
                                    mode='bilinear').to(self.device).squeeze()
            valid_x, valid_y = x_coords[valid_mask[0, j_id]], y_coords[valid_mask[0, j_id]]
            j_dino_valid = j_dino[:, valid_mask[0, j_id]]
            i_dino_valid = i_dino[:, valid_y, valid_x]

            # Compute cosine similarity for each valid position
            similarity = F.normalize(j_dino_valid, p=2, dim=0).mul(F.normalize(i_dino_valid, p=2, dim=0)).sum(dim=0)
            matching_mask = similarity > 0.9

            # Update projected disparity and counts based on the similarity check
            j_projected_disp = torch.zeros_like(self.mono_disps_up[idx])
            matched_disp = projected_j_to_i[valid_mask[0, j_id]][matching_mask]
            matched_x, matched_y = valid_x[matching_mask], valid_y[matching_mask]
            j_projected_disp[matched_y, matched_x] = matched_disp[..., 2]

            # Error calculation and count updates
            error = torch.abs(1 / j_projected_disp[matched_y, matched_x] - 1 / i_disp[matched_y, matched_x]) * j_projected_disp[matched_y, matched_x]
            correct_mask = error < 0.02

            # Batch update correct and bad counts
            accurate_count[matched_y[correct_mask], matched_x[correct_mask]] += 1
            inaccurate_count[matched_y[~correct_mask], matched_x[~correct_mask]] += 1

        # Clean the gpu memory
        torch.cuda.empty_cache()

        self.mono_disps_mask_up[idx][(accurate_count<=1)&(inaccurate_count>0)&(self.mono_disps_up[idx]>0)] = False

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1,
           motion_only=False):
        if self.uncertainty_aware:
            weight *= self.uncertainties_inv[ii][None, :, :, :, None]

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()

            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            else:
                mono_valid_mask = self.mono_disps_mask_up[:,self.slice_h,self.slice_w].clone().to(self.device)
                
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps*mono_valid_mask,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            
            self.disps.clamp_(min=1e-5)
            
            # REMOVED: visual_poses saving logic - we now use BA poses directly with velocity clamping
            # The velocity clamping in imu_utils.py handles BA correction artifacts
            
            # Apply IMU prior soft constraints (Stage 3 Integration)
            if self.imu_enabled and hasattr(self, 'cfg') and self.use_in_ba:
                weight_T = self.cfg.get('imu', {}).get('prior_weight_T', 0.0)
                weight_R = self.cfg.get('imu', {}).get('prior_weight_R', 0.0)
                
                if weight_T > 0 or weight_R > 0:
                    print(f"[DepthVideo] BA [{t0}:{t1}]: Applying IMU prior constraints")
                    self._apply_imu_prior_constraints(t0, t1, weight_T, weight_R)
                elif self.imu_enabled:
                    print(f"[DepthVideo] BA [{t0}:{t1}]: IMU enabled but prior weights are zero (constraints disabled)")
    
    def _apply_imu_prior_constraints(self, t0, t1, weight_T, weight_R, lr=0.001, iters=10):
        """
        Apply IMU prior soft constraints to poses after CUDA BA.
        
        This refines poses by minimizing:
        L_IMU = λ_T * ||t_est - t_prior||² + λ_R * ||log(R_prior^{-1} @ R_est)||²
        
        Args:
            t0: Start frame index for BA window
            t1: End frame index for BA window  
            weight_T: Translation prior weight (λ_T)
            weight_R: Rotation prior weight (λ_R)
            lr: Learning rate for gradient descent
            iters: Number of refinement iterations
        """
        from src.utils.imu_utils import compute_imu_prior_loss
        
        # Collect frames with IMU priors in BA window
        frames_with_priors = []
        for idx in range(t0, min(t1, self.counter.value)):
            prior_pose, prior_cov = self.get_imu_prior(idx)
            if prior_pose is not None:
                frames_with_priors.append((idx, prior_pose, prior_cov))
        
        if len(frames_with_priors) == 0:
            print(f"             ✗ No IMU priors found in window [{t0}:{t1}]")
            return  # No priors to apply
        
        print(f"             ✓ Found {len(frames_with_priors)} frames with IMU priors")
        
        # Make poses differentiable for gradient computation
        original_poses = self.poses.clone()
        initial_losses = []
        
        for iteration in range(iters):
            total_loss = 0.0
            pose_gradients = torch.zeros_like(self.poses)
            
            for idx, prior_pose, prior_cov in frames_with_priors:
                # Get current pose estimate - CRITICAL: must enable gradients properly
                # Shared memory tensors require extra care with gradients
                current_pose_data = self.poses[idx].clone()  # First clone from shared memory
                current_pose = current_pose_data.detach().requires_grad_(True)  # Then detach and enable grad
                
                # DEBUG: Check tensor properties
                # if iteration == 0 and idx == frames_with_priors[0][0]:
                #     print(f"             [DEBUG Frame {idx}] current_pose.dtype: {current_pose.dtype}, device: {current_pose.device}")
                #     print(f"             [DEBUG Frame {idx}] current_pose.requires_grad: {current_pose.requires_grad}")
                #     print(f"             [DEBUG Frame {idx}] current_pose.is_leaf: {current_pose.is_leaf}")
                #     print(f"             [DEBUG Frame {idx}] prior_pose.dtype: {prior_pose.dtype}, device: {prior_pose.device}")
                #     print(f"             [DEBUG Frame {idx}] prior_pose.requires_grad: {prior_pose.requires_grad}")
                #     print(f"             [DEBUG Frame {idx}] current_pose values: {current_pose}")
                #     print(f"             [DEBUG Frame {idx}] prior_pose values: {prior_pose}")
                
                # Compute IMU prior loss (current_pose is differentiable, prior_pose is not)
                loss = compute_imu_prior_loss(
                    current_pose, prior_pose,
                    weight_translation=weight_T,
                    weight_rotation=weight_R,
                    device=self.device
                )
                
                # # DEBUG: Check loss properties
                if iteration == 0 and idx == frames_with_priors[0][0]:
                #     print(f"             [DEBUG Frame {idx}] loss.dtype: {loss.dtype}, device: {loss.device}")
                #     print(f"             [DEBUG Frame {idx}] loss.requires_grad: {loss.requires_grad}")
                    print(f"             [DEBUG Frame {idx}] loss.grad_fn: {loss.grad_fn}")
                    print(f"             [DEBUG Frame {idx}] loss.is_leaf: {loss.is_leaf}")
                
                # Compute gradients
                if loss.requires_grad:
                    try:
                        loss.backward()
                        if current_pose.grad is not None:
                            pose_gradients[idx] = current_pose.grad.clone()
                        else:
                            print(f"             ⚠ Frame {idx}: loss.backward() succeeded but grad is None")
                    except Exception as e:
                        print(f"             ✗ Frame {idx}: Gradient computation failed: {e}")
                        pose_gradients[idx] = torch.zeros_like(current_pose)
                else:
                    print(f"             ⚠ Frame {idx}: Loss does not require grad (loss={loss.item():.6f})")
                    
                total_loss += loss.item()
            
            # Store initial loss for comparison
            if iteration == 0:
                initial_total_loss = total_loss
            
            # Apply gradient descent step
            if pose_gradients.abs().sum() > 0:
                self.poses.sub_(lr * pose_gradients)
                
                # Normalize quaternions after update
                # lietorch format: [tx, ty, tz, qx, qy, qz, qw]
                for idx in range(t0, min(t1, self.counter.value)):
                    quat = self.poses[idx, 3:7]  # Get quaternion part
                    self.poses[idx, 3:7] = quat / torch.norm(quat)
            
            # Early stopping if loss is small
            if total_loss < 1e-6 * len(frames_with_priors):
                print(f"             - Iteration {iteration+1}/{iters}: Early stop (loss={total_loss:.6f})")
                break
        
        # Calculate final metrics
        final_total_loss = total_loss
        avg_initial_loss = initial_total_loss / len(frames_with_priors)
        avg_final_loss = final_total_loss / len(frames_with_priors)
        loss_reduction = (initial_total_loss - final_total_loss) / initial_total_loss * 100 if initial_total_loss > 0 else 0
        
        print(f"             ✓ IMU constraints applied: {len(frames_with_priors)} frames, {iteration+1} iters")
        print(f"             - Avg loss: {avg_initial_loss:.6f} → {avg_final_loss:.6f} ({loss_reduction:.1f}% reduction)")

    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            if self.metric_depth_reg:
                est_disp = self.mono_disps_up[index].clone().to(device)  # [h, w]
                est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
                depth_mask = torch.ones_like(est_disp,dtype=torch.bool).to(device)
                c2w = self.get_pose(index,device)
            else:
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    @torch.no_grad()
    def update_all_uncertainty_mask(self):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')
        
        i = 0
        while i*20 < self.counter.value:
            dino_feat_batch = self.dino_feats[i*20:min((i+1)*20,self.counter.value),:,:,:].to(self.device)
            with Lock():
                uncer = self.uncer_network(dino_feat_batch)
            train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']

            h = self.images.shape[2]
            w = self.images.shape[3]
            uncer = torch.clip(uncer, min=0.1) + 1e-3
            uncer = uncer.unsqueeze(1)
            uncer = F.interpolate(uncer, size=(h, w), mode="bilinear").squeeze(1).detach()
            data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
            uncer = uncer[:, self.slice_h, self.slice_w]
            uncer = (uncer - 0.1) * data_rate + 0.1
            self.uncertainties_inv[i*20:min((i+1)*20,self.counter.value),:,:] = torch.clamp(0.5/uncer**2, 0.0, 1.0)

            i += 1

    @torch.no_grad()
    def update_uncertainty_mask_given_index(self,idxs):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')
        
        dino_feat_batch = self.dino_feats[idxs,:,:,:].to(self.device)
        with Lock():
            uncer = self.uncer_network(dino_feat_batch)
        train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']

        h = self.images.shape[2]
        w = self.images.shape[3]
        uncer = torch.clip(uncer, min=0.1) + 1e-3
        uncer = uncer.unsqueeze(1)
        uncer = torch.nn.functional.interpolate(uncer, size=(h, w), mode="bilinear").squeeze(1).detach()
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
        uncer = uncer[:, self.slice_h, self.slice_w]
        uncer = (uncer - 0.1) * data_rate + 0.1
        self.uncertainties_inv[idxs,:,:] = torch.clamp(0.5/uncer**2, 0.0, 1.0)

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)


    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        """This is from splat-slam, not used in WildGS-SLAM
        """
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()