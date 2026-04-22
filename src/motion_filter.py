import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor
# V2: Use new continuous velocity tracking approach
# from src.utils.imu_utils_v2 import propagate_imu_continuous, is_s3e_dataset
from src.utils.imu_utils_updated import propagate_imu_continuous, is_s3e_dataset

# Stage 5: MambaVision descriptor extraction
from src.modules.mamba_descriptor import MambaDescriptorExtractor

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.metric_depth_estimator = get_metric_depth_estimator(cfg)
        if cfg['mapping']["uncertainty_params"]['activate']:
            # If mapping needs dino features, we still need feature extractor
            self.feat_extractor = get_feature_extractor(cfg)
        
        # IMU integration settings
        self.imu_enabled = cfg.get('imu', {}).get('enabled', False)
        self.use_imu_orientation = cfg.get('imu', {}).get('use_orientation', False)
        self.warmup = cfg['tracking']['warmup']  # Number of frames before Frontend initialization
        
        # V2: Check if dataset is S3E (special rotation handling)
        self.dataset_is_s3e = is_s3e_dataset(cfg) if self.imu_enabled else False
        
        # V2.2: Probabilistic drift mitigation using IMU noise parameters
        self.sigma_acc_walk = cfg.get('imu', {}).get('sigma_acc_walk', 2.18e-04)  # m/s²/√s
        self.sigma_acc = cfg.get('imu', {}).get('sigma_acc', 1.09e-02)  # m/s²
        
        if self.imu_enabled:
            self.imu_gravity = torch.tensor(
                cfg.get('imu', {}).get('gravity', [0.0, 0.0, -9.81]),
                device=self.device, dtype=torch.float32
            )
            print(f"[MotionFilter] ✓ IMU propagation ENABLED (V2.2: Probabilistic Drift Mitigation)")
            print(f"[MotionFilter]   - Starts after {self.warmup} warmup frames")
            print(f"[MotionFilter]   - Gravity: {self.imu_gravity.cpu().numpy()}")
            print(f"[MotionFilter]   - Use IMU orientation: {self.use_imu_orientation}")
            print(f"[MotionFilter]   - Dataset type: {'S3E (rotation frozen)' if self.dataset_is_s3e else 'Normal (gyro integrated)'}")
            print(f"[MotionFilter]   - Accelerometer noise: σ_acc = {self.sigma_acc:.2e} m/s²")
            print(f"[MotionFilter]   - Bias random walk: σ_acc_walk = {self.sigma_acc_walk:.2e} m/s²/√s")
            print(f"[MotionFilter]   - Damping based on uncertainty growth (no hard limits)")
        else:
            self.imu_gravity = None
            print(f"[MotionFilter] ✗ IMU propagation DISABLED")
        
        # IMU statistics tracking
        self.imu_propagation_count = 0
        self.imu_propagation_failures = 0
        self.last_imu_propagated_frame = -1  # Track last propagated frame to prevent duplicates
        
        # Stage 5: MambaVision descriptor extractor for loop closure
        self.use_mamba_descriptors = cfg.get('loop_closure', {}).get('use_mamba_descriptors', False)
        self.enable_local_reranking = cfg.get('loop_closure', {}).get('enable_reranking', False)
        if self.use_mamba_descriptors:
            print(f"[MotionFilter] MambaVision descriptor extraction ENABLED")
            model_name = cfg.get('loop_closure', {}).get('mamba_model', 'nvidia/MambaVision-T-1K')
            entropy_threshold_t1 = cfg.get('loop_closure', {}).get('entropy_threshold_t1', 0.3)
            print(f"[MotionFilter]   - Model: {model_name}")
            print(f"[MotionFilter]   - Local reranking: {'ENABLED' if self.enable_local_reranking else 'DISABLED'}")
            self.descriptor_extractor = MambaDescriptorExtractor(
                model_name=model_name,
                device=device,
                input_size=(224, 224),
                enable_local_features=self.enable_local_reranking,
                entropy_threshold_t1=entropy_threshold_t1
            )
            self.descriptor_extraction_count = 0
            self.total_descriptor_time = 0.0
        else:
            self.descriptor_extractor = None
            print(f"[MotionFilter] ✗ MambaVision descriptor extraction DISABLED")

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None, imu_data=None):
        """ main update operation - run on every frame in video 
        
        Args:
            tstamp: Frame timestamp/index
            image: RGB image tensor [3, H, W]
            intrinsics: Camera intrinsics matrix
            imu_data: Optional dict with 'timestamps', 'angular_velocity', 'linear_acceleration'
                      for current frame. If None and IMU enabled, will be fetched from dataset.
        """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        force_to_add_keyframe = False
        
        # Prepare IMU chunk to pass to video.append()
        imu_chunk_for_storage = None

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
            if self.uncertainty_aware:
                dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    # If mapping needs dino features, we predict here and store the value in local disk
                    _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
            
            # Store IMU data for first frame if available
            if self.imu_enabled and imu_data is not None:
                imu_chunk_for_storage = imu_data
            
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, 
                            intrinsics / float(self.video.down_scale), gmap, 
                            net[0,0], inp[0,0], dino_features, imu_chunk_for_storage)
        ### only add new frame if there is enough motion ###
        else:
            # IMU PROPAGATION: Compute prior pose before feature-based motion check
            # CRITICAL: Skip IMU during warmup phase to avoid garbage velocity estimates
            current_frame_idx = self.video.counter.value
            
            # DEBUG: Log track() entry
            if self.imu_enabled and imu_data is not None:
                n_imu_samples = len(imu_data.get('timestamps', [])) if imu_data else 0
                print(f"[MotionFilter.track] Frame {current_frame_idx}: Entry with {n_imu_samples} IMU samples")
            
            if self.imu_enabled and imu_data is not None and current_frame_idx >= self.warmup:
                # CRITICAL: Prevent duplicate IMU propagation for the same frame
                if current_frame_idx == self.last_imu_propagated_frame:
                    print(f"[MotionFilter] ⚠ SKIPPING duplicate IMU propagation for frame {current_frame_idx}")
                    imu_chunk_for_storage = None
                else:
                    print(f"[MotionFilter] ✓ Performing IMU propagation for frame {current_frame_idx}")
                    self.last_imu_propagated_frame = current_frame_idx
                    try:
                        # Get previous two poses from DepthVideo
                        prev_idx = self.video.counter.value - 1
                        prev2_idx = max(0, prev_idx - 1)
                        
                        # Use BA-optimized poses directly (velocity clamping handles BA artifacts)
                        pose_t1 = self.video.poses[prev_idx].clone().to(dtype=torch.float32)  # t-1
                        pose_t2 = self.video.poses[prev2_idx].clone().to(dtype=torch.float32)  # t-2
                        print(f"               [DEBUG] Using BA-optimized poses (velocity clamping enabled)")
                        
                        # Get camera-to-IMU transform - ensure float32
                        c2i_transform = self.video.c2i_transform.clone().to(dtype=torch.float32)
                        
                        # Ensure IMU data is also float32
                        imu_data_float32 = {
                            'timestamps': imu_data['timestamps'].to(dtype=torch.float32),
                            'angular_velocity': imu_data['angular_velocity'].to(dtype=torch.float32),
                            'linear_acceleration': imu_data['linear_acceleration'].to(dtype=torch.float32),
                            'c2i_transform': imu_data['c2i_transform'].to(dtype=torch.float32)
                        }
                        if 'orientation' in imu_data:
                            imu_data_float32['orientation'] = imu_data['orientation'].to(dtype=torch.float32)
                        
                        # Compute time delta between camera frames from stored timestamps
                        if prev_idx > 0 and self.video.timestamp[prev_idx] > 0:
                            dt_cam = float(self.video.timestamp[prev_idx] - self.video.timestamp[prev2_idx])
                            # Sanity check: typical camera rates 5-30Hz (0.03-0.2s)
                            if dt_cam <= 0 or dt_cam > 1.0:
                                dt_cam = 0.1  # Fallback to 10Hz default
                        else:
                            dt_cam = 0.1  # Default 10Hz for first frame or invalid timestamps
                        
                        # Log IMU propagation attempt
                        n_imu_samples = len(imu_data_float32['timestamps'])
                        print(f"[MotionFilter] Frame {self.video.counter.value}: IMU propagation starting (V2)")
                        print(f"               - {n_imu_samples} IMU samples, dt_cam={dt_cam:.4f}s")
                        print(f"               - Prev pose index: t-1={prev_idx}")
                        
                        # V2: Get previous velocity from video state
                        prev_velocity = self.video.get_velocity(prev_idx) if prev_idx >= 0 else None
                        if prev_velocity is not None:
                            vel_mag = torch.norm(prev_velocity).item()
                            print(f"               - Input velocity: [{prev_velocity[0]:.3f}, {prev_velocity[1]:.3f}, {prev_velocity[2]:.3f}] m/s (mag={vel_mag:.3f})")
                        else:
                            print(f"               - Input velocity: None (will use zero initialization)")
                        
                        # Determine if dataset is S3E (skip gyro integration)
                        dataset_is_s3e = is_s3e_dataset(self.cfg)
                        
                        # Gravity vector in world frame
                        gravity = torch.tensor([0.0, 0.0, -9.81], device=pose_t1.device, dtype=torch.float32)
                        
                        # V2.2: Call new propagation function with probabilistic drift mitigation
                        propagated_pose, updated_velocity, covariance = propagate_imu_continuous(
                            pose_t1=pose_t1,
                            prev_velocity=prev_velocity,
                            imu_chunk=imu_data_float32,
                            c2i_transform=c2i_transform,
                            dt_cam=dt_cam,
                            gravity=gravity,
                            device=pose_t1.device,
                            dataset_is_s3e=dataset_is_s3e,
                            sigma_acc_walk=self.sigma_acc_walk,
                            sigma_acc=self.sigma_acc,
                            frames_since_init=self.video.counter.value,  # Use frame count for uncertainty model
                            power_factor=self.cfg.get('imu', {}).get('uncertainty_power_factor', 0.5)
                        )
                        
                        # V2: Store updated velocity for next frame
                        current_timestamp = float(tstamp)  # Use frame timestamp
                        next_idx = self.video.counter.value  # Will be incremented by append
                        self.video.set_velocity(
                            frame_idx=next_idx,
                            velocity=updated_velocity,
                            timestamp=current_timestamp
                        )
                        print(f"               ✓ Velocity stored at index {next_idx} for next frame")
                        
                        # Store propagated prior (will be used by frontend for initialization)
                        # Note: We store at prev_idx+1 which will become the current frame index
                        # after append is called
                        # Store prior at next index (current index + 1) so that
                        # the frontend which initializes the upcoming frame can use it.
                        # video.counter.value points to the index that will be used for the
                        # next append; store at +1 to be robust against different call orders.
                        current_idx = self.video.counter.value + 1
                        print(f"               [DEBUG] Storing IMU prior at index {current_idx} (reserved for upcoming frame)")
                        print(f"               [DEBUG] video.counter.value = {self.video.counter.value}")
                        try:
                            self.video.set_imu_prior(current_idx, propagated_pose, covariance)
                        except Exception:
                            # Fallback: if setting at +1 fails (out of bounds), set at current index
                            print(f"               [DEBUG] set_imu_prior at {current_idx} failed; falling back to {self.video.counter.value}")
                            self.video.set_imu_prior(self.video.counter.value, propagated_pose, covariance)
                        
                        # Compute delta from previous pose for logging
                        translation_delta = torch.norm(propagated_pose[:3] - pose_t1[:3]).item()
                        
                        # Log success
                        self.imu_propagation_count += 1
                        print(f"               ✓ IMU propagation SUCCESS")
                        print(f"               - Translation delta: {translation_delta:.4f}m")
                        print(f"               - Total propagations: {self.imu_propagation_count}")
                        
                        # Prepare IMU data for storage
                        imu_chunk_for_storage = imu_data
                        
                    except Exception as e:
                        self.imu_propagation_failures += 1
                        print(f"[MotionFilter] ✗ IMU propagation FAILED (frame {self.video.counter.value})")
                        print(f"               Error: {e}")
                        print(f"               Total failures: {self.imu_propagation_failures}/{self.imu_propagation_count + self.imu_propagation_failures}")
                        imu_chunk_for_storage = None
            elif self.imu_enabled and imu_data is not None and current_frame_idx < self.warmup:
                # During warmup, skip IMU propagation (visual_poses not yet reliable)
                print(f"[MotionFilter] Frame {current_frame_idx}: Skipping IMU during warmup ({current_frame_idx}/{self.warmup})")
                imu_chunk_for_storage = None
            elif self.imu_enabled and imu_data is None:
                print(f"[MotionFilter] ⚠ IMU enabled but no data received (frame {self.video.counter.value})")
                imu_chunk_for_storage = None
            else:
                imu_chunk_for_storage = None
                            
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']


            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
                if self.uncertainty_aware:
                    dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        # if mapping needs dino features, we predict here and store the value in local disk
                        _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device)
                
                self.video.append(tstamp, image[0], None, None, mono_depth, 
                                intrinsics / float(self.video.down_scale), gmap, 
                                net[0], inp[0], dino_features, imu_chunk_for_storage)
                
                # Stage 5: Extract MambaVision descriptor for this keyframe
                if self.use_mamba_descriptors:
                    import time
                    desc_start = time.time()

                    try:
                        # Extract descriptor (and optionally local features)
                        frame_idx = self.video.counter.value - 1

                        if self.enable_local_reranking:
                            # Extract both global descriptor and local features
                            descriptor, local_feats = self.descriptor_extractor.extract_descriptor(
                                image[0], return_local=True
                            )
                            # Store local features
                            self.video.set_local_features(frame_idx, local_feats)
                            num_kp = local_feats.shape[0]
                        else:
                            # Extract global descriptor only
                            descriptor = self.descriptor_extractor.extract_descriptor(image[0])
                            num_kp = 0

                        # Store global descriptor in DepthVideo
                        self.video.set_descriptor(frame_idx, descriptor)

                        # Add to FAISS index
                        success = self.video.add_descriptor_to_index(frame_idx)

                        desc_time = (time.time() - desc_start) * 1000  # ms
                        self.descriptor_extraction_count += 1
                        self.total_descriptor_time += desc_time

                        if success:
                            kp_info = f", kp={num_kp}" if self.enable_local_reranking else ""
                            print(f"[MotionFilter] Descriptor extracted for frame {frame_idx} "
                                  f"(time: {desc_time:.1f}ms{kp_info}, total: {self.descriptor_extraction_count})")

                    except Exception as e:
                        print(f"[MotionFilter] ERROR: Failed to extract descriptor for frame {tstamp}: {e}")

            else:
                self.count += 1

        return force_to_add_keyframe

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,suffix=suffix)
        return dino_features
