import torch
from src.factor_graph import FactorGraph
from src.backend import Backend as LoopClosing

class Frontend:
    # mainly inherited from GO-SLAM
    def __init__(self, net, video, cfg):
        self.cfg = cfg
        self.video = video
        self.update_op = net.update
        
        # local optimization window
        self.t1 = 0

        # frontent variables
        self.is_initialized = False

        self.max_age = cfg['tracking']['max_age']
        self.iters1 = 4*2
        self.iters2 = 2*2

        self.warmup = cfg['tracking']['warmup']
        self.beta = cfg['tracking']['beta']
        self.frontend_nms = cfg['tracking']['frontend']['nms']
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh']
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.frontend_thresh = cfg['tracking']['frontend']['thresh']
        self.frontend_radius = cfg['tracking']['frontend']['radius']
        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors']

        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, cfg)

        self.graph = FactorGraph(
            video, net.update,
            device=cfg['device'],
            corr_impl='volume',
            max_factors=self.frontend_max_factors
        )

        ## This is to avoid too many consecutive candidate keyframes which:
        #  1. capture large moving objects (high optical flow)
        #  2. don't have much camera motion (will be removed from the candidate later on)
        ## If there are too many of this kind of keyframes, we will have 0 edge in the graph.
        #  Because when a frame is determined as potential keyframe, other edges will be updated as well
        #  even if this frame is removed at the end due to less camera motion. And we will remove the edges
        #  that have been updated more than cfg['tracking']['max_age']
        self.max_consecutive_drop_of_keyframes = (cfg['tracking']['max_age']/self.iters1)//3
        self.num_keyframes_dropped = 0
        
        # IMU prior settings with feature flags
        self.imu_enabled = cfg.get('imu', {}).get('enabled', False)
        self.use_imu_in_frontend = cfg.get('imu', {}).get('use_in_frontend', False) and self.imu_enabled
        self.use_imu_in_ba = cfg.get('imu', {}).get('use_in_ba', False) and self.imu_enabled
        self.imu_warmup = cfg.get('imu', {}).get('warmup', 20)  # Separate warmup for IMU
        
        print(f"[Frontend] IMU Integration Settings:")
        print(f"  - IMU Enabled: {self.imu_enabled}")
        print(f"  - Use in Frontend (pose init): {self.use_imu_in_frontend}")
        print(f"  - Use in Backend (BA constraint): {self.use_imu_in_ba}")
        print(f"  - IMU Warmup: {self.imu_warmup} frames")
        
        # IMU prior usage tracking
        self.imu_priors_used = 0
        self.imu_priors_unavailable = 0

    def __update(self, force_to_add_keyframe):
        """ add edges, perform update """

        self.t1 += 1
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

            if not self.cfg['fast_mode']:
                if itr == 1 and self.video.metric_depth_reg and self.cfg['tracking']["uncertainty_params"]['activate']:
                    self.video.filter_high_err_mono_depth(self.t1-1,self.graph.ii,self.graph.jj)

        d = self.video.distance([self.t1-2], [self.t1-1], beta=self.beta, bidirectional=True)
        # Ssee self.max_consecutive_drop_of_keyframes in initi for explanation of the following process
        if (d.item() < self.keyframe_thresh) & (self.num_keyframes_dropped < self.max_consecutive_drop_of_keyframes) & (not force_to_add_keyframe):
            self.graph.rm_keyframe(self.t1 - 1)         
            self.num_keyframes_dropped += 1
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        else:
            cur_t = self.video.counter.value
            self.num_keyframes_dropped  = 0
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, 
                                                         motion_only=False, local_graph=self.graph,
                                                         enable_wq=True)
                if n_edge == 0:
                    for itr in range(self.iters2):
                        self.graph.update(t0=None, t1=None, use_inactive=True)
                self.last_loop_t = cur_t
            else:
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True)

        # set pose for next iteration
        # Initialize with constant velocity
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        
        # Try to use IMU prior (overwrites poses[t1] if available)
        # Only use IMU after warmup period (t1 >= imu_warmup)
        if self.use_imu_in_frontend and self.t1 >= self.imu_warmup:
            # MotionFilter stores the IMU prior for the upcoming frame index.
            # Fetch the prior for the current frame index to match that storage.
            prior_pose, _ = self.video.get_imu_prior(self.t1)
            if prior_pose is not None and torch.isfinite(prior_pose).all():
                print("[Frontend] Using IMU propogated pose for initialisation", self.t1)
                self.video.poses[self.t1] = prior_pose
                self.imu_priors_used += 1
            else:
                self.imu_priors_unavailable += 1
                if self.t1 < self.imu_warmup + 5:  # Log first few failures
                    print(f"[Frontend] Frame {self.t1}: IMU prior unavailable, using CV")
        elif self.use_imu_in_frontend and self.t1 < self.imu_warmup:
            # During IMU warmup: explicitly use constant velocity
            if self.t1 == 0 or (self.t1 < 5):
                print(f"[Frontend] Frame {self.t1}: IMU warmup, using CV initialization")
        
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.set_dirty(self.graph.ii.min(), self.t1)
        torch.cuda.empty_cache()

    def __initialize(self):
        """ initialize the SLAM system, i.e. bootstrapping """

        self.t1 = self.video.counter.value
        
        # Removed: visual_poses skip flag - not needed anymore with velocity clamping

        self.graph.add_neighborhood_factors(0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # Removed: visual_poses initialization - not needed anymore
        
        # self.video.normalize()
        
        # Removed: visual_poses saving after BA - using velocity clamping instead
        
        # Initialize next pose: constant velocity (before IMU warmup) or IMU prior (after warmup)
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        
        if self.use_imu_in_frontend and self.t1 >= self.imu_warmup:
            # Fetch prior for the current frame (MotionFilter writes priors into next index)
            prior_pose, _ = self.video.get_imu_prior(self.t1)
            if prior_pose is not None and torch.isfinite(prior_pose).all():
                self.video.poses[self.t1] = prior_pose
                self.imu_priors_used += 1
                print(f"[Frontend] __initialize: IMU prior applied to frame {self.t1}")
            else:
                self.imu_priors_unavailable += 1
                print(f"[Frontend] __initialize: IMU prior unavailable for frame {self.t1}, using CV")
        else:
            print(f"[Frontend] __initialize: Frame {self.t1} < IMU warmup ({self.imu_warmup}), using CV")
        
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.set_dirty(0, self.t1)

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def initialize_second_stage(self):
        """ 2nd stage of initializing the SLAM system after we have reliable uncertainty mask from mapping """
        self.t1 = self.video.counter.value

        # update mask
        if self.cfg['tracking']["uncertainty_params"]['activate']:
            self.video.update_all_uncertainty_mask()

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        
        # Removed: visual_poses saving - using velocity clamping in IMU propagation instead
        
        # Initialize next pose: constant velocity or IMU prior
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        
        if self.video.imu_enabled and hasattr(self.video, 'visual_poses'):
            self.video.visual_poses[self.t1] = self.video.poses[self.t1-1].clone()
        
        if self.use_imu_in_frontend and self.t1 >= self.imu_warmup:
            # Fetch prior for the current frame (MotionFilter writes priors into next index)
            prior_pose, _ = self.video.get_imu_prior(self.t1)
            if prior_pose is not None and torch.isfinite(prior_pose).all():
                self.video.poses[self.t1] = prior_pose
                self.imu_priors_used += 1
                print(f"[Frontend] initialize_second_stage: IMU prior applied to frame {self.t1}")
            else:
                self.imu_priors_unavailable += 1
                print(f"[Frontend] initialize_second_stage: IMU prior unavailable, using CV")
        else:
            print(f"[Frontend] initialize_second_stage: Frame {self.t1} < IMU warmup ({self.imu_warmup}), using CV")
        
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.set_dirty(0, self.t1)

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self, force_to_add_keyframe):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            self.video.update_valid_depth_mask()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            if self.cfg['tracking']["uncertainty_params"]['activate']:
                self.video.update_all_uncertainty_mask()
            self.__update(force_to_add_keyframe)
            self.video.update_valid_depth_mask()

