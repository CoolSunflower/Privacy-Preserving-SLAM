import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify
from time import gmtime, strftime

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            self.uncer_network = generate_uncertainty_mlp(n_features)
            self.uncer_network.share_memory()
        else:
            self.uncer_network = None
            if self.cfg["tracking"]["uncertainty_params"]["activate"]:
                raise ValueError(
                    "uncertainty estimation cannot be activated on tracking while not on mapping"
                )

        self.video = DepthVideo(cfg, self.printer, uncer_network=self.uncer_network)

        # Propagate camera-to-IMU transform from dataset to DepthVideo
        if self.video.imu_enabled and hasattr(stream, 'c2i_transform'):
            self.video.set_c2i_transform(stream.c2i_transform)

        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Tracking Starts!", FontColor.TRACKER)
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print("Tracking Done!", FontColor.TRACKER)

    def mapping(self, pipe, q_main2vis, q_vis2main):
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        else:
            self.mapper = Mapper(self, pipe, None, q_main2vis, q_vis2main)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Mapping Starts!", FontColor.MAPPER)
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        # print current time
        end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print("\nMapping Completed!\n" + f"{end_time}\n")

        self.terminate()

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""
        
        self.printer.print("="*60, FontColor.EVAL)
        self.printer.print("TERMINATE FUNCTION CALLED", FontColor.EVAL)
        self.printer.print("="*60, FontColor.EVAL)

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)

            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="before_refine",
            )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(f"Keyframe trajectory evaluation failed: {str(e)}", FontColor.ERROR)
                import traceback
                traceback.print_exc()
        
        if self.cfg.get("eval_manual", False):
            self.printer.print("Manual trajectory evaluation triggered due to eval_manual=True.", FontColor.EVAL)
            try:
                from scripts.eval_kf_traj import eval_traj
                eval_traj(self.save_dir, self.cfg['data']['input_folder'])
                self.printer.print(">>> Manual eval completed", FontColor.EVAL)
            except Exception as e:
                self.printer.print(f"Manual eval failed (non-critical): {e}", FontColor.INFO)
                import traceback
                traceback.print_exc()

        self.printer.print(">>> Checking final_ba config...", FontColor.EVAL)
        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.printer.print(">>> Running final_refine...", FontColor.EVAL)
            self.mapper.final_refine(
                iters=self.cfg["mapping"]["final_refine_iters"]
            )  # this performs a set of optimizations with RGBD loss to correct
            self.printer.print(">>> final_refine completed", FontColor.EVAL)

        # Evaluate the metrics
        self.printer.print(">>> Saving keyframe figures...", FontColor.EVAL)
        self.mapper.save_all_kf_figs(
            self.save_dir,
            iteration="after_refine",
        )
        self.printer.print(">>> Keyframe figures saved", FontColor.EVAL)

        ## Not used, see head comments of the function
        # self._eval_depth_all(ate_statistics, global_scale, r_a, t_a)

        # Regenerate feature extractor for non-keyframes
        self.printer.print(">>> Setting up feature extractor...", FontColor.EVAL)
        self.traj_filler.setup_feature_extractor()
        self.printer.print(">>> Running full_traj_eval...", FontColor.EVAL)
        full_traj_eval(
            self.traj_filler,
            self.mapper,
            f"{self.save_dir}/traj",
            "full_traj",
            self.stream,
            self.logger,
            self.printer,
            self.cfg['fast_mode'],
        )
        self.printer.print(">>> full_traj_eval completed", FontColor.EVAL)

        # ====== Privacy-Preserving SLAM: Post-processing excision ======
        # Run post-processing to excise private Gaussians before final export
        self.printer.print("Checking privacy configuration...", FontColor.EVAL)
        self.printer.print(f"Privacy enabled: {self.cfg.get('privacy', {}).get('enable', False)}", FontColor.EVAL)
        
        privacy_masks_collected = {}
        if self.cfg.get("privacy", {}).get("enable", False):
            try:
                from src.privacy import PrivacyManager
                privacy_cfg = self.cfg.get("privacy", {})
                privacy_mgr = PrivacyManager(privacy_cfg, self.device)

                # Get cameras dict from mapper
                cameras = self.mapper.get_cameras_dict() if hasattr(self.mapper, 'get_cameras_dict') else {}

                # Run post-processing excision and filling
                try:
                    num_excised, num_filled = privacy_mgr.postprocess_map(
                        self.mapper.gaussians,
                        cameras,
                        self.stream
                    )
                    self.printer.print(
                        f"Privacy: excised {num_excised} Gaussians, filled with {num_filled}",
                        FontColor.INFO
                    )
                except FileNotFoundError as e:
                    self.printer.print(
                        f"Privacy postprocessing skipped: {e}",
                        FontColor.INFO
                    )
                    self.printer.print(
                        f"Please run: bash scripts/download_privacy_weights.sh",
                        FontColor.INFO
                    )
                    num_excised = 0
                    num_filled = 0

                # Final sanitization before export
                try:
                    num_sanitized = privacy_mgr.sanitize_for_export(self.mapper.gaussians)
                    if num_sanitized > 0:
                        self.printer.print(
                            f"Privacy: final sanitization removed {num_sanitized} Gaussians",
                            FontColor.INFO
                        )
                except Exception as e:
                    self.printer.print(f"Privacy sanitization skipped: {e}", FontColor.INFO)
                    num_sanitized = 0

                # Collect privacy masks for metrics computation (even if postprocessing failed)
                privacy_masks_collected = privacy_mgr.frame_masks if hasattr(privacy_mgr, 'frame_masks') else {}
                
                # Debug: log what we collected
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                print(f"\n[METRICS DEBUG] Privacy masks collected: {len(privacy_masks_collected)} frames", flush=True)
                if privacy_masks_collected:
                    print(f"[METRICS DEBUG] Frame indices with masks: {list(privacy_masks_collected.keys())[:5]}...", flush=True)
                else:
                    print(f"[METRICS DEBUG] WARNING: No privacy masks collected!", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()

                # Save privacy metrics
                import json
                privacy_stats = privacy_mgr.get_statistics()
                privacy_metrics_path = f"{self.save_dir}/privacy_metrics.json"
                with open(privacy_metrics_path, 'w') as f:
                    json.dump({
                        "mode": privacy_stats.mode,
                        "num_frames_processed": privacy_stats.num_frames_processed,
                        "num_keyframes_with_privacy": privacy_stats.num_keyframes_with_privacy,
                        "total_excised": privacy_stats.total_excised,
                        "total_filled": privacy_stats.total_filled,
                    }, f, indent=2)
                self.printer.print(f"Privacy metrics saved to {privacy_metrics_path}", FontColor.INFO)

            except Exception as e:
                self.printer.print(f"Privacy post-processing failed: {str(e)}", FontColor.ERROR)
                import traceback
                traceback.print_exc()

        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        # ====== Compute PSNR/SSIM Metrics ======
        self.printer.print("About to compute render metrics...", FontColor.EVAL)
        try:
            self._compute_and_save_render_metrics(privacy_masks_collected)
        except Exception as e:
            self.printer.print(f"Render metrics computation failed: {str(e)}", FontColor.ERROR)
            import traceback
            traceback.print_exc()

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            torch.save(
                self.mapper.uncer_network.state_dict(),
                self.save_dir + "/uncertainty_mlp_weight.pth",
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in WildGS-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def _compute_and_save_render_metrics(self, privacy_masks: dict = None):
        """
        Compute comprehensive rendering and privacy metrics.
        
        Computes on ALL keyframes:
        - PSNR (Peak Signal-to-Noise Ratio)
        - SSIM (Structural Similarity Index) using pytorch_msssim
        - Depth L1 error
        - SSIM-Sensitive (privacy metric for private regions)
        - Re-ID Score (face detection on rendered images)
        
        Args:
            privacy_masks: Dict mapping frame_idx -> privacy mask tensor
        """
        import json
        import torch
        import time
        from thirdparty.gaussian_splatting.gaussian_renderer import render
        
        self.printer.print("="*80, FontColor.EVAL)
        self.printer.print("Computing comprehensive render metrics on ALL keyframes...", FontColor.EVAL)
        self.printer.print("="*80, FontColor.EVAL)
        
        start_time = time.time()
        
        # Get cameras dict from mapper
        cameras = self.mapper.get_cameras_dict() if hasattr(self.mapper, 'get_cameras_dict') else {}
        
        if not cameras:
            self.printer.print("No cameras available for metrics computation", FontColor.INFO)
            return
        
        num_cameras = len(cameras)
        self.printer.print(f"Total keyframes to evaluate: {num_cameras}", FontColor.EVAL)
        
        # Initialize metric accumulators
        psnr_values = []
        ssim_values = []
        depth_l1_values = []
        ssim_sensitive_values = []
        reid_detections = {"rendered": 0, "ground_truth": 0}
        
        # Try to import pytorch_msssim for proper SSIM computation
        try:
            from pytorch_msssim import ssim as compute_ssim_torch
            use_proper_ssim = True
            self.printer.print("Using pytorch_msssim for exact SSIM computation", FontColor.EVAL)
        except ImportError:
            use_proper_ssim = False
            self.printer.print("Warning: pytorch_msssim not available, using approximation", FontColor.INFO)
        
        # Initialize Re-ID detector (use SAME detector as privacy masks for consistency)
        # Privacy masks are created by Grounding DINO + SAM, so Re-ID should also use it
        reid_detector = None
        
        # Debug logging
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"\n[RE-ID DEBUG] privacy_masks parameter: {type(privacy_masks)}, len={len(privacy_masks) if privacy_masks else 0}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        if privacy_masks:
            self.printer.print(f"Privacy masks available for {len(privacy_masks)} frames", FontColor.EVAL)
            try:
                # Use Grounding DINO + SAM for Re-ID (same as postprocess detector)
                # This ensures we detect the SAME objects that created the privacy masks
                from src.privacy.detectors.grounding_sam_detector import GroundingSAMDetector
                privacy_cfg = self.cfg.get("privacy", {})
                gsam_cfg = privacy_cfg.get("postprocess_config", {})
                if not gsam_cfg:
                    # Default Grounding DINO config matching what postprocess uses
                    gsam_cfg = {
                        "text_prompts": ["person", "human face", "screen"],
                        "box_threshold": 0.35,
                        "text_threshold": 0.25,
                    }
                reid_detector = GroundingSAMDetector(gsam_cfg, self.device)
                reid_detector.load_model()
                self.printer.print("Initialized Grounding DINO + SAM for Re-ID score computation", FontColor.EVAL)
                print(f"[RE-ID DEBUG] Grounding DINO detector initialized successfully", flush=True)
            except Exception as e:
                self.printer.print(f"Could not initialize Re-ID detector: {e}", FontColor.INFO)
                import traceback
                traceback.print_exc()
                # Fallback to YOLO if Grounding DINO fails
                try:
                    from src.privacy.detectors.yolo_detector import YOLOPrivacyDetector
                    detector_cfg = privacy_cfg.get("yolo_config", {})
                    reid_detector = YOLOPrivacyDetector(detector_cfg, self.device)
                    reid_detector.load_model()
                    self.printer.print("Fallback: Using YOLO detector for Re-ID", FontColor.INFO)
                except Exception as e2:
                    self.printer.print(f"Could not initialize fallback YOLO detector: {e2}", FontColor.INFO)
                print(f"[RE-ID DEBUG] YOLO initialization failed: {e}", flush=True)
        else:
            self.printer.print("No privacy masks available - skipping Re-ID computation", FontColor.INFO)
            print(f"[RE-ID DEBUG] No privacy masks - Re-ID skipped", flush=True)
        
        # Evaluate all keyframes
        eval_indices = sorted(cameras.keys())
        
        for idx, video_idx in enumerate(eval_indices):
            if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == num_cameras:
                self.printer.print(
                    f"Progress: {idx + 1}/{num_cameras} keyframes processed "
                    f"({100.0 * (idx + 1) / num_cameras:.1f}%)",
                    FontColor.EVAL
                )
            
            try:
                camera = cameras[video_idx]
                frame_idx = camera.uid if hasattr(camera, 'uid') else video_idx
                
                # Render from Gaussians
                with torch.no_grad():
                    render_pkg = render(
                        camera, 
                        self.mapper.gaussians, 
                        self.mapper.pipeline_params, 
                        self.mapper.background
                    )
                
                rendered = render_pkg["render"]  # (3, H, W) in [0, 1]
                gt_image = camera.original_image  # (3, H, W) in [0, 1]
                
                # ====== PSNR Computation ======
                mse = torch.mean((rendered - gt_image) ** 2).item()
                if mse > 1e-10:
                    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
                else:
                    psnr = 100.0
                psnr_values.append(psnr)
                
                # ====== SSIM Computation ======
                if use_proper_ssim:
                    # Use pytorch_msssim for exact SSIM
                    # Need (B, C, H, W) format
                    rendered_batch = rendered.unsqueeze(0)  # (1, 3, H, W)
                    gt_batch = gt_image.unsqueeze(0)  # (1, 3, H, W)
                    
                    ssim_val = compute_ssim_torch(
                        rendered_batch, 
                        gt_batch, 
                        data_range=1.0,
                        size_average=True
                    ).item()
                    ssim_values.append(ssim_val)
                else:
                    # Fallback: correlation-based approximation
                    rendered_flat = rendered.flatten()
                    gt_flat = gt_image.flatten()
                    
                    rendered_mean = rendered_flat.mean()
                    gt_mean = gt_flat.mean()
                    rendered_std = rendered_flat.std()
                    gt_std = gt_flat.std()
                    
                    if rendered_std > 1e-6 and gt_std > 1e-6:
                        covariance = ((rendered_flat - rendered_mean) * (gt_flat - gt_mean)).mean()
                        ssim_approx = (2 * rendered_mean * gt_mean) * (2 * covariance) / (
                            (rendered_mean**2 + gt_mean**2) * (rendered_std**2 + gt_std**2) + 1e-8
                        )
                        ssim_values.append(max(0.0, min(1.0, ssim_approx.item())))
                
                # ====== Depth L1 Error ======
                if "depth" in render_pkg and hasattr(camera, 'depth') and camera.depth is not None:
                    rendered_depth = render_pkg["depth"][0]  # (H, W)
                    gt_depth = torch.from_numpy(camera.depth).to(rendered_depth.device)
                    
                    # Compute L1 error where gt depth is valid (> 0)
                    valid_mask = gt_depth > 0.01
                    if valid_mask.sum() > 0:
                        l1_error = torch.abs(rendered_depth[valid_mask] - gt_depth[valid_mask]).mean().item()
                        depth_l1_values.append(l1_error)
                
                # ====== SSIM-Sensitive (Privacy Metric) ======
                if privacy_masks and frame_idx in privacy_masks:
                    mask = privacy_masks[frame_idx]
                    if mask.sum() > 100:  # Need enough pixels
                        # Proper SSIM-Sensitive: SSIM between rendered and black in private regions
                        mask_resized = mask
                        if mask.shape != rendered.shape[1:]:
                            import torch.nn.functional as F
                            mask_resized = F.interpolate(
                                mask.unsqueeze(0).unsqueeze(0).float(),
                                size=rendered.shape[1:],
                                mode='nearest'
                            )[0, 0]
                        
                        mask_3d = mask_resized.unsqueeze(0) > 0.5  # (1, H, W)
                        
                        # Target is black (0.0) for excised regions
                        target_black = torch.zeros_like(rendered)
                        
                        # Compute SSIM only in masked regions
                        # Extract patches and compute mean
                        rendered_private = rendered * mask_3d
                        target_private = target_black * mask_3d
                        
                        # Simpler metric: mean absolute difference to black
                        # Higher score = more similar to black = better privacy
                        private_pixel_mean = rendered_private[mask_3d.expand_as(rendered)].mean().item()
                        ssim_sensitive = 1.0 - private_pixel_mean  # Inverted: closer to black = higher score
                        ssim_sensitive_values.append(max(0.0, min(1.0, ssim_sensitive)))
                
                # ====== Re-ID Score (Privacy Object Detection) ======
                # Uses Grounding DINO (same as privacy mask creation) to ensure consistency
                if reid_detector is not None and privacy_masks and frame_idx in privacy_masks:
                    mask = privacy_masks[frame_idx]
                    if mask.sum() > 100:
                        # Clamp images to [0,1] - rendered output from Gaussian Splatting may have
                        # values outside this range, which breaks detection
                        rendered_clamped = rendered.clamp(0, 1)
                        gt_clamped = gt_image.clamp(0, 1)
                        
                        # Detect privacy objects in rendered vs ground truth image
                        rendered_result = reid_detector.detect(rendered_clamped)
                        gt_result = reid_detector.detect(gt_clamped)
                        
                        rendered_det_count = len(rendered_result.detections) if rendered_result.has_detections else 0
                        gt_det_count = len(gt_result.detections) if gt_result.has_detections else 0
                        
                        # Track total detections for debugging (not just those in private regions)
                        total_rendered_dets = rendered_det_count
                        total_gt_dets = gt_det_count
                        
                        # Resize mask to match rendered image dimensions if needed
                        # (mask is created during postprocess at camera.original_image resolution)
                        mask_for_reid = mask
                        rendered_h, rendered_w = rendered.shape[1], rendered.shape[2]
                        if mask.shape[0] != rendered_h or mask.shape[1] != rendered_w:
                            import torch.nn.functional as F
                            mask_for_reid = F.interpolate(
                                mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(rendered_h, rendered_w),
                                mode='nearest'
                            )[0, 0]
                        
                        # Debug logging for first few frames
                        if idx < 5:
                            print(f"[Re-ID] Frame {frame_idx}: rendered_dets={rendered_det_count}, gt_dets={gt_det_count}, "
                                  f"mask_pixels={int(mask_for_reid.sum().item())}, mask_shape={tuple(mask_for_reid.shape)}", flush=True)
                        
                        # Count detections that overlap with privacy mask
                        if rendered_result.has_detections:
                            for det in rendered_result.detections:
                                x1, y1, x2, y2 = det.bbox
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(mask_for_reid.shape[1], int(x2)), min(mask_for_reid.shape[0], int(y2))
                                if x2 > x1 and y2 > y1:
                                    roi_mask = mask_for_reid[y1:y2, x1:x2]
                                    overlap = roi_mask.sum() / roi_mask.numel() if roi_mask.numel() > 0 else 0
                                    if idx < 5:
                                        print(f"  Rendered det bbox=({x1},{y1},{x2},{y2}), overlap={overlap:.2f}", flush=True)
                                    if overlap > 0.3:  # 30% overlap
                                        reid_detections["rendered"] += 1
                        
                        if gt_result.has_detections:
                            for det in gt_result.detections:
                                x1, y1, x2, y2 = det.bbox
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(mask_for_reid.shape[1], int(x2)), min(mask_for_reid.shape[0], int(y2))
                                if x2 > x1 and y2 > y1:
                                    roi_mask = mask_for_reid[y1:y2, x1:x2]
                                    overlap = roi_mask.sum() / roi_mask.numel() if roi_mask.numel() > 0 else 0
                                    if idx < 5:
                                        print(f"  GT det bbox=({x1},{y1},{x2},{y2}), overlap={overlap:.2f}", flush=True)
                                    if overlap > 0.3:
                                        reid_detections["ground_truth"] += 1
                        
            except Exception as e:
                self.printer.print(f"Error computing metrics for frame {video_idx}: {e}", FontColor.INFO)
                import traceback
                traceback.print_exc()
                continue
        
        elapsed_time = time.time() - start_time
        self.printer.print(f"Metrics computation completed in {elapsed_time:.1f} seconds", FontColor.EVAL)
        
        # ====== Debug: Report Re-ID statistics ======
        if privacy_masks:
            self.printer.print(f"Re-ID: Evaluated {len(privacy_masks)} frames with privacy masks", FontColor.EVAL)
            self.printer.print(
                f"Re-ID: Detections - Rendered: {reid_detections['rendered']}, GT: {reid_detections['ground_truth']}",
                FontColor.EVAL
            )
            print(f"[RE-ID SUMMARY] Total: rendered={reid_detections['rendered']}, gt={reid_detections['ground_truth']}", flush=True)
        
        # ====== Aggregate Metrics ======
        avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
        avg_depth_l1 = sum(depth_l1_values) / len(depth_l1_values) if depth_l1_values else float('nan')
        avg_ssim_sensitive = sum(ssim_sensitive_values) / len(ssim_sensitive_values) if ssim_sensitive_values else float('nan')
        
        # Compute Re-ID score: lower detection rate = better privacy
        # Score = 1 - (rendered_detections / gt_detections)
        if reid_detections["ground_truth"] > 0:
            reid_score = 1.0 - min(1.0, reid_detections["rendered"] / reid_detections["ground_truth"])
        else:
            reid_score = float('nan')
        
        # ====== Print Summary ======
        self.printer.print("="*80, FontColor.EVAL)
        self.printer.print("METRICS SUMMARY", FontColor.EVAL)
        self.printer.print("="*80, FontColor.EVAL)
        self.printer.print(f"  PSNR:              {avg_psnr:.2f} dB  (n={len(psnr_values)})", FontColor.EVAL)
        self.printer.print(f"  SSIM:              {avg_ssim:.4f}     (n={len(ssim_values)})", FontColor.EVAL)
        if depth_l1_values:
            self.printer.print(f"  Depth L1:          {avg_depth_l1:.4f} m   (n={len(depth_l1_values)})", FontColor.EVAL)
        if ssim_sensitive_values:
            self.printer.print(f"  SSIM-Sensitive:    {avg_ssim_sensitive:.4f}     (n={len(ssim_sensitive_values)})", FontColor.EVAL)
        if not np.isnan(reid_score):
            self.printer.print(
                f"  Re-ID Score:       {reid_score:.4f}     "
                f"(rendered: {reid_detections['rendered']}, GT: {reid_detections['ground_truth']})",
                FontColor.EVAL
            )
        self.printer.print("="*80, FontColor.EVAL)
        
        # ====== Save Metrics to JSON ======
        metrics_path = f"{self.save_dir}/metrics.json"
        metrics_data = {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "depth_l1": avg_depth_l1,
            "num_frames_evaluated": len(psnr_values),
            "computation_time_seconds": elapsed_time,
            "ssim_method": "pytorch_msssim" if use_proper_ssim else "correlation_approximation"
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.printer.print(f"Saved metrics to: {metrics_path}", FontColor.EVAL)
        
        # ====== Update Privacy Metrics ======
        if privacy_masks:
            privacy_metrics_path = f"{self.save_dir}/privacy_metrics.json"
            if os.path.exists(privacy_metrics_path):
                with open(privacy_metrics_path, 'r') as f:
                    priv_data = json.load(f)
            else:
                priv_data = {}
            
            priv_data["ssim_sensitive"] = avg_ssim_sensitive
            priv_data["reid_score"] = reid_score
            priv_data["reid_detections_rendered"] = reid_detections["rendered"]
            priv_data["reid_detections_ground_truth"] = reid_detections["ground_truth"]
            priv_data["num_privacy_frames_evaluated"] = len(ssim_sensitive_values)
            
            with open(privacy_metrics_path, 'w') as f:
                json.dump(priv_data, f, indent=2)
            
            self.printer.print(f"Updated privacy metrics: {privacy_metrics_path}", FontColor.EVAL)
            
            # ====== Save Qualitative Visualizations ======
            self._save_qualitative_results(cameras, privacy_masks, num_frames=5)

    def _save_qualitative_results(self, cameras: dict, privacy_masks: dict = None, num_frames: int = 5):
        """
        Save qualitative comparison images for privacy evaluation.
        
        Generates:
        - Side-by-side comparisons: GT | Rendered | Privacy Mask | Masked Rendered
        - Top high-privacy frames ranked by mask coverage
        - Summary grid image
        """
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from thirdparty.gaussian_splatting.gaussian_renderer import render
        except ImportError as e:
            self.printer.print(f"Could not import required libraries for visualization: {e}", FontColor.ERROR)
            return
        
        self.printer.print("Generating qualitative visualizations...", FontColor.INFO)
        
        # Create visualizations directory
        viz_dir = f"{self.save_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        if not privacy_masks:
            self.printer.print("No privacy masks available - saving basic renders only", FontColor.INFO)
            privacy_masks = {}
        
        # Find frames with highest privacy pixel coverage
        frame_privacy_scores = []
        for frame_idx, mask in privacy_masks.items():
            coverage = mask.sum().item() / mask.numel()
            frame_privacy_scores.append((frame_idx, coverage, mask.sum().item()))
        
        # Sort by coverage (descending) to get frames with most privacy content
        frame_privacy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top frames for visualization
        top_frames = [f[0] for f in frame_privacy_scores[:num_frames]] if frame_privacy_scores else []
        
        # If no privacy frames, use evenly spaced keyframes
        if not top_frames:
            all_indices = sorted(cameras.keys())
            step = max(1, len(all_indices) // num_frames)
            top_frames = [all_indices[i] for i in range(0, len(all_indices), step)][:num_frames]
        
        saved_count = 0
        comparison_images = []
        
        for frame_idx in top_frames:
            # Find camera for this frame
            camera = None
            for vid_idx, cam in cameras.items():
                if (hasattr(cam, 'uid') and cam.uid == frame_idx) or vid_idx == frame_idx:
                    camera = cam
                    break
            
            if camera is None:
                continue
            
            try:
                # Render from Gaussians
                with torch.no_grad():
                    render_pkg = render(
                        camera,
                        self.mapper.gaussians,
                        self.mapper.pipeline_params,
                        self.mapper.background
                    )
                
                rendered = render_pkg["render"].clamp(0, 1)  # (3, H, W)
                gt_image = camera.original_image  # (3, H, W)
                
                # Convert to numpy HWC for visualization
                rendered_np = (rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Get privacy mask if available
                mask = privacy_masks.get(frame_idx, None)
                
                if mask is not None:
                    # Resize mask to match image dimensions
                    if mask.shape != rendered.shape[1:]:
                        import torch.nn.functional as F
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=rendered.shape[1:],
                            mode='nearest'
                        )[0, 0]
                    
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    
                    # Create mask overlay on GT (red tint for private regions)
                    mask_overlay = gt_np.copy()
                    mask_bool = mask_np > 128
                    mask_overlay[mask_bool, 0] = np.minimum(255, mask_overlay[mask_bool, 0] + 100)  # Red tint
                    
                    # Create masked rendered view (show only non-private regions)
                    masked_rendered = rendered_np.copy()
                    masked_rendered[mask_bool] = [50, 50, 50]  # Dark gray for private
                    
                    # Create 4-panel comparison
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    axes[0].imshow(gt_np)
                    axes[0].set_title(f'Ground Truth (Frame {frame_idx})')
                    axes[0].axis('off')
                    
                    axes[1].imshow(rendered_np)
                    axes[1].set_title('Rendered')
                    axes[1].axis('off')
                    
                    axes[2].imshow(mask_overlay)
                    axes[2].set_title(f'Privacy Mask ({int(mask.sum().item())} pixels)')
                    axes[2].axis('off')
                    
                    axes[3].imshow(masked_rendered)
                    axes[3].set_title('Rendered (Privacy Hidden)')
                    axes[3].axis('off')
                    
                    coverage = mask.sum().item() / mask.numel() * 100
                    fig.suptitle(f'Privacy Comparison - Frame {frame_idx} ({coverage:.1f}% private)', fontsize=14)
                    
                else:
                    # Create 2-panel comparison (no mask)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    axes[0].imshow(gt_np)
                    axes[0].set_title(f'Ground Truth (Frame {frame_idx})')
                    axes[0].axis('off')
                    
                    axes[1].imshow(rendered_np)
                    axes[1].set_title('Rendered')
                    axes[1].axis('off')
                    
                    fig.suptitle(f'Render Comparison - Frame {frame_idx}', fontsize=14)
                
                plt.tight_layout()
                save_path = f"{viz_dir}/comparison_frame_{frame_idx:04d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                comparison_images.append(save_path)
                saved_count += 1
                
            except Exception as e:
                self.printer.print(f"Could not visualize frame {frame_idx}: {e}", FontColor.INFO)
                continue
        
        # Create a summary grid if we have multiple images
        if len(comparison_images) >= 2:
            try:
                fig, axes = plt.subplots(len(comparison_images), 1, figsize=(15, 5 * len(comparison_images)))
                if len(comparison_images) == 1:
                    axes = [axes]
                
                for i, img_path in enumerate(comparison_images):
                    img = plt.imread(img_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/summary_grid.png", dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.printer.print(f"Could not create summary grid: {e}", FontColor.INFO)
        
        # Save privacy mask statistics
        if frame_privacy_scores:
            stats_path = f"{viz_dir}/privacy_coverage_stats.txt"
            with open(stats_path, 'w') as f:
                f.write("Privacy Coverage Statistics\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total frames with privacy masks: {len(frame_privacy_scores)}\n")
                f.write(f"Top {min(10, len(frame_privacy_scores))} frames by coverage:\n\n")
                f.write(f"{'Frame ID':<12} {'Coverage %':<12} {'Pixels':<15}\n")
                f.write("-" * 40 + "\n")
                for frame_idx, coverage, pixels in frame_privacy_scores[:10]:
                    f.write(f"{frame_idx:<12} {coverage*100:<12.2f} {int(pixels):<15}\n")
            
            self.printer.print(f"Saved privacy statistics to: {stats_path}", FontColor.INFO)
        
        self.printer.print(f"Saved {saved_count} comparison images to: {viz_dir}", FontColor.EVAL)

    def run(self):
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,q_main2vis,q_vis2main)),
        ]
        self.num_running_thread += len(processes)
        if self.cfg['gui']:
            self.num_running_thread += 1
        for p in processes:
            p.start()

        if self.cfg['gui']:
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.all_trigered += 1


        for p in processes:
            p.join()

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
