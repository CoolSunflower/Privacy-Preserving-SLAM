import glob
import os
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def load_metric_depth(idx,path):
    # omnidata depth
    mono_depth_path = f"{path}/mono_priors/depths/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    
    return mono_depth_tensor  

def load_img_feature(idx,path,suffix=''):
    # image features
    feat_path = f"{path}/mono_priors/features/{idx:05d}{suffix}.npy"
    feat = np.load(feat_path)
    feat_tensor = torch.from_numpy(feat)
    
    return feat_tensor  


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig = self.fx, self.fy, self.cx, self.cy
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        self.H_out_with_edge, self.W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge
        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()

        self.fovx = focal2fov(self.fx, self.W_out)
        self.fovy = focal2fov(self.fy, self.H_out)

        self.W_edge_full = int(math.ceil(self.W_edge*self.W/self.W_out_with_edge))
        self.H_edge_full =  int(math.ceil(self.H_edge*self.H/self.H_out_with_edge))
        self.H_out_full, self.W_out_full = self.H - self.H_edge_full * 2, self.W - self.W_edge_full * 2

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        self.input_folder = cfg['data']['input_folder']
        if "ROOT_FOLDER_PLACEHOLDER" in self.input_folder:
            self.input_folder = self.input_folder.replace("ROOT_FOLDER_PLACEHOLDER", cfg['data']['root_folder'])

        # ====== Privacy-Preserving SLAM: Input blur mode initialization ======
        self._input_blur_enabled = False
        self._input_blur_detector = None
        self._input_blur_radius = 21
        privacy_cfg = cfg.get("privacy", {})
        if privacy_cfg.get("enable", False) and privacy_cfg.get("mode") == "input_blur":
            self._input_blur_enabled = True
            self._input_blur_radius = privacy_cfg.get("blur_radius", 21)
            # Ensure blur radius is odd
            if self._input_blur_radius % 2 == 0:
                self._input_blur_radius += 1
            try:
                from src.privacy.detectors.yolo_detector import YOLOPrivacyDetector
                detector_cfg = privacy_cfg.get("yolo_config", {})
                self._input_blur_detector = YOLOPrivacyDetector(detector_cfg, device)
                self._input_blur_detector.load_model()
                print(f"[Dataset] Input blur mode enabled (radius={self._input_blur_radius})")
            except Exception as e:
                print(f"[Dataset] Failed to initialize input blur detector: {e}")
                self._input_blur_enabled = False


    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    def get_color(self,index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        
        # ====== Privacy-Preserving SLAM: Apply input blur if enabled ======
        if self._input_blur_enabled and self._input_blur_detector is not None:
            color_data = self._apply_input_blur(color_data, index)
        
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data
    
    def _apply_input_blur(self, color_tensor: torch.Tensor, index: int) -> torch.Tensor:
        """
        Apply Gaussian blur to detected private regions in the image.
        
        This is Method B: Input blurring - blur private regions BEFORE SLAM processing.
        Expected to degrade tracking but provides baseline privacy comparison.
        
        Args:
            color_tensor: RGB image tensor (3, H, W) with values in [0, 1]
            index: Frame index (for logging)
        
        Returns:
            Image tensor with private regions blurred
        """
        try:
            # Detect private regions
            result = self._input_blur_detector.detect(color_tensor)
            
            if not result.has_detections:
                return color_tensor
            
            mask = result.combined_mask  # (H, W)
            
            # Convert to numpy for OpenCV blur (faster than torch)
            img_np = (color_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Apply Gaussian blur to entire image
            blurred = cv2.GaussianBlur(
                img_np,
                (self._input_blur_radius, self._input_blur_radius),
                0  # sigma computed from kernel size
            )
            
            # Composite: use blurred for private regions, original elsewhere
            mask_np = mask.cpu().numpy()
            mask_3d = np.stack([mask_np > 0.5] * 3, axis=-1)
            
            result_np = np.where(mask_3d, blurred, img_np)
            
            # Convert back to tensor
            result_tensor = torch.from_numpy(result_np).permute(2, 0, 1).float() / 255.0
            
            return result_tensor.to(color_tensor.device)
            
        except Exception as e:
            # On error, return original image
            print(f"[Dataset] Input blur failed for frame {index}: {e}")
            return color_tensor

    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H   
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge   
        return intrinsic 
    
    def get_intrinsic_full_resol(self):
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge_full
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge_full
        return intrinsic 
    
    def get_color_full_resol(self,index):
        # not used now
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data_fullsize = torch.from_numpy(color_data_fullsize).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data_fullsize = color_data_fullsize.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge_full > 0:
            edge = self.W_edge_full
            color_data_fullsize = color_data_fullsize[:, :, :, edge:-edge]

        if self.H_edge_full > 0:
            edge = self.H_edge_full
            color_data_fullsize = color_data_fullsize[:, :, edge:-edge, :]
        return color_data_fullsize


    def __getitem__(self, index):
        color_data = self.get_color(index)

        depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
        if depth_data_fullsize is not None:
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            outsize = (self.H_out_with_edge, self.W_out_with_edge)
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]
        else:
            depth_data = torch.zeros(color_data.shape[-2:])


        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            depth_data = depth_data[:, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            depth_data = depth_data[edge:-edge, :]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float() #torch.from_numpy(np.linalg.inv(self.poses[0]) @ self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, pose


class Replica(BaseDataset):
    """This is from splat-slam, never test it (todo)"""
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    """This is from splat-slam, never test it (todo)"""
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)
        stride = cfg['stride']
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, device)
        # frame_rate is set to be 32 in MonoGS, we make it to 60 to avoid less frame dropped
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=60, pose_correct_bonn = cfg['dataset']=='bonn_dynamic')
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1, pose_correct_bonn=False):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=0)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])

            if pose_correct_bonn:
                c2w = self.correct_gt_pose_bonn(c2w)

            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            poses += [c2w]

        self.w2c_first_pose = inv_pose

        return images, depths, poses
    
    def correct_gt_pose_bonn(self, T):
        """Specific operation for Bonn dynamic dataset"""
        Tm = np.array([[1.0157, 0.1828, -0.2389, 0.0113],
               [0.0009, -0.8431, -0.6413, -0.0098],
               [-0.3009, 0.6147, -0.8085, 0.0111],
               [0, 0, 0, 1]])
        T_ROS = np.zeros((4,4))
        T_ROS[0,0] = -1
        T_ROS[1,2] = 1
        T_ROS[2,1] = 1
        T_ROS[3,3] = 1

        return T_ROS.T @ T @ T_ROS @ Tm

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

class SevenScenes(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(SevenScenes, self).__init__(cfg, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/seq-01/*.color.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/seq-01/*.depth.png'))
        
        scene_name = os.path.basename(self.input_folder)
        pose_data = np.loadtxt(os.path.join(self.input_folder, f'../{scene_name}.txt'),dtype=np.unicode_)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        self.poses = []
        assert len(self.color_paths) == len(pose_vecs), "Number of images and poses do not match"
        inv_pose = None
        for i in range(len(self.color_paths)):
            c2w = self.pose_matrix_from_quaternion(pose_vecs[i])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            self.poses += [c2w]


        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
class RGB_NoPose(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(RGB_NoPose, self).__init__(cfg, device)
        # Try different image patterns and extensions
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/frame*.png'))
        if len(self.color_paths) == 0:
            self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/*.png'))
        if len(self.color_paths) == 0:
            self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/*.jpg'))
        if len(self.color_paths) == 0:
            self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/frame*.jpg'))
        
        self.depth_paths = None
        self.poses = None

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.n_img = len(self.color_paths)
        
        if self.n_img == 0:
            raise ValueError(f"No RGB images found in {self.input_folder}/rgb/. "
                           f"Searched for patterns: frame*.png, *.png, *.jpg, frame*.jpg")
        
        print(f"INFO: {self.n_img} images loaded from {self.input_folder}/rgb/")


class S3E_RGBD_IMU(BaseDataset):
    """
    Dataset loader for S3E format with RGB-D-IMU data.
    Extends BaseDataset to provide IMU measurements synchronized with RGB frames.
    
    Supports optional depth loading via config:
    - data.has_depth: True/False (whether actual depth maps exist)
    - data.depth_folder: folder name containing depth images (default: 'depth')
    """
    def __init__(self, cfg, device='cuda:0'):
        super(S3E_RGBD_IMU, self).__init__(cfg, device)
        
        # Check if actual depth data exists
        self.has_depth = cfg.get('data', {}).get('has_depth', False)
        self.depth_folder = cfg.get('data', {}).get('depth_folder', 'depth')
        
        # Load file lists
        image_list = os.path.join(self.input_folder, "rgb.txt")
        depth_list = os.path.join(self.input_folder, "depth.txt")
        imu_list = os.path.join(self.input_folder, "imu.txt")
        pose_list = os.path.join(self.input_folder, "groundtruth.txt")
        
        # Parse data files
        image_data = self._parse_list(image_list, skiprows=0)
        depth_data = self._parse_list(depth_list, skiprows=0)
        imu_data = self._parse_list(imu_list, skiprows=1)  # Skip header
        pose_data = self._parse_list(pose_list, skiprows=1)  # Skip header
        
        # Extract timestamps
        self.tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_imu = imu_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64) if pose_data.shape[0] > 0 else None
        
        # Store IMU measurements (columns: timestamp, ori(4), ori_cov(9), ang(3), ang_cov(9), acc(3), acc_cov(9))
        self.imu_vecs = imu_data[:, 1:].astype(np.float64)  # All 36 columns after timestamp
        self.tstamp_imu = tstamp_imu
        
        # Associate frames with IMU data
        associations = self._associate_frames(
            self.tstamp_image, tstamp_depth, tstamp_pose, tstamp_imu
        )
        
        # Build paths and IMU associations
        self.color_paths = []
        self.depth_paths = []
        self.poses = []
        self.imu_indices = []  # Store IMU index ranges for each frame
        
        for assoc in associations:
            # Handle both cases: with pose (i, j, k, imu_idx_range) and without (i, j, imu_idx_range)
            if len(assoc) == 4:
                i, j, k, imu_idx_range = assoc
                # Load pose if available
                pose_vec = pose_data[k, 1:].astype(np.float64)
                c2w = self._pose_matrix_from_quaternion(pose_vec)
                self.poses.append(torch.from_numpy(c2w).float())
            else:
                i, j, imu_idx_range = assoc
                self.poses.append(None)
            
            self.color_paths.append(os.path.join(self.input_folder, image_data[i, 1]))
            
            # Conditionally build depth paths based on config
            if self.has_depth:
                # Use actual depth maps from depth folder
                depth_filename = os.path.basename(image_data[i, 1])  # Use same filename as RGB
                depth_path = os.path.join(self.input_folder, self.depth_folder, depth_filename)
                self.depth_paths.append(depth_path)
            else:
                # No depth available - set to None, will be handled in __getitem__
                self.depth_paths.append(None)
            
            self.imu_indices.append(imu_idx_range)
        
        # Apply stride
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)
        
        print(len(self.poses))
        print(len([p for p in self.poses if p is not None]))
        
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride] if self.poses else None
        # Don't subsample imu_indices yet - need to recalculate for strided frames
        self.image_timestamps = self.tstamp_image[:max_frames][::stride]
        
        # CRITICAL: Recalculate IMU indices for strided frames
        # With stride=2, we want IMU data between frame[i] and frame[i+2], not frame[i] and frame[i+1]
        self.imu_indices = []
        for i in range(len(self.image_timestamps)):
            if i == 0:
                # First frame: IMU data from start up to current time
                t_prev = tstamp_imu[0]
            else:
                # Previous strided frame's timestamp
                t_prev = self.image_timestamps[i-1]
            
            t_curr = self.image_timestamps[i]
            
            # Find IMU samples in range [t_prev, t_curr]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_curr)
            imu_idx_range = np.where(imu_mask)[0]
            
            if len(imu_idx_range) == 0:
                print(f"WARNING: No IMU data found between timestamps {t_prev} and {t_curr}. Using closest IMU sample.")
                # No IMU data in range, use closest
                imu_end = np.argmin(np.abs(tstamp_imu - t_curr))
                imu_idx_range = np.array([imu_end])

            if i % 10 == 0:
                print(f"Frame {i}: IMU indices from {imu_idx_range[0]} to {imu_idx_range[-1]} (len: {len(imu_idx_range)}) for timestamps {t_prev:.3f} to {t_curr:.3f}")
            
            self.imu_indices.append(imu_idx_range)
        
        self.n_img = len(self.color_paths)
        
        # Load camera-to-IMU transformation
        self.c2i_transform = self._load_c2i_transform()
        
        # Create poses dictionary for evaluation (poses indexed by timestamp)
        # Keep poses_list for __getitem__ indexing, create poses dict for eval
        self.poses_list = self.poses  # Save list version for __getitem__
        self.poses = {}
        for idx, (timestamp, pose) in enumerate(zip(self.image_timestamps, self.poses_list)):
            if pose is not None:
                # Convert to numpy if it's a tensor
                pose_np = pose.cpu().numpy() if torch.is_tensor(pose) else pose
                self.poses[int(timestamp)] = pose_np
        
        depth_status = "with depth" if self.has_depth else "without depth (monocular)"
        print(f"INFO: Loaded {self.n_img} RGB-IMU frames {depth_status} from {self.input_folder}")
        print(f"INFO: IMU data ranges from {tstamp_imu[0]:.3f} to {tstamp_imu[-1]:.3f}")
        if len(self.poses) > 0:
            print(f"INFO: {len(self.poses)} ground truth poses available for evaluation")
    
    def _parse_list(self, filepath, skiprows=0):
        """Read list data from file"""
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            return np.array([]).reshape(0, 0)
        data = np.loadtxt(filepath, delimiter=" ", dtype=str, skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    
    def _associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, tstamp_imu, max_dt=0.05):
        """
        Associate images, depths, poses, and IMU measurements using timestamps.
        Returns list of tuples: (img_idx, depth_idx, [pose_idx], imu_idx_range)
        
        Note: For monocular SLAM without ground truth poses, we still accept frames
        even if pose association fails (pose will be None in that case).
        """
        associations = []
        imu_start = 0
        
        # Use larger tolerance for pose association (ground truth may be at lower frequency)
        max_dt_pose = 0.6  # 600ms tolerance for 1Hz pose data
        
        for i, t_img in enumerate(tstamp_image):
            # Find closest depth
            j = np.argmin(np.abs(tstamp_depth - t_img))
            if np.abs(tstamp_depth[j] - t_img) > max_dt:
                continue
            
            # Find closest pose (if available) - but don't skip frame if pose missing
            pose_idx = None
            if tstamp_pose is not None and len(tstamp_pose) > 0:
                k = np.argmin(np.abs(tstamp_pose - t_img))
                if np.abs(tstamp_pose[k] - t_img) <= max_dt_pose:
                    pose_idx = k
                # Don't skip frame if pose not found - we still want the RGB+IMU data
            
            # Find IMU measurements between previous and current frame
            # For the first frame, use IMU data from start up to current time
            if i == 0:
                t_prev = tstamp_imu[0]
            else:
                t_prev = tstamp_image[i-1]
            
            # Find IMU samples in range [t_prev, t_img]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_img)
            imu_idx_range = np.where(imu_mask)[0]
            
            if len(imu_idx_range) == 0:
                # No IMU data in range, use closest
                imu_end = np.argmin(np.abs(tstamp_imu - t_img))
                imu_idx_range = np.array([imu_end])
            
            # Always include frame with RGB+IMU, pose is optional
            if pose_idx is not None:
                associations.append((i, j, pose_idx, imu_idx_range))
            else:
                associations.append((i, j, imu_idx_range))
        
        return associations
    
    def _pose_matrix_from_quaternion(self, pvec):
        """Convert (tx, ty, tz, qx, qy, qz, qw) to 4x4 pose matrix"""
        from scipy.spatial.transform import Rotation
        
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
    def _load_c2i_transform(self):
        """
        Load camera-to-IMU transformation from tf.txt
        
        NOTE: tf.txt contains camera-to-IMU transform (despite the official 
        calibration file naming it 'Tic'). We use it directly without inversion.
        The comment in the calibration file says "from left camera to imu" which
        is camera-to-IMU (Tci), matching the Tlc "camera to lidar" convention.
        """
        tf_path = os.path.join(self.input_folder, "tf.txt")
        if not os.path.exists(tf_path):
            print("WARNING: tf.txt not found, using identity transform for camera-to-IMU")
            return torch.eye(4).float().to(self.device)
        
        tf_data = self._parse_list(tf_path, skiprows=3)  # Skip header lines
        if tf_data.shape[0] == 0:
            return torch.eye(4).float().to(self.device)
        
        # Parse: tx ty tz qx qy qz qw
        # This is ALREADY camera-to-IMU (Tci), use directly
        tf_vec = tf_data[0].astype(np.float64)
        c2i = self._pose_matrix_from_quaternion(tf_vec)
        
        return torch.from_numpy(c2i).float().to(self.device)
    
    def get_imu_data(self, index):
        """
        Get IMU measurements for a specific frame.
        Returns: dict with 'timestamps', 'angular_velocity', 'linear_acceleration', 'orientation'
        
        S3E IMU Data Format (37 columns total):
        - Column 0: timestamp
        - Columns 1-4: orientation quaternion [qx, qy, qz, qw]
        - Columns 5-13: orientation covariance (3x3 flattened)
        - Columns 14-16: angular velocity [wx, wy, wz] rad/s
        - Columns 17-25: angular velocity covariance (3x3 flattened)
        - Columns 26-28: linear acceleration [ax, ay, az] m/s²
        - Columns 29-37: linear acceleration covariance (3x3 flattened)
        """
        imu_idx_range = self.imu_indices[index]
        
        if len(imu_idx_range) == 0:
            return None
        
        # Extract relevant columns from imu_vecs (which starts from column 1, timestamp excluded)
        imu_measurements = self.imu_vecs[imu_idx_range, :]
        
        imu_data = {
            'timestamps': torch.from_numpy(self.tstamp_imu[imu_idx_range]).float(),
            # Orientation quaternion (columns 1-4 in file → 0-3 in imu_vecs)
            'orientation': torch.from_numpy(imu_measurements[:, 0:4]).float(),  # [qx, qy, qz, qw]
            # Angular velocity (columns 14-16 in file → 13-15 in imu_vecs)
            'angular_velocity': torch.from_numpy(imu_measurements[:, 13:16]).float(),  # [wx, wy, wz] rad/s
            # Linear acceleration (columns 26-28 in file → 25-27 in imu_vecs)
            'linear_acceleration': torch.from_numpy(imu_measurements[:, 25:28]).float(),  # [ax, ay, az] m/s²
            # Camera-to-IMU transform (same for all samples)
            'c2i_transform': self.c2i_transform,
        }
        
        return imu_data
    
    def __getitem__(self, index):
        """
        Returns: (index, color, depth, intrinsic, pose, imu_data)
        Note: depth will be None if has_depth=False in config
        """
        color_data = self.get_color(index)
        
        # Only load depth if has_depth is True and path exists
        if self.has_depth and self.depth_paths[index] is not None:
            depth_data_fullsize = self.depthloader(index, self.depth_paths, self.png_depth_scale)
            
            if depth_data_fullsize is not None:
                H, W = depth_data_fullsize.shape
                depth_data = cv2.resize(depth_data_fullsize, (self.W_out, self.H_out), interpolation=cv2.INTER_NEAREST)
                depth_data = torch.from_numpy(depth_data).float()
            else:
                depth_data = None
        else:
            # No depth available
            depth_data = None
        
        intrinsic = self.get_intrinsic()
        
        # Use poses_list for index-based access (poses dict is for eval with timestamps)
        pose = self.poses_list[index] if hasattr(self, 'poses_list') and self.poses_list is not None else None
        
        # Get IMU data for this frame
        imu_data = self.get_imu_data(index)
        
        return index, color_data, depth_data, intrinsic, pose, imu_data


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "bonn_dynamic": TUM_RGBD,
    "wild_slam_mocap": TUM_RGBD,
    "7scenes": SevenScenes,
    "wild_slam_iphone": RGB_NoPose,
    "s3e_rgbd_imu": S3E_RGBD_IMU
}


class VIODEDataset(BaseDataset):
    """
    Dataset loader for VIODE simulated dataset format (parking_lot/...).
    Closely follows S3E_RGBD_IMU structure but adapted for VIODE format.
    
    Expected structure under input_folder:
      groundtruth.txt  # timestamp tx ty tz qx qy qz qw
      imu.txt          # timestamp qx qy qz qw wx wy wz ax ay az
      rgb.txt          # timestamp relative_path (e.g. left/000001.png)
      left/            # left camera images
    
    IMU format: timestamp qx qy qz qw wx wy wz ax ay az (10 columns)
    - Columns 1-4: orientation quaternion [qx, qy, qz, qw]
    - Columns 5-7: angular velocity [wx, wy, wz] rad/s
    - Columns 8-10: linear acceleration [ax, ay, az] m/s²
    """
    def __init__(self, cfg, device='cuda:0'):
        super(VIODEDataset, self).__init__(cfg, device)
        self.cfg = cfg
        
        # Load file lists
        image_list = os.path.join(self.input_folder, "rgb.txt")
        imu_list = os.path.join(self.input_folder, "imu.txt")
        pose_list = os.path.join(self.input_folder, "groundtruth.txt")
        
        # Parse data files
        image_data = self._parse_list(image_list, skiprows=0)
        imu_data = self._parse_list(imu_list, skiprows=0)
        pose_data = self._parse_list(pose_list, skiprows=0)
        
        # Extract timestamps
        self.tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_imu = imu_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64) if pose_data.shape[0] > 0 else None
        
        # Store IMU measurements (columns after timestamp)
        # Format: qx qy qz qw wx wy wz ax ay az (9 columns)
        self.imu_vecs = imu_data[:, 1:].astype(np.float64)
        self.tstamp_imu = tstamp_imu
        
        # Build image paths and IMU associations
        self.color_paths = []
        self.poses = []
        self.imu_indices = []
        
        # Build associations between images, poses, and IMU
        for i in range(len(self.tstamp_image)):
            t_img = self.tstamp_image[i]
            
            # Image path
            img_rel_path = image_data[i, 1]
            img_path = os.path.join(self.input_folder, img_rel_path)
            self.color_paths.append(img_path)
            
            # Find closest pose
            if tstamp_pose is not None and len(tstamp_pose) > 0:
                pose_idx = np.argmin(np.abs(tstamp_pose - t_img))
                if np.abs(tstamp_pose[pose_idx] - t_img) < 0.1:  # 100ms tolerance
                    pose_vec = pose_data[pose_idx, 1:].astype(np.float64)
                    # Format: tx ty tz qx qy qz qw (7 elements)
                    if pose_vec.size == 7:
                        pose = self._pose_matrix_from_quaternion(pose_vec)
                    else:
                        pose = np.eye(4)
                else:
                    pose = None
            else:
                pose = None
            self.poses.append(pose)
            
            # Find IMU samples between this frame and next
            if i == 0:
                t_prev = tstamp_imu[0]
            else:
                t_prev = self.tstamp_image[i-1]
            
            t_curr = t_img
            
            # Find IMU samples in range [t_prev, t_curr]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_curr)
            imu_idx_range = np.where(imu_mask)[0]
            
            if len(imu_idx_range) == 0:
                # No IMU data in range, use closest
                imu_end = np.argmin(np.abs(tstamp_imu - t_curr))
                imu_idx_range = np.array([imu_end])
            
            self.imu_indices.append(imu_idx_range)
        
        # Apply stride and max_frames
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)
        
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.image_timestamps = self.tstamp_image[:max_frames][::stride]
        
        # Recalculate IMU indices for strided frames
        self.imu_indices = []
        for i in range(len(self.image_timestamps)):
            t_curr = self.image_timestamps[i]
            
            if i == 0:
                # For first frame, use IMU data from start to current time
                t_prev = tstamp_imu[0]
            else:
                t_prev = self.image_timestamps[i-1]
            
            # Find IMU samples in [t_prev, t_curr]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_curr)
            imu_idx_range = np.where(imu_mask)[0]
            
            if len(imu_idx_range) == 0:
                imu_end = np.argmin(np.abs(tstamp_imu - t_curr))
                imu_idx_range = np.array([imu_end])
            
            self.imu_indices.append(imu_idx_range)
        
        self.n_img = len(self.color_paths)
        
        # Load camera-to-IMU transformation
        self.c2i_transform = self._load_c2i_transform()
        
        # Create poses dictionary for evaluation (poses indexed by timestamp)
        self.poses_list = self.poses  # Save list version for __getitem__
        self.poses = {}
        for idx, (timestamp, pose) in enumerate(zip(self.image_timestamps, self.poses_list)):
            if pose is not None:
                pose_np = pose if isinstance(pose, np.ndarray) else np.array(pose)
                self.poses[float(timestamp)] = pose_np
        
        print(f"INFO: Loaded {self.n_img} VIODE frames from {self.input_folder}")
        print(f"INFO: IMU data ranges from {tstamp_imu[0]:.3f} to {tstamp_imu[-1]:.3f}")
        if len(self.poses) > 0:
            print(f"INFO: {len(self.poses)} ground truth poses available for evaluation")
    
    def _parse_list(self, filepath, skiprows=0):
        """Read list data from file"""
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            return np.array([]).reshape(0, 0)
        data = np.loadtxt(filepath, delimiter=" ", dtype=str, skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    
    def _pose_matrix_from_quaternion(self, pvec):
        """Convert (tx, ty, tz, qx, qy, qz, qw) to 4x4 pose matrix"""
        from scipy.spatial.transform import Rotation
        
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
    def _load_c2i_transform(self):
        """
        Load camera-to-IMU transformation.
        From user's config: body_T_cam0 (IMU/body to camera transform)
        We need the inverse for camera-to-IMU
        """
        # Try loading from config first (body_T_cam0 matrix)
        if 'imu' in self.cfg and 'body_T_cam0' in self.cfg['imu']:
            try:
                body_T_cam0 = self.cfg.get('imu', {}).get('body_T_cam0', None)
                if body_T_cam0 is None:
                    raise ValueError("body_T_cam0 not found in config['imu']")
                # no need to invert since follows Kalibr convention: body_T_cam0 is camera-to-IMU
                c2i = np.array(body_T_cam0, dtype=np.float64).reshape(4, 4)
                print(f"[Dataset] Loaded camera-to-IMU transform from config body_T_cam0")
                return torch.from_numpy(c2i).float().to(self.device)
                
                # body_T_cam = np.array(self.cfg['imu']['body_T_cam0'])
                # if body_T_cam.size == 16:
                #     body_T_cam = body_T_cam.reshape(4, 4)
                #     # This is body(IMU) to camera, we need camera to body(IMU)
                #     cam_T_body = np.linalg.inv(body_T_cam)
                #     return torch.from_numpy(cam_T_body.astype(np.float32)).to(self.device)
            except Exception as e:
                print(f"WARNING: Failed to parse body_T_cam0 from config: {e}")
        
        # Fallback: identity transform
        print("WARNING: Using identity transform for camera-to-IMU (set body_T_cam0 in config)")
        return torch.eye(4).float().to(self.device)
    
    def get_imu_data(self, index):
        """
        Get IMU measurements for a specific frame.
        Returns: dict with 'timestamps', 'angular_velocity', 'linear_acceleration', 'orientation'
        
        VIODE IMU Data Format (9 columns after timestamp):
        - Columns 0-3: orientation quaternion [qx, qy, qz, qw]
        - Columns 4-6: angular velocity [wx, wy, wz] rad/s
        - Columns 7-9: linear acceleration [ax, ay, az] m/s²
        """
        imu_idx_range = self.imu_indices[index]
        
        if len(imu_idx_range) == 0:
            return None
        
        # Extract relevant columns from imu_vecs
        imu_measurements = self.imu_vecs[imu_idx_range, :]
        
        imu_data = {
            'timestamps': torch.from_numpy(self.tstamp_imu[imu_idx_range]).float(),
            # Orientation quaternion (columns 0-3)
            'orientation': torch.from_numpy(imu_measurements[:, 0:4]).float(),  # [qx, qy, qz, qw]
            # Angular velocity (columns 4-6)
            'angular_velocity': torch.from_numpy(imu_measurements[:, 4:7]).float(),  # [wx, wy, wz] rad/s
            # Linear acceleration (columns 7-9)
            'linear_acceleration': torch.from_numpy(imu_measurements[:, 7:10]).float(),  # [ax, ay, az] m/s²
            # Camera-to-IMU transform (same for all samples)
            'c2i_transform': self.c2i_transform,
        }
        
        return imu_data
    
    def __getitem__(self, index):
        """
        Returns: (index, color, depth, intrinsic, pose, imu_data)
        Note: depth will be None for VIODE (no depth available)
        """
        color_data = self.get_color(index)
        
        # VIODE has no depth
        depth_data = None
        
        intrinsic = self.get_intrinsic()
        
        # Use poses_list for index-based access
        pose = self.poses_list[index] if hasattr(self, 'poses_list') and self.poses_list is not None else None
        
        # Get IMU data for this frame
        imu_data = self.get_imu_data(index)
        
        return index, color_data, depth_data, intrinsic, pose, imu_data


# Register new dataset key
dataset_dict["viode"] = VIODEDataset


class ADVIODataset(BaseDataset):
    """
    Dataset loader for ADVIO (Advanced Visual-Inertial Odometry) dataset.
    Converted to MainSLAM format using scripts/convert_advio.py.

    Expected structure under input_folder:
      groundtruth.txt  # timestamp tx ty tz qx qy qz qw
      imu.txt          # timestamp qx qy qz qw wx wy wz ax ay az
      rgb.txt          # timestamp relative_path (e.g. left/00001.png)
      left/            # extracted images

    IMU format (10 columns after timestamp):
    - Columns 0-3: orientation quaternion [qx, qy, qz, qw]
    - Columns 4-6: angular velocity [wx, wy, wz] rad/s
    - Columns 7-9: linear acceleration [ax, ay, az] m/s^2

    Calibration for ADVIO sequences 13-17 (office environments):
    - Image size: 1280x720 (portrait, but stored as 720x1280 landscape in some cases)
    - Intrinsics: fx=1082.4, fy=1084.4, cx=364.68, cy=643.31
    - T_cam_imu provided in calibration file
    """
    def __init__(self, cfg, device='cuda:0'):
        super(ADVIODataset, self).__init__(cfg, device)
        self.cfg = cfg

        # Load file lists
        image_list = os.path.join(self.input_folder, "rgb.txt")
        imu_list = os.path.join(self.input_folder, "imu.txt")
        pose_list = os.path.join(self.input_folder, "groundtruth.txt")

        # Parse data files
        image_data = self._parse_list(image_list, skiprows=0)
        imu_data = self._parse_imu_list(imu_list)
        pose_data = self._parse_list(pose_list, skiprows=0)

        # Extract timestamps
        self.tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_imu = imu_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64) if pose_data.shape[0] > 0 else None

        # Store IMU measurements (columns after timestamp)
        # Format: qx qy qz qw wx wy wz ax ay az (10 columns)
        self.imu_vecs = imu_data[:, 1:].astype(np.float64)
        self.tstamp_imu = tstamp_imu

        # Build image paths and IMU associations
        self.color_paths = []
        self.poses = []
        self.imu_indices = []

        # Build associations between images, poses, and IMU
        max_dt_pose = 0.1  # 100ms tolerance for pose association

        for i in range(len(self.tstamp_image)):
            t_img = self.tstamp_image[i]

            # Image path
            img_rel_path = image_data[i, 1]
            img_full_path = os.path.join(self.input_folder, img_rel_path)
            self.color_paths.append(img_full_path)

            # Find nearest pose
            if tstamp_pose is not None and len(tstamp_pose) > 0:
                k = np.argmin(np.abs(tstamp_pose - t_img))
                if np.abs(tstamp_pose[k] - t_img) <= max_dt_pose:
                    pose_vec = pose_data[k, 1:].astype(np.float64)
                    c2w = self._pose_matrix_from_quaternion(pose_vec)
                    pose = c2w
                else:
                    pose = None
            else:
                pose = None
            self.poses.append(pose)

            # Find IMU samples between this frame and next
            if i == 0:
                t_prev = tstamp_imu[0]
            else:
                t_prev = self.tstamp_image[i-1]

            t_curr = t_img

            # Find IMU samples in range [t_prev, t_curr]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_curr)
            imu_idx_range = np.where(imu_mask)[0]

            if len(imu_idx_range) == 0:
                # No IMU data in range, use closest
                imu_end = np.argmin(np.abs(tstamp_imu - t_curr))
                imu_idx_range = np.array([imu_end])

            self.imu_indices.append(imu_idx_range)

        # Apply stride and max_frames
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.image_timestamps = self.tstamp_image[:max_frames][::stride]

        # Recalculate IMU indices for strided frames
        self.imu_indices = []
        for i in range(len(self.image_timestamps)):
            t_curr = self.image_timestamps[i]

            if i == 0:
                # For first frame, use IMU data from start to current time
                t_prev = tstamp_imu[0]
            else:
                t_prev = self.image_timestamps[i-1]

            # Find IMU samples in [t_prev, t_curr]
            imu_mask = (tstamp_imu >= t_prev) & (tstamp_imu <= t_curr)
            imu_idx_range = np.where(imu_mask)[0]

            if len(imu_idx_range) == 0:
                imu_end = np.argmin(np.abs(tstamp_imu - t_curr))
                imu_idx_range = np.array([imu_end])

            self.imu_indices.append(imu_idx_range)

        self.n_img = len(self.color_paths)

        # Load camera-to-IMU transformation
        self.c2i_transform = self._load_c2i_transform()

        # Create poses dictionary for evaluation (poses indexed by timestamp)
        self.poses_list = self.poses  # Save list version for __getitem__
        self.poses = {}
        for i, timestamp in enumerate(self.image_timestamps):
            pose = self.poses_list[i]
            if pose is not None:
                pose_np = pose if isinstance(pose, np.ndarray) else np.array(pose)
                self.poses[float(timestamp)] = pose_np

        print(f"INFO: Loaded {self.n_img} ADVIO frames from {self.input_folder}")
        print(f"INFO: IMU data ranges from {tstamp_imu[0]:.3f} to {tstamp_imu[-1]:.3f}")
        if len(self.poses) > 0:
            print(f"INFO: {len(self.poses)} ground truth poses available for evaluation")

    def _parse_list(self, filepath, skiprows=0):
        """Read list data from file (space-delimited)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
        data = np.loadtxt(filepath, delimiter=" ", dtype=str, skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def _parse_imu_list(self, filepath):
        """Read IMU data from file (space-delimited, numeric)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required IMU file not found: {filepath}")
        return np.loadtxt(filepath, dtype=np.float64)

    def _pose_matrix_from_quaternion(self, pvec):
        """Convert (tx, ty, tz, qx, qy, qz, qw) to 4x4 pose matrix (camera-to-world)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        # Rotation from quaternion (scalar-last: qx, qy, qz, qw)
        R = Rotation.from_quat(pvec[3:7]).as_matrix()
        pose[:3, :3] = R
        pose[:3, 3] = pvec[:3]
        return pose

    def _load_c2i_transform(self):
        """
        Load camera-to-IMU transformation from config or use ADVIO default.

        ADVIO calibration T_cam_imu for sequences 13-17:
        This transform goes from camera frame to IMU frame.
        """
        # Try loading from config first (body_T_cam0 matrix)
        if 'imu' in self.cfg and 'body_T_cam0' in self.cfg['imu']:
            try:
                body_T_cam0 = self.cfg.get('imu', {}).get('body_T_cam0', None)
                if body_T_cam0 is not None:
                    c2i = np.array(body_T_cam0, dtype=np.float64).reshape(4, 4)
                    print(f"[ADVIO] Loaded camera-to-IMU transform from config body_T_cam0")
                    return torch.from_numpy(c2i).float().to(self.device)
            except Exception as e:
                print(f"WARNING: Failed to parse body_T_cam0 from config: {e}")

        # Fallback: Use ADVIO default calibration for sequences 13-17
        # T_cam_imu from iphone-03.yaml
        T_cam_imu = np.array([
            [0.9999763379093255, -0.004079205042965442, -0.005539287650170447, -0.008977668364731128],
            [-0.004066386342107199, -0.9999890330121858, 0.0023234365646622014, 0.07557012320238939],
            [-0.00554870467502187, -0.0023008567036498766, -0.9999819588046867, -0.005545773942541918],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        print(f"[ADVIO] Using default T_cam_imu calibration for sequences 13-17")
        return torch.from_numpy(T_cam_imu).float().to(self.device)

    def get_imu_data(self, index):
        """
        Get IMU measurements for a specific frame.
        Returns: dict with 'timestamps', 'angular_velocity', 'linear_acceleration', 'orientation'

        ADVIO IMU Data Format (10 columns after timestamp):
        - Columns 0-3: orientation quaternion [qx, qy, qz, qw]
        - Columns 4-6: angular velocity [wx, wy, wz] rad/s
        - Columns 7-9: linear acceleration [ax, ay, az] m/s^2
        """
        imu_idx_range = self.imu_indices[index]

        if len(imu_idx_range) == 0:
            return None

        # Extract relevant columns from imu_vecs
        imu_measurements = self.imu_vecs[imu_idx_range, :]

        imu_data = {
            'timestamps': torch.from_numpy(self.tstamp_imu[imu_idx_range]).float(),
            # Orientation quaternion (columns 0-3)
            'orientation': torch.from_numpy(imu_measurements[:, 0:4]).float(),  # [qx, qy, qz, qw]
            # Angular velocity (columns 4-6)
            'angular_velocity': torch.from_numpy(imu_measurements[:, 4:7]).float(),  # [wx, wy, wz] rad/s
            # Linear acceleration (columns 7-9)
            'linear_acceleration': torch.from_numpy(imu_measurements[:, 7:10]).float(),  # [ax, ay, az] m/s^2
            # Camera-to-IMU transform (same for all samples)
            'c2i_transform': self.c2i_transform,
        }

        return imu_data

    def __getitem__(self, index):
        """
        Returns: (index, color, depth, intrinsic, pose, imu_data)
        Note: depth will be None for ADVIO (monocular VIO dataset)
        """
        color_data = self.get_color(index)

        # ADVIO has no depth (monocular VIO)
        depth_data = torch.zeros(color_data.shape[-2:])

        intrinsic = self.get_intrinsic()

        # Use poses_list for index-based access
        pose = self.poses_list[index] if hasattr(self, 'poses_list') and self.poses_list is not None else None

        # Get IMU data for this frame
        imu_data = self.get_imu_data(index)

        return index, color_data, depth_data, intrinsic, pose, imu_data


# Register ADVIO dataset
dataset_dict["advio"] = ADVIODataset
