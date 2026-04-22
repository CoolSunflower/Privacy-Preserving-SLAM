import numpy as np
import torch
from lietorch import SE3
from evo.core.trajectory import PoseTrajectory3D

def eval_traj(output_folder, input_folder, N=-1):
    # Open required files
    data = np.load(output_folder + '/video.npz')
    gt_poses = np.loadtxt(input_folder + '/groundtruth.txt')
    RGB_PATH = input_folder + '/rgb.txt'

    # Print the keys in the .npz file
    # print("Keys in the .npz file:", data.files)
    # Keys in the .npz file: ['poses', 'depths', 'timestamps', 'valid_depth_masks', 'scale']

    # Access and print the shape of the 'poses' array
    poses = data['poses']
    # print("Original shape of 'poses':", poses.shape)
    if N > 0:
        poses = poses[:N]
    # print("Shape of 'poses':", poses.shape)
    # print("First 3 poses:\n", poses[:3])
    # (41, 4, 4) = (N, 4, 4) where N is the number of frames

    # Access and print the shape of the 'timestamps' array
    timestamps = data['timestamps']
    if N > 0:
        timestamps = timestamps[:N]
    # print("Shape of 'timestamps':", timestamps.shape)
    # print("First 3 timestamps:\n", timestamps[:3])
    # (41,) = (N,) where N is the number of frames
    # First 3: [0. 9. 16.] # hmmm wtf --> so these are frame indices, not actual timestamps


    # --------------

    # Now load ground truth .txt file
    # print("Shape of ground truth poses:", gt_poses.shape)
    # print("First 3 ground truth poses:\n", gt_poses[:3])
    # (292, 8) = (M, 8) where M is the number of ground truth frames
    # First value: [ 1.66116388e+09  7.44140326e+05  2.55290270e+06 -1.23890000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]

    # --------------

    # Now we need map the timestamps from the video to the ground truth timestamps
    # Open rgb.txt (This is at 10Hz and groundtruth.txt is at 1Hz)

    rgb_timestamps = []
    with open(RGB_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            timestamp_str, _ = line.split(' ', 1)
            rgb_timestamps.append(float(timestamp_str))
    rgb_timestamps = np.array(rgb_timestamps)  # shape (K,)
    # print("Shape of rgb timestamps:", rgb_timestamps.shape)
    # (2927,) = (K,) where K is the number of rgb frames

    # Map video timestamps to rgb timestamps
    video_timestamps = []
    for vt in timestamps:
        # Find timestamp at that index in rgb_timestamps
        # print(f'Mapping video timestamp index {vt} to rgb timestamp {rgb_timestamps[int(vt)]}')
        video_timestamps.append(rgb_timestamps[int(vt)])
    video_timestamps = np.array(video_timestamps)  # shape (N,)
    # print("Mapped video timestamps shape:", video_timestamps.shape, video_timestamps[:5])
    # (41,) = (N,)

    # Now for each video timestamp, find the closest ground truth pose
    gt_poses_for_video = []
    for vt in video_timestamps:
        # Find closest timestamp in gt_poses (first column)
        gt_timestamps = gt_poses[:, 0]
        closest_idx = np.argmin(np.abs(gt_timestamps - vt))
        gt_poses_for_video.append(gt_poses[closest_idx, 1:])  # Exclude timestamp
    gt_poses_for_video = np.array(gt_poses_for_video)  # shape (N, 7)
    # print("Ground truth poses for video shape:", gt_poses_for_video.shape)
    # print("First 3 ground truth poses for video:\n", gt_poses_for_video[:3])

    # Convert gt_poses_for_video from (tx, ty, tz, qx, qy, qz, qw) to (4,4) SE3 matrices
    def _pose_matrix_from_quaternion(pvec):
        """Convert (tx, ty, tz, qx, qy, qz, qw) to 4x4 pose matrix"""
        from scipy.spatial.transform import Rotation
        
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    c2w = torch.from_numpy(np.array([_pose_matrix_from_quaternion(pose_vec) for pose_vec in gt_poses_for_video]))  # (N, 4, 4)
    # print("Converted ground truth poses to SE3 matrices shape:", c2w.shape)
    gt_poses_for_video = c2w  # Convert to SE3 type from lietorch

    # Now we have:
    # - poses: estimated poses from the .npz file (N, 4, 4)
    # - gt_poses_for_video: ground truth poses corresponding to the video timestamps (N, 4, 4)

    traj_est = PoseTrajectory3D(poses_se3=poses, timestamps=video_timestamps)
    traj_ref = PoseTrajectory3D(poses_se3=gt_poses_for_video, timestamps=video_timestamps)

    from evo.core import sync, metrics
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    # print("\n============ Manual Evaluation for Keyframe ============\n")
    # print("Alignment results:")
    # print("Rotation:\n", r_a)
    # print("Translation:\n", t_a)
    # print("Scale:", s)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()
    # print("ATE statistics:", ape_statistics)

    # Also write above to a file
    with open(output_folder + 'kf_manual_eval.txt', "w") as f:
        f.write("\n============ Manual Evaluation for Keyframe ============\n")
        f.write("\nAlignment results:\n")
        f.write(f"Rotation:\n{r_a}\n")
        f.write(f"Translation:\n{t_a}\n")
        f.write(f"Scale: {s}\n")
        f.write(f"ATE Statistics:{ape_statistics}")

    # Now to plot the trajectories
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    def plot_trajectory(ax, traj, label, color):
        # AttributeError: 'PoseTrajectory3D' object has no attribute 'get_positions'
        positions = traj.positions_xyz
        # print(f'Plotting trajectory {label} with {positions.shape[0]} points.', positions[:3])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=label, color=color)
    plot_trajectory(ax, traj_ref, label='Ground Truth', color='g')
    plot_trajectory(ax, traj_est, label='Estimated', color='r')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    if N > 0:
        plt.savefig(output_folder + f'kf_trajectory_N{N}.png')
    plt.savefig(output_folder + 'kf_trajectory.png')
    plt.close()

    # Also plot xy plot of trajectory
    plt.figure()
    def plot_trajectory_xy(traj, label, color):
        positions = traj.positions_xyz
        plt.plot(positions[:, 0], positions[:, 1], label=label, color=color)
    plot_trajectory_xy(traj_ref, label='Ground Truth', color='g')
    plot_trajectory_xy(traj_est, label='Estimated', color='r')
    plt.title('Trajectory Comparison (XY Plane)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.savefig(output_folder + 'kf_trajectory_xy.png')
    if N > 0:
        plt.savefig(output_folder + f'kf_trajectory_xy_N{N}.png')
    plt.close()

    return ape_statistics

if __name__ == "__main__":
    output_folders = [
        '../output/vio_parking_lot_fe_imu_full_05decay/parking_lot/',
        '../output/vio_parking_lot_fe_imu_full_1decay/parking_lot/',
        '../output/vio_parking_lot_fe_imu_mamba_full/parking_lot/',
        '../output/vio_parking_lot_base_full/parking_lot/',
        '../output/vio_parking_lot_fe_imu_depth_corr_loss/parking_lot/',
    ]
    N_vals_viode = [50, 100, 125, 150, 175, 200, -1]
    inputs_folders = [
        '../datasets/VIODE/parking_lot/',
        '../datasets/VIODE/parking_lot/',
        '../datasets/VIODE/parking_lot/',
        '../datasets/VIODE/parking_lot/',
        '../datasets/VIODE/parking_lot/',
        '../datasets/S3E/playground_1/',
        '../datasets/S3E/playground_1/',
        '../datasets/S3E/playground_1/',
    ]
    input_folder = '../datasets/VIODE/parking_lot/'
    for N in N_vals_viode:
        for i in range(len(output_folders)):
            output_folder = output_folders[i]
            input_folder = inputs_folders[i]
            stats = eval_traj(output_folder, input_folder, N=N)
            print(f"{output_folder}, {N}:", stats['rmse'])
        print()