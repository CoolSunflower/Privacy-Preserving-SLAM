#!/usr/bin/env python3
"""
ADVIO Dataset Converter for MainSLAM

Converts ADVIO dataset format to MainSLAM-compatible format:
- Extracts frames from video using ffmpeg
- Creates rgb.txt (timestamp -> image path mapping)
- Creates groundtruth.txt (TUM format poses)
- Creates imu.txt (synchronized IMU with orientation)

ADVIO Raw Format:
- iphone/frames.mov: H.264 video (60fps, 1280x720 portrait)
- iphone/frames.csv: timestamp,frame_index
- iphone/accelerometer.csv: timestamp,ax,ay,az (m/s^2, despite docs saying g's)
- iphone/gyro.csv: timestamp,wx,wy,wz (rad/s)
- iphone/arkit.csv: timestamp,tx,ty,tz,qw,qx,qy,qz (scalar-first quaternion)
- ground-truth/pose.csv: timestamp,tx,ty,tz,qx,qy,qz,qw (scalar-last quaternion)

MainSLAM Expected Format:
- left/*.png or left/*.jpg: extracted images
- rgb.txt: timestamp image_path
- groundtruth.txt: timestamp tx ty tz qx qy qz qw
- imu.txt: timestamp qx qy qz qw wx wy wz ax ay az

Author: ADVIO Dataset Converter
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path


def load_csv(filepath, has_header=False):
    """Load CSV file as numpy array."""
    skiprows = 1 if has_header else 0
    return np.loadtxt(filepath, delimiter=',', skiprows=skiprows)


def slerp_quaternions(query_times, ref_times, ref_quats):
    """
    Spherical linear interpolation of quaternions.

    Args:
        query_times: Times at which to interpolate (N,)
        ref_times: Reference timestamps (M,)
        ref_quats: Reference quaternions (M, 4) in [qx, qy, qz, qw] scipy format

    Returns:
        Interpolated quaternions (N, 4) in [qx, qy, qz, qw] format
    """
    # Create scipy Rotation objects
    rotations = Rotation.from_quat(ref_quats)  # scipy expects [qx, qy, qz, qw]

    # Create Slerp interpolator
    slerp = Slerp(ref_times, rotations)

    # Clamp query times to valid range
    query_times_clamped = np.clip(query_times, ref_times[0], ref_times[-1])

    # Interpolate
    interp_rotations = slerp(query_times_clamped)

    return interp_rotations.as_quat()  # Returns [qx, qy, qz, qw]


def extract_frames(video_path, frames_csv_path, output_dir, image_format='png'):
    """
    Extract frames from video at exact timestamps specified in frames.csv.

    Uses ffmpeg to extract frames based on timestamps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load frame timestamps
    frames_data = load_csv(frames_csv_path)
    timestamps = frames_data[:, 0]
    frame_indices = frames_data[:, 1].astype(int)

    print(f"Extracting {len(timestamps)} frames from video...")
    print(f"  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")

    # Extract all frames at once (more efficient than per-frame extraction)
    # Get video FPS first
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip().rstrip(',')  # Remove trailing comma if present
    if '/' in fps_str:
        parts = fps_str.split('/')
        num, den = int(parts[0]), int(parts[1])
        video_fps = num / den
    else:
        video_fps = float(fps_str)
    print(f"  Video FPS: {video_fps:.2f}")

    # Extract all frames
    extract_cmd = [
        'ffmpeg', '-i', video_path,
        '-q:v', '2',  # High quality
        '-start_number', '1',
        os.path.join(output_dir, f'%05d.{image_format}')
    ]
    print(f"  Running ffmpeg extraction...")
    result = subprocess.run(extract_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Warning: ffmpeg returned code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")

    # Count extracted frames
    extracted = list(Path(output_dir).glob(f'*.{image_format}'))
    print(f"  Extracted {len(extracted)} frames to {output_dir}")

    return timestamps, frame_indices


def convert_advio_sequence(input_dir, output_dir=None, image_format='png'):
    """
    Convert a single ADVIO sequence to MainSLAM format.

    Args:
        input_dir: Path to ADVIO sequence (e.g., datasets/ADVIO/advio-13)
        output_dir: Output directory (default: same as input_dir)
        image_format: Output image format ('png' or 'jpg')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    print(f"\n{'='*60}")
    print(f"Converting ADVIO sequence: {input_dir.name}")
    print(f"{'='*60}")

    # Define file paths
    video_path = input_dir / 'iphone' / 'frames.mov'
    frames_csv = input_dir / 'iphone' / 'frames.csv'
    accel_csv = input_dir / 'iphone' / 'accelerometer.csv'
    gyro_csv = input_dir / 'iphone' / 'gyro.csv'
    arkit_csv = input_dir / 'iphone' / 'arkit.csv'
    pose_csv = input_dir / 'ground-truth' / 'pose.csv'

    # Check required files exist
    required_files = [frames_csv, accel_csv, gyro_csv, arkit_csv, pose_csv]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    if not video_path.exists():
        print(f"Warning: Video file not found at {video_path}")
        print("Skipping frame extraction - assuming frames already extracted")
        video_exists = False
    else:
        video_exists = True

    # Create output directories
    images_dir = output_dir / 'left'
    os.makedirs(images_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Extract frames from video
    # =========================================================================
    print("\n[Step 1] Extracting frames from video...")

    if video_exists:
        frame_timestamps, frame_indices = extract_frames(
            str(video_path), str(frames_csv), str(images_dir), image_format
        )
    else:
        # Load timestamps from frames.csv
        frames_data = load_csv(str(frames_csv))
        frame_timestamps = frames_data[:, 0]
        frame_indices = frames_data[:, 1].astype(int)
        print(f"  Loaded {len(frame_timestamps)} frame timestamps from frames.csv")

    # =========================================================================
    # Step 2: Create rgb.txt
    # =========================================================================
    print("\n[Step 2] Creating rgb.txt...")

    rgb_txt_path = output_dir / 'rgb.txt'
    with open(rgb_txt_path, 'w') as f:
        for ts, idx in zip(frame_timestamps, frame_indices):
            img_path = f"left/{int(idx):05d}.{image_format}"
            f.write(f"{ts:.6f} {img_path}\n")

    print(f"  Created {rgb_txt_path} with {len(frame_timestamps)} entries")

    # =========================================================================
    # Step 3: Create groundtruth.txt from pose.csv
    # =========================================================================
    print("\n[Step 3] Creating groundtruth.txt...")

    pose_data = load_csv(str(pose_csv))
    # pose.csv format: timestamp, tx, ty, tz, qx, qy, qz, qw (TUM format, scalar-last)
    # This is already in the correct format for MainSLAM!

    gt_txt_path = output_dir / 'groundtruth.txt'
    np.savetxt(str(gt_txt_path), pose_data, fmt='%.9f', delimiter=' ')

    print(f"  Created {gt_txt_path} with {len(pose_data)} poses")
    print(f"  Time range: {pose_data[0, 0]:.3f}s to {pose_data[-1, 0]:.3f}s")

    # =========================================================================
    # Step 4: Create synchronized imu.txt
    # =========================================================================
    print("\n[Step 4] Creating synchronized imu.txt...")

    # Load IMU data
    accel_data = load_csv(str(accel_csv))
    gyro_data = load_csv(str(gyro_csv))
    arkit_data = load_csv(str(arkit_csv))

    print(f"  Accelerometer: {len(accel_data)} samples, {accel_data[0, 0]:.3f}s to {accel_data[-1, 0]:.3f}s")
    print(f"  Gyroscope: {len(gyro_data)} samples, {gyro_data[0, 0]:.3f}s to {gyro_data[-1, 0]:.3f}s")
    print(f"  ARKit: {len(arkit_data)} samples, {arkit_data[0, 0]:.3f}s to {arkit_data[-1, 0]:.3f}s")

    # Use gyroscope timestamps as reference
    imu_timestamps = gyro_data[:, 0]
    gyro_xyz = gyro_data[:, 1:4]  # wx, wy, wz (rad/s)

    # Interpolate accelerometer to gyro timestamps
    accel_timestamps = accel_data[:, 0]
    accel_xyz_raw = accel_data[:, 1:4]  # ax, ay, az (in g's!)

    accel_interp = np.zeros((len(imu_timestamps), 3))
    for i in range(3):
        accel_interp[:, i] = np.interp(imu_timestamps, accel_timestamps, accel_xyz_raw[:, i])

    # ADVIO accelerometer data is already in m/s^2 (magnitude ~9.81 at rest)
    # despite documentation saying it's in g's - verified by checking magnitude
    accel_interp_ms2 = accel_interp

    # Interpolate ARKit orientation to gyro timestamps using SLERP
    arkit_timestamps = arkit_data[:, 0]
    # arkit.csv format: timestamp, tx, ty, tz, qw, qx, qy, qz (scalar-first)
    # Convert to scipy format [qx, qy, qz, qw]
    arkit_quat_wxyz = arkit_data[:, 4:8]  # qw, qx, qy, qz
    arkit_quat_xyzw = np.column_stack([
        arkit_quat_wxyz[:, 1],  # qx
        arkit_quat_wxyz[:, 2],  # qy
        arkit_quat_wxyz[:, 3],  # qz
        arkit_quat_wxyz[:, 0],  # qw
    ])

    # Validate ARKit timestamps are valid for interpolation
    valid_mask = (imu_timestamps >= arkit_timestamps[0]) & (imu_timestamps <= arkit_timestamps[-1])
    print(f"  IMU samples within ARKit range: {valid_mask.sum()} / {len(imu_timestamps)}")

    # SLERP interpolation for orientation
    print("  Performing SLERP interpolation for orientation...")
    orient_interp = slerp_quaternions(imu_timestamps, arkit_timestamps, arkit_quat_xyzw)

    # For samples outside ARKit range, use nearest ARKit orientation
    for i in range(len(imu_timestamps)):
        if imu_timestamps[i] < arkit_timestamps[0]:
            orient_interp[i] = arkit_quat_xyzw[0]
        elif imu_timestamps[i] > arkit_timestamps[-1]:
            orient_interp[i] = arkit_quat_xyzw[-1]

    # Build imu.txt with format: timestamp qx qy qz qw wx wy wz ax ay az
    imu_output = np.column_stack([
        imu_timestamps,          # timestamp
        orient_interp[:, 0],     # qx
        orient_interp[:, 1],     # qy
        orient_interp[:, 2],     # qz
        orient_interp[:, 3],     # qw
        gyro_xyz[:, 0],          # wx
        gyro_xyz[:, 1],          # wy
        gyro_xyz[:, 2],          # wz
        accel_interp_ms2[:, 0],  # ax (m/s^2)
        accel_interp_ms2[:, 1],  # ay (m/s^2)
        accel_interp_ms2[:, 2],  # az (m/s^2)
    ])

    imu_txt_path = output_dir / 'imu.txt'
    np.savetxt(str(imu_txt_path), imu_output, fmt='%.9f', delimiter=' ')

    print(f"  Created {imu_txt_path} with {len(imu_output)} samples")
    print(f"  IMU time range: {imu_timestamps[0]:.3f}s to {imu_timestamps[-1]:.3f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Conversion complete for {input_dir.name}")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  - {rgb_txt_path}")
    print(f"  - {gt_txt_path}")
    print(f"  - {imu_txt_path}")
    print(f"  - {images_dir}/")

    # Check for timestamp alignment
    print(f"\nTimestamp alignment check:")
    print(f"  Frames:      {frame_timestamps[0]:.3f}s - {frame_timestamps[-1]:.3f}s")
    print(f"  Ground truth: {pose_data[0, 0]:.3f}s - {pose_data[-1, 0]:.3f}s")
    print(f"  IMU:         {imu_timestamps[0]:.3f}s - {imu_timestamps[-1]:.3f}s")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert ADVIO dataset to MainSLAM format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single sequence
  python convert_advio.py --input ./datasets/ADVIO/advio-13

  # Convert all sequences in a directory
  python convert_advio.py --input ./datasets/ADVIO --all

  # Convert with custom output directory
  python convert_advio.py --input ./datasets/ADVIO/advio-13 --output ./datasets/ADVIO_converted/advio-13
"""
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input ADVIO sequence directory or parent directory (with --all)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Convert all advio-* sequences in input directory')
    parser.add_argument('--format', '-f', choices=['png', 'jpg'], default='png',
                        help='Output image format (default: png)')
    parser.add_argument('--sequences', '-s', nargs='+', type=int,
                        help='Specific sequence numbers to convert (e.g., 13 14 15 16)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.all or args.sequences:
        # Convert multiple sequences
        if args.sequences:
            seq_dirs = [input_path / f'advio-{i:02d}' for i in args.sequences]
        else:
            seq_dirs = sorted(input_path.glob('advio-*'))

        if not seq_dirs:
            print(f"No ADVIO sequences found in {input_path}")
            sys.exit(1)

        print(f"Found {len(seq_dirs)} sequences to convert")

        for seq_dir in seq_dirs:
            if seq_dir.is_dir():
                try:
                    output_dir = Path(args.output) / seq_dir.name if args.output else None
                    convert_advio_sequence(seq_dir, output_dir, args.format)
                except Exception as e:
                    print(f"Error converting {seq_dir}: {e}")
                    continue
    else:
        # Convert single sequence
        convert_advio_sequence(input_path, args.output, args.format)


if __name__ == '__main__':
    main()
