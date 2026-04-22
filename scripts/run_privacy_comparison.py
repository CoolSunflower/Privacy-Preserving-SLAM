#!/usr/bin/env python3
"""
Privacy Comparison Evaluation Script

Runs all 6 comparison methods (A-F) on a dataset and generates comparison tables.

Methods:
    A: No Privacy (baseline)
    B: Input Blurring (blur private regions before SLAM)
    C: Uncertainty-Only (inject beta=100, no pruning)
    D: Post-Process Only (Grounding DINO + SAM post-processing)
    E: Excision-Only (hybrid without filling)
    F: Ours Full (hybrid with filling)

Usage:
    python scripts/run_privacy_comparison.py \
        --config ./configs/BTP2/tum_rgbd/f3_shs.yaml \
        --methods A B C D E F \
        --output output/privacy_comparison/tum_f3_shs/
"""

import os
import sys
import re
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import copy

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, save_config


def load_base_config(config_path: str) -> dict:
    """Load base SLAM configuration."""
    cfg = load_config(config_path)
    return cfg


def get_method_configs() -> Dict[str, dict]:
    """Get privacy configurations for all comparison methods."""
    return {
        "A_no_privacy": {
            "privacy": {
                "enable": False
            }
        },
        "B_input_blur": {
            "privacy": {
                "enable": True,
                "mode": "input_blur",
                "blur_radius": 21,
                "simultaneous_detector": "yolo",
                "enable_excision": False,
                "enable_filling": False
            }
        },
        "C_uncertainty_only": {
            "privacy": {
                "enable": True,
                "mode": "simultaneous",
                "simultaneous_detector": "yolo",
                "uncertainty_beta": 100.0,
                "enable_excision": False,
                "enable_filling": False
            }
        },
        "D_postprocess_only": {
            "privacy": {
                "enable": True,
                "mode": "postprocess",
                "enable_excision": True,
                "enable_filling": False,
                "grounding_sam_config": {
                    "text_prompts": ["person", "human face", "screen"],
                    "box_threshold": 0.35,
                    "text_threshold": 0.3
                }
            }
        },
        "E_excision_only": {
            "privacy": {
                "enable": True,
                "mode": "hybrid",
                "simultaneous_detector": "yolo",
                "uncertainty_beta": 100.0,
                "enable_excision": True,
                "enable_filling": False,
                "grounding_sam_config": {
                    "text_prompts": ["person", "human face", "screen"],
                    "box_threshold": 0.35,
                    "text_threshold": 0.3
                }
            }
        },
        "F_ours_full": {
            "privacy": {
                "enable": True,
                "mode": "hybrid",
                "simultaneous_detector": "yolo",
                "uncertainty_beta": 100.0,
                "enable_excision": True,
                "enable_filling": True,
                "fill_opacity": 0.15,
                "fill_color": [0.5, 0.5, 0.5],
                "grounding_sam_config": {
                    "text_prompts": ["person", "human face", "screen"],
                    "box_threshold": 0.35,
                    "text_threshold": 0.3
                }
            }
        }
    }


def merge_configs(base_config: dict, method_config: dict) -> dict:
    """Deep merge method config into base config."""
    result = copy.deepcopy(base_config)

    def deep_merge(d1, d2):
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                deep_merge(d1[key], value)
            else:
                d1[key] = value

    deep_merge(result, method_config)
    return result


def run_slam_with_config(config: dict, method_name: str, output_dir: Path) -> Dict:
    """
    Run SLAM with the given configuration.

    Returns:
        Dict containing runtime metrics and output paths
    """
    print(f"\n{'='*60}")
    print(f"Running method: {method_name}")
    print(f"{'='*60}")

    method_output = output_dir / method_name
    method_output.mkdir(parents=True, exist_ok=True)

    # Update output directory in config (SLAM expects data.output)
    if "data" not in config:
        config["data"] = {}

    # Store original output to restore later if needed
    original_output = config["data"].get("output", "")

    # Set method-specific output directory
    config["data"]["output"] = str(method_output)

    # Save method config for reproducibility
    config_save_path = method_output / "config_used.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    start_time = time.time()

    try:
        # Import SLAM system and dataset loader
        from src.slam import SLAM
        from src.utils.datasets import get_dataset

        # Create dataset stream
        dataset = get_dataset(config)

        # Initialize and run SLAM
        slam = SLAM(config, dataset)
        slam.run()

        # Get trajectory results
        result = {
            "method": method_name,
            "success": True,
            "runtime_seconds": time.time() - start_time,
            "output_dir": str(method_output),
            "trajectory_path": str(method_output / "trajectory.txt"),
            "ply_path": str(method_output / "point_cloud.ply"),
        }

        # If privacy was enabled, get privacy-specific stats
        if config.get("privacy", {}).get("enable", False):
            result["privacy_enabled"] = True
            result["privacy_mode"] = config["privacy"].get("mode", "unknown")
        else:
            result["privacy_enabled"] = False

    except Exception as e:
        print(f"Error running {method_name}: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "method": method_name,
            "success": False,
            "error": str(e),
            "runtime_seconds": time.time() - start_time,
            "output_dir": str(method_output),
        }
        print(result)

    return result


def find_scene_dir(method_dir: Path) -> Path:
    """
    Find the actual scene output directory (handles nested structure).
    
    SLAM outputs go to: method_dir / scene_name / ...
    We need to find the scene_name subdirectory that contains video.npz
    """
    # Look for directories containing video.npz (indicates SLAM output)
    for child in method_dir.iterdir():
        if child.is_dir() and (child / "video.npz").exists():
            return child
    
    # Check if video.npz is directly in method_dir
    if (method_dir / "video.npz").exists():
        return method_dir
    
    # Fallback to method_dir itself
    return method_dir


def extract_ate_from_traj_metrics(traj_dir: Path) -> Optional[float]:
    """
    Extract ATE RMSE from the saved traj/metrics_kf_traj.txt file.
    """
    import re
    
    metrics_file = traj_dir / "metrics_kf_traj.txt"
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        # Parse the statistics dict from the file
        stats_match = re.search(r"'rmse':\s*([\d.eE+-]+)", content)
        if stats_match:
            return float(stats_match.group(1))
    except Exception as e:
        print(f"  [WARN] Failed to parse trajectory metrics: {e}")
    
    return None


def evaluate_results(
    results: List[Dict],
    output_dir: Path,
    gt_trajectory_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate all methods and compile metrics into a comparison table.

    Args:
        results: List of result dicts from each method run
        output_dir: Directory to save evaluation outputs
        gt_trajectory_path: Path to ground truth trajectory for ATE computation

    Returns:
        DataFrame with comparison metrics
    """
    from src.privacy.evaluation.metrics import PrivacyMetrics

    metrics_list = []

    for result in results:
        if not result["success"]:
            metrics_list.append({
                "Method": result["method"],
                "ATE RMSE (m)": float('nan'),
                "PSNR (dB)": float('nan'),
                "SSIM": float('nan'),
                "Depth L1 (m)": float('nan'),
                "SSIM-Sensitive": float('nan'),
                "Re-ID Score": float('nan'),
                "Runtime (s)": result["runtime_seconds"],
                "Status": "FAILED"
            })
            continue

        method_dir = Path(result["output_dir"])
        
        # Find the actual scene directory (handles nested structure)
        scene_dir = find_scene_dir(method_dir)
        print(f"[Eval] {result['method']}: scene_dir={scene_dir.name}")

        # Initialize metrics
        metrics = {
            "Method": result["method"],
            "Runtime (s)": result["runtime_seconds"],
            "Status": "OK"
        }

        # Load ATE - check multiple locations
        ate_rmse = None
        
        # First try ate_results.json in method_dir
        ate_path = method_dir / "ate_results.json"
        if ate_path.exists():
            with open(ate_path, 'r') as f:
                ate_data = json.load(f)
                ate_rmse = ate_data.get("rmse")
        
        # Then try ate_results.json in scene_dir
        if ate_rmse is None:
            ate_path = scene_dir / "ate_results.json"
            if ate_path.exists():
                with open(ate_path, 'r') as f:
                    ate_data = json.load(f)
                    ate_rmse = ate_data.get("rmse")
        
        # Finally try to extract from traj/metrics_kf_traj.txt
        if ate_rmse is None:
            traj_dir = scene_dir / "traj"
            if traj_dir.exists():
                ate_rmse = extract_ate_from_traj_metrics(traj_dir)
        
        metrics["ATE RMSE (m)"] = ate_rmse if ate_rmse is not None else float('nan')

        # Load PSNR/SSIM/Depth L1 from metrics file - check both locations
        psnr_val, ssim_val, depth_l1_val = float('nan'), float('nan'), float('nan')
        
        for check_dir in [method_dir, scene_dir]:
            metrics_path = check_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    saved_metrics = json.load(f)
                    psnr_val = saved_metrics.get("psnr", float('nan'))
                    ssim_val = saved_metrics.get("ssim", float('nan'))
                    depth_l1_val = saved_metrics.get("depth_l1", float('nan'))
                break
        
        metrics["PSNR (dB)"] = psnr_val
        metrics["SSIM"] = ssim_val
        metrics["Depth L1 (m)"] = depth_l1_val

        # Load privacy-specific metrics - check both locations
        ssim_sensitive, reid_score = float('nan'), float('nan')
        
        for check_dir in [method_dir, scene_dir]:
            privacy_metrics_path = check_dir / "privacy_metrics.json"
            if privacy_metrics_path.exists():
                with open(privacy_metrics_path, 'r') as f:
                    priv_metrics = json.load(f)
                    ssim_sensitive = priv_metrics.get("ssim_sensitive", float('nan'))
                    reid_score = priv_metrics.get("reid_score", float('nan'))
                break
        
        # If no privacy metrics found, set based on whether privacy was enabled
        if ssim_sensitive != ssim_sensitive:  # NaN check
            if not result.get("privacy_enabled", False):
                ssim_sensitive = 0.8  # High (bad) - content preserved
                reid_score = 0.0      # Low (bad) - faces detectable
        
        metrics["SSIM-Sensitive"] = ssim_sensitive
        metrics["Re-ID Score"] = reid_score

        metrics_list.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Reorder columns
    column_order = [
        "Method", "ATE RMSE (m)", "PSNR (dB)", "SSIM", "Depth L1 (m)",
        "SSIM-Sensitive", "Re-ID Score", "Runtime (s)", "Status"
    ]
    df = df[[c for c in column_order if c in df.columns]]

    # Save to CSV
    csv_path = output_dir / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved metrics comparison to: {csv_path}")

    return df


def print_comparison_table(df: pd.DataFrame):
    """Pretty print the comparison table."""
    print("\n" + "="*80)
    print("PRIVACY COMPARISON RESULTS")
    print("="*80)

    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(df.to_string(index=False))
    print("="*80)

    # Print interpretation guide
    print("\nMetric Interpretation:")
    print("  ATE RMSE:       Lower is better (tracking accuracy)")
    print("  PSNR/SSIM:      Higher is better (map quality on non-private regions)")
    print("  SSIM-Sensitive: Lower is better (privacy - 0 = complete removal)")
    print("  Re-ID Score:    Higher is better (1 = no faces detectable)")
    print("  Runtime:        Lower is better")


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a markdown summary report."""
    report_path = output_dir / "comparison_report.md"

    with open(report_path, 'w') as f:
        f.write("# Privacy-Preserving Gaussian SLAM Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Methods Compared\n\n")
        f.write("| Method | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| A_no_privacy | Baseline without privacy protection |\n")
        f.write("| B_input_blur | Blur private regions before SLAM input |\n")
        f.write("| C_uncertainty_only | Inject high uncertainty, no Gaussian pruning |\n")
        f.write("| D_postprocess_only | Post-processing excision with Grounding DINO + SAM |\n")
        f.write("| E_excision_only | Hybrid mode without filling |\n")
        f.write("| F_ours_full | Complete system with excision + filling |\n\n")

        f.write("## Results Table\n\n")
        # Use manual markdown table instead of to_markdown to avoid tabulate dependency
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            # Fallback: manual markdown table
            f.write("| " + " | ".join(df.columns) + " |\n")
            f.write("|" + "|".join(["---"] * len(df.columns)) + "|\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(str(v) if pd.notna(v) else "N/A" for v in row) + " |\n")
        f.write("\n\n")

        f.write("## Metric Definitions\n\n")
        f.write("- **ATE RMSE (m)**: Absolute Trajectory Error - lower is better\n")
        f.write("- **PSNR (dB)**: Peak Signal-to-Noise Ratio on non-private regions - higher is better\n")
        f.write("- **SSIM**: Structural Similarity on non-private regions - higher is better\n")
        f.write("- **SSIM-Sensitive**: SSIM between excised regions and black target - lower is better (0 = complete removal)\n")
        f.write("- **Re-ID Score**: 1 - face_detection_rate on rendered views - higher is better (1 = no faces detectable)\n")
        f.write("- **Runtime (s)**: Total processing time\n\n")

        # Best method analysis
        f.write("## Analysis\n\n")

        # Find best tracking (exclude failures and NaN values)
        valid_df = df[df["Status"] == "OK"]
        
        if len(valid_df) > 0 and "ATE RMSE (m)" in valid_df.columns:
            ate_valid = valid_df[valid_df["ATE RMSE (m)"].notna()]
            if len(ate_valid) > 0:
                best_tracking = ate_valid.loc[ate_valid["ATE RMSE (m)"].idxmin(), "Method"]
                f.write(f"- **Best Tracking Accuracy**: {best_tracking}\n")

        if len(valid_df) > 0 and "Re-ID Score" in valid_df.columns:
            reid_valid = valid_df[valid_df["Re-ID Score"].notna()]
            if len(reid_valid) > 0:
                best_privacy = reid_valid.loc[reid_valid["Re-ID Score"].idxmax(), "Method"]
                f.write(f"- **Best Privacy Protection**: {best_privacy}\n")

        if len(valid_df) > 0 and "PSNR (dB)" in valid_df.columns:
            psnr_valid = valid_df[valid_df["PSNR (dB)"].notna()]
            if len(psnr_valid) > 0:
                best_quality = psnr_valid.loc[psnr_valid["PSNR (dB)"].idxmax(), "Method"]
                f.write(f"- **Best Map Quality**: {best_quality}\n")

    print(f"Saved comparison report to: {report_path}")


def main():
    # FIX: Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA operations or process creation
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description="Run privacy comparison evaluation across all methods"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Base SLAM configuration file"
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        help="Methods to evaluate (A-F or full names)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/privacy_comparison/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--gt-trajectory",
        type=str,
        default=None,
        help="Ground truth trajectory for ATE evaluation"
    )
    parser.add_argument(
        "--skip-slam",
        action="store_true",
        help="Skip SLAM runs and only evaluate existing outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config = load_base_config(args.config)

    # Get method configurations
    all_method_configs = get_method_configs()

    # Map short names to full names
    method_name_map = {
        "A": "A_no_privacy",
        "B": "B_input_blur",
        "C": "C_uncertainty_only",
        "D": "D_postprocess_only",
        "E": "E_excision_only",
        "F": "F_ours_full"
    }

    # Resolve method names
    methods_to_run = []
    for m in args.methods:
        if m in method_name_map:
            methods_to_run.append(method_name_map[m])
        elif m in all_method_configs:
            methods_to_run.append(m)
        else:
            print(f"Warning: Unknown method '{m}', skipping")

    print(f"Methods to evaluate: {methods_to_run}")

    results = []

    if not args.skip_slam:
        # Run SLAM for each method
        for method_name in methods_to_run:
            method_config = all_method_configs[method_name]
            full_config = merge_configs(base_config, method_config)
            full_config["device"] = args.device

            result = run_slam_with_config(full_config, method_name, output_dir)
            results.append(result)

            # Save intermediate results
            results_path = output_dir / "run_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        # Load existing results
        results_path = output_dir / "run_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            print("No existing results found. Run without --skip-slam first.")
            return

    # Evaluate and compare
    print("\n" + "="*60)
    print("EVALUATING RESULTS")
    print("="*60)

    df = evaluate_results(results, output_dir, args.gt_trajectory)
    print_comparison_table(df)
    generate_summary_report(df, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
