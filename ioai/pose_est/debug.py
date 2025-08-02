# Debug

import time
import numpy as np
from pose_est import PoseEstimator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation Debug Script")
    parser.add_argument(
        "--debug_file_dir",
        type=str,
        required=True,
        help="Directory containing debug files (default: pose_est_dbg_20250730181921)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of results (default: False)",
    )
    args = parser.parse_args()

    debug_file_dir = args.debug_file_dir
    is_visualize = args.visualize

    params_est = np.load(f"{debug_file_dir}/params_est.npz")
    params_init = np.load(f"{debug_file_dir}/params_init.npz")

    # Debug: Print the types and values of loaded parameters
    print("=== Debug Information ===")
    print(
        f"rgb_path type: {type(params_est['rgb_path'])}, value: {params_est['rgb_path']}"
    )
    print(
        f"depth_path type: {type(params_est['depth_path'])}, value: {params_est['depth_path']}"
    )
    print(
        f"mask_path type: {type(params_est['mask_path'])}, value: {params_est['mask_path']}"
    )
    print(
        f"cad_name type: {type(params_est['cad_name'])}, value: {params_est['cad_name']}"
    )
    print("========================\n")

    # Convert numpy arrays/bytes to strings if necessary
    def ensure_string(param):
        """Convert parameter to string if it's not already"""
        if isinstance(param, np.ndarray):
            # If it's a numpy array, get the item (which might be bytes or string)
            param = param.item()

        if isinstance(param, bytes):
            # If it's bytes, decode to string
            param = param.decode("utf-8")

        return str(param)  # Ensure it's a string

    def process_parameter(param):
        """Process parameter and handle special cases like None"""
        if isinstance(param, np.ndarray):
            param = param.item()

        if isinstance(param, bytes):
            param = param.decode("utf-8")

        # Handle special case where 'None' string should be None
        if isinstance(param, str) and param.lower() == "none":
            return None

        return param

    def process_numeric_parameter(param):
        """Process numeric parameter and handle special cases"""
        if isinstance(param, np.ndarray):
            param = param.item()

        if isinstance(param, bytes):
            param = param.decode("utf-8")

        # Handle special case where 'None' string should be None
        if isinstance(param, str) and param.lower() == "none":
            return None

        # Convert to float if it's a string representation of a number
        if isinstance(param, str):
            try:
                return float(param)
            except ValueError:
                return param

        return param

    # Convert all path parameters to proper strings
    rgb_path = args.debug_file_dir + "/" + "rgb_image.png"
    depth_path = args.debug_file_dir + "/" + "depth_image.png"
    mask_path = args.debug_file_dir + "/" + "mask_image.png"
    cad_name = ensure_string(params_est["cad_name"])

    # Process init parameters (handle None values properly)
    camera_matrix = params_init["camera_matrix"]
    depth_scale = process_numeric_parameter(params_init["depth_scale"])
    model_scale_factor = process_parameter(params_init["model_scale_factor"])

    # Ensure depth_scale is a float
    if depth_scale is None or not isinstance(depth_scale, (int, float)):
        print(f"Warning: Invalid depth_scale value {depth_scale}, using default 0.001")
        depth_scale = 0.001
    else:
        depth_scale = float(depth_scale)

    print("=== Converted Parameters ===")
    print(f"rgb_path: {rgb_path}")
    print(f"depth_path: {depth_path}")
    print(f"mask_path: {mask_path}")
    print(f"cad_name: {cad_name}")
    print(f"camera_matrix: {camera_matrix}")
    print(f"depth_scale: {depth_scale} (type: {type(depth_scale)})")
    print(
        f"model_scale_factor: {model_scale_factor} (type: {type(model_scale_factor)})"
    )
    print("============================\n")

    # Initialize pose estimator
    PE = PoseEstimator(
        camera_matrix=camera_matrix,
        depth_scale=depth_scale,
        model_scale_factor=model_scale_factor,
        visualize=is_visualize,
        log_debug=False,  # Enable debug logging
    )

    # Estimate pose
    ts = time.perf_counter()
    pose = PE.estimate_pose(
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        cad_name=cad_name,
    )
    te = time.perf_counter()
    print(f"Pose estimation completed in {te - ts:.2f} seconds")

    if pose is not None:
        print("Final 6D pose matrix:")
        print(np.array_str(pose, precision=4, suppress_small=True))
