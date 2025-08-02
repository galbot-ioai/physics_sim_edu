import os
import argparse
import time
import numpy as np
from pose_est import PoseEstimator


def main(args):
    # Get parent directory (ioai/pose_est/) for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Set simulation camera matrix
    camera_matrix = [359.0587537547767, 359.0587537547767, 640.0, 360.0]  # fx, fy, cx, cy
    
    # Use provided paths directly
    rgb_path = args.rgb_path
    depth_path = args.depth_path
    mask_path = args.mask_path
    object_name = args.object_name

    # Initialize pose estimator
    PE = PoseEstimator(
        camera_matrix=camera_matrix,
        depth_scale=args.depth_scale,
        model_scale_factor=args.model_scale_factor,
        visualize=args.visualize,
    )

    # Estimate pose
    ts = time.perf_counter()
    pose = PE.estimate_pose(
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        cad_name=object_name,
    )
    te = time.perf_counter()
    print(f"Pose estimation completed in {te - ts:.2f} seconds")

    if pose is not None:
        print("Final 6D pose matrix:")
        print(np.array_str(pose, precision=4, suppress_small=True))


if __name__ == "__main__":
    # Get parent directory for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    parser = argparse.ArgumentParser(description="Pose estimation example")
    parser.add_argument('--object_name', type=str, default='mug', 
                       help='Object name for pose estimation')
    parser.add_argument('--rgb_path', type=str, 
                       default=os.path.join(parent_dir, 'test_data/sim_data/mug/images/00_color_image.jpg'), 
                       help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, 
                       default=os.path.join(parent_dir, 'test_data/sim_data/mug/depth/00.png'), 
                       help='Path to depth image')
    parser.add_argument('--mask_path', type=str, 
                       default=os.path.join(parent_dir, 'test_data/sim_data/mug/mask/00.png'), 
                       help='Path to mask image')
    parser.add_argument('--depth_scale', type=float, default=0.001, 
                       help='Depth scale factor')
    parser.add_argument('--model_scale_factor', type=float, default=None, 
                       help='Model scale factor')
    parser.add_argument('--visualize', action='store_true', default=True, 
                       help='Enable visualization')
    args = parser.parse_args()
    
    main(args)
