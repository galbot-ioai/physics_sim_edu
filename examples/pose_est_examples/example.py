# Usage example

import os
import time
import numpy as np
from pose_est import PoseEstimator

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    object_name = "extrusion"
    camera_matrix = [637.7254326533274, 637.7254326533274, 640.0, 360.0]
    rgb_path = os.path.join(script_dir, f"sim_test_data/images/{object_name}.jpg")
    depth_path = os.path.join(script_dir, f"sim_test_data/depth/{object_name}_depth.png")
    mask_path = os.path.join(script_dir, f"sim_test_data/mask/{object_name}_mask.png")
    cad_path = os.path.join(script_dir, f"models/{object_name}.obj")

    # Initialize pose estimator
    PE = PoseEstimator(
        camera_matrix=camera_matrix,
        depth_scale=0.001,
        model_scale_factor=None,
        visualize=True,  # Enable visualization
        output_dir=os.path.join(script_dir, "pose_estimation_results"),
    )

    # Estimate pose
    ts = time.perf_counter()
    pose = PE.estimate_pose(
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        cad_path=cad_path,
    )
    te = time.perf_counter()
    print(f"Pose estimation completed in {te - ts:.2f} seconds")

    if pose is not None:
        print("Final 6D pose matrix:")
        print(np.array_str(pose, precision=4, suppress_small=True))
