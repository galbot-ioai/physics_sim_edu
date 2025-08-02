# Usage example

import time
import numpy as np
from pose_est import PoseEstimator

if __name__ == "__main__":
    data_type = "real"  # Change to "sim" for simulation data

    if data_type == "real":
        object_name = "cube"  # Change to your real object name
        camera_matrix = [638.315, 637.683, 636.496, 363.410]  # fx, fy, cx, cy
        rgb_path = "test_data/real/rgb_image.png"
        depth_path = "test_data/real/depth_image.png"
        mask_path = "test_data/real/mask_image.png"
    else:  # Simulation data
        object_name = "mug"  # Change to your object name
        camera_matrix = [
            359.0587537547767,
            359.0587537547767,
            640.0,
            360.0,
        ]  # fx, fy, cx, cy
        rgb_path = f"test_data/sim_data/{object_name}/images/00_color_image.jpg"
        depth_path = f"test_data/sim_data/{object_name}/depth/00.png"
        mask_path = f"test_data/sim_data/{object_name}/mask/00.png"

    # Initialize pose estimator
    PE = PoseEstimator(
        camera_matrix=camera_matrix,
        depth_scale=0.001,
        model_scale_factor=None,
        visualize=True,  # Enable visualization
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
