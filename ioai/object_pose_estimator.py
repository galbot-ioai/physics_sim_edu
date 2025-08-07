######################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
######################################################################################
#
#  ██████  ██    ██ ██    ██ ████████ ██     ██ ██    ██  ███████  ██     ██    ███
# ██    ██  ██  ██  ███   ██    ██    ██     ██ ███   ██ ██     ██ ██     ██   ██ ██
# ██         ████   ████  ██    ██    ██     ██ ████  ██ ██     ██ ██     ██  ██   ██
#  ██████     ██    ██ ██ ██    ██    █████████ ██ ██ ██ ██     ██ ██     ██ ██     ██
#       ██    ██    ██  ████    ██    ██     ██ ██  ████ ██     ██  ██   ██  █████████
# ██    ██    ██    ██   ███    ██    ██     ██ ██   ███ ██     ██   ██ ██   ██     ██
#  ██████     ██    ██    ██    ██    ██     ██ ██    ██  ███████     ███    ██     ██
#
######################################################################################
#
# Description: Object pose estimators for the IOAI environment
# Author: Chenyu Cao, Herman Ye@Galbot
#
######################################################################################

from abc import ABC, abstractmethod
import numpy as np
from ioai_env import IOAIEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Tuple, List
import copy
import os
from pathlib import Path
import cv2
import tempfile
from physics_simulator.utils import preprocess_depth


#####################################################################################
# NOTE for Competition Participants:
#   You will receive a higher score if you use vision-based methods (e.g., camera
#   images, depth maps, point clouds) to estimate object poses. Using ground-truth
#   information directly from the simulator is allowed but will result in lower scores.
#   For better performance, please implement your pose estimation using perception
#   and vision techniques whenever possible!
#####################################################################################


class BaseObjectPoseEstimator(ABC):
    """Abstract base class for object pose estimators in the IOAI environment.

    All pose estimator implementations should inherit from this class and
    implement the estimate_pose method.

    This base class is designed to be flexible and not impose strict constraints
    on input parameters or return values, allowing for various implementation approaches.
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the pose estimator with a reference to the IOAI environment.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def estimate_pose(self, *args, **kwargs):
        """Estimate the pose of an object.

        This method should be implemented by subclasses to estimate object poses.
        The input parameters and return values are flexible to accommodate different
        implementation approaches.

        Args:
            *args: Variable length argument list for flexible input parameters.
            **kwargs: Arbitrary keyword arguments for flexible input parameters.

        Returns:
            Any: The estimated pose in any format suitable for the implementation.
                Common formats include:
                - Tuple[np.ndarray, np.ndarray]: (position, orientation)
                - np.ndarray: Combined pose [x, y, z, qx, qy, qz, qw]
                - Dict: Dictionary containing pose information
                - Any other format that suits the implementation

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement estimate_pose().")


class GroundTruthObjectPoseEstimator(BaseObjectPoseEstimator):
    """A pose estimator that returns the ground-truth object pose from the simulator.

    This class is intended as a baseline for evaluation. It directly queries the
    simulator for the true pose of the object, which is not possible in a real-world
    or vision-based setting.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
    """

    # Mapping from object names to their corresponding prim paths in the simulation.
    OBJECT_PRIM_PATHS = {
        "cube": "/World/Cube",
        "power_drill": "/World/PowerDrill",
        "extrusion": "/World/Extrusion",
        "cone": "/World/Cone",
        "bin": "/World/Bin",
        "toy": "/World/Toy",
    }

    def __init__(self, environment: IOAIEnv):
        """Initialize the ground-truth pose estimator.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        super().__init__(environment)

    def estimate_pose(
        self, object_name: str, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ground-truth pose of the specified object.

        This method queries the simulator for the true position and orientation
        of the object, and transforms them into the robot's coordinate frame.

        Args:
            object_name (str): The name of the object whose pose is to be estimated.
            *args: Additional arguments (ignored in this implementation).
            **kwargs: Additional keyword arguments (ignored in this implementation).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The position (3,) and orientation (4,)
                of the object in the robot's coordinate frame.
                The position is ordered as [x, y, z], and the orientation quaternion is
                ordered as [qx, qy, qz, qw].

        Raises:
            KeyError: If the object_name is not found in the prim path dictionary.
            RuntimeError: If the simulator fails to return the object state.
        """
        if object_name not in self.OBJECT_PRIM_PATHS:
            raise KeyError(f"Object name '{object_name}' is not recognized.")

        # Retrieve the prim path for the specified object.
        object_prim_path = self.OBJECT_PRIM_PATHS[object_name]

        # Query the simulator for the object's ground-truth state.
        object_state_wrt_world = self.environment.simulator.get_object_state(
            object_prim_path
        )
        if (
            object_state_wrt_world is None
            or "position" not in object_state_wrt_world
            or "orientation" not in object_state_wrt_world
        ):
            raise RuntimeError(f"Failed to retrieve state for object '{object_name}'.")

        # Transform the pose from world frame to robot frame.
        # The world_to_robot_frame function returns a tuple: (position, orientation)
        position_wrt_robot, orientation_wrt_robot = (
            self.environment.world_to_robot_frame(
                object_state_wrt_world["position"],
                object_state_wrt_world["orientation"],
            )
        )

        # Return the position and orientation in robot frame.
        # Position: [x, y, z], Orientation (quaternion): [qx, qy, qz, qw]
        return position_wrt_robot, orientation_wrt_robot


class YoloSegObjectPoseEstimator(BaseObjectPoseEstimator):
    """A pose estimator that uses YOLO segmentation and pose estimation for object pose detection.

    This class combines YOLO segmentation to detect objects in images and pose estimation
    to determine their 3D pose. It processes RGB and depth images from the front head camera
    to estimate object poses in the robot's coordinate frame.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
    """

    def __init__(self, environment: IOAIEnv, yolo_seg_model_path: str):
        """Initialize the YOLO segmentation pose estimator.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
            yolo_seg_model_path (str): Path to the YOLO segmentation model file.
        """
        from yolo_seg.seg import YoloSeg
        from pose_est.pose_est import PoseEstimator

        super().__init__(environment)

        self.yolo_seg = YoloSeg(model_path=yolo_seg_model_path)
        # Camera intrinsic parameters: [fx, fy, cx, cy]
        self.camera_matrix = [638.315, 637.683, 636.496, 363.410]
        # Depth scale factor (meters per unit in depth image)
        self.depth_scale = 0.001
        # Model scale factor for pose estimation (None for default)
        self.model_scale_factor = None
        self.pose_est = PoseEstimator(
            camera_matrix=self.camera_matrix,
            depth_scale=self.depth_scale,
            model_scale_factor=self.model_scale_factor,
            visualize=False,
            log_debug=True,
        )

    def estimate_pose(
        self, object_name: str, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """Estimate the pose of an object using YOLO segmentation and pose estimation.

        This method captures RGB and depth images from the front head camera, performs
        YOLO segmentation to detect the target object, and uses pose estimation to
        determine the object's 3D pose in the robot's coordinate frame.

        Args:
            object_name (str): The name of the object whose pose is to be estimated.
            *args: Additional arguments (ignored in this implementation).
            **kwargs: Additional keyword arguments (ignored in this implementation).

        Returns:
            Tuple[np.ndarray, np.ndarray] | None: The position (3,) and orientation (4,)
                of the object in the robot's coordinate frame, or None if pose estimation fails.
                The position is ordered as [x, y, z], and the orientation quaternion is
                ordered as [qx, qy, qz, qw].
        """
        # Get RGB and depth images from front head camera
        rgb = self.environment.interface.front_head_camera.get_rgb()
        depth = self.environment.interface.front_head_camera.get_depth()

        # Preprocess depth image: scale to millimeters, clip values, convert to uint16
        depth = preprocess_depth(
            depth,
            scale=1000,
            min_value=0.0,
            max_value=5 * 1000,
            data_type=np.uint16,
        )

        # Convert RGB to BGR format for OpenCV processing
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Create temporary files for image processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as rgb_file:
            rgb_path = rgb_file.name
            cv2.imwrite(rgb_path, bgr)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as depth_file:
            depth_path = depth_file.name
            cv2.imwrite(depth_path, depth)

        # Initialize mask_path to None to handle cases where mask creation fails
        mask_path = None
        
        try:
            # Perform YOLO segmentation on the RGB image
            seg_results = self.yolo_seg.segment_image(rgb_path)

            # Debug: Print number of detected masks
            for result in seg_results:
                if result.masks is not None:
                    print(f"Detected {len(result.masks)} masks in the image.")
                else:
                    print("No masks detected.")

            # Extract the best segmentation mask for the target object
            mask = self.yolo_seg.get_best_mask(seg_results, object_name)

            if mask is None:
                print(f"No mask found for {object_name}")
                return None

            # Resize mask to match original image dimensions and create binary mask
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Save binary mask to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as mask_file:
                mask_path = mask_file.name
                cv2.imwrite(mask_path, mask_binary)

            # Estimate pose using the pose estimator with RGB, depth, and mask
            pose_matrix = self.pose_est.estimate_pose(
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
                cad_name=object_name,
            )

            # Convert 4x4 pose matrix to position and quaternion orientation
            if pose_matrix is not None:
                # Extract position from translation part of transformation matrix
                position = pose_matrix[:3, 3]
                # Extract rotation matrix and convert to quaternion
                rotation_matrix = pose_matrix[:3, :3]
                quat = R.from_matrix(rotation_matrix).as_quat()  # [qx, qy, qz, qw]
                # Combine position and orientation into single pose vector
                pose = np.concatenate([position, quat])  # [x, y, z, qx, qy, qz, qw]
            else:
                print(f"Pose estimation failed for {object_name}")
                return None

            # Transform pose from camera coordinate frame to robot coordinate frame
            position_wrt_robot, orientation_wrt_robot = (
                self.environment.camera_to_robot_frame(pose[:3], pose[3:])
            )

            return position_wrt_robot, orientation_wrt_robot

        finally:
            # Clean up temporary files to avoid disk space issues
            files_to_cleanup = [rgb_path, depth_path]
            if mask_path is not None:
                files_to_cleanup.append(mask_path)
                
            for file_path in files_to_cleanup:
                try:
                    os.unlink(file_path)
                except (OSError, NameError):
                    pass  # File might not exist or variable not defined
