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
# Description: Base class for object pose estimators
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
    ) -> Tuple[np.ndarray, np.ndarray] | None:
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

from yolo_seg.seg import YoloSeg
from pose_est.pose_est import PoseEstimator

class YoloSegObjectPoseEstimator(BaseObjectPoseEstimator):
    def __init__(self, environment: IOAIEnv, yolo_seg_model_path: str):
        super().__init__(environment)
        self.yolo_seg = YoloSeg(model_path=yolo_seg_model_path)
        self.camera_matrix = [638.315, 637.683, 636.496, 363.410]
        self.depth_scale = 0.001
        self.model_scale_factor = None
        self.pose_est = PoseEstimator(
            camera_matrix=self.camera_matrix,
            depth_scale=self.depth_scale,
            model_scale_factor=self.model_scale_factor,
            visualize=False,
            log_debug=True,
        )

    def estimate_pose(self, object_name: str, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray] | None:
        """Estimate the pose of an object using YOLO segmentation and pose estimation.
        
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
        # Get RGB and depth images from front camera
        rgb = self.environment.interface.front_head_camera.get_rgb()
        depth = self.environment.interface.front_head_camera.get_depth()

        # Preprocess depth
        depth = preprocess_depth(
            depth,
            scale=1000,
            min_value=0.0,
            max_value=5 * 1000,
            data_type=np.uint16,
        )

        # Convert RGB to BGR and save images to temporary files
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Use temporary files for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as rgb_file:
            rgb_path = rgb_file.name
            cv2.imwrite(rgb_path, bgr)
            
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as depth_file:
            depth_path = depth_file.name
            cv2.imwrite(depth_path, depth)
        
        try:
            # Perform YOLO segmentation
            seg_results = self.yolo_seg.segment_image(rgb_path)

            for result in seg_results:
                if result.masks is not None:
                    print(f"Detected {len(result.masks)} masks in the image.")
                else:
                    print("No masks detected.")
            
            # Get the best mask for the target object
            mask = self.yolo_seg.get_best_mask(seg_results, object_name)

            if mask is None:
                print(f"No mask found for {object_name}")
                return None

            # Resize mask to match image dimensions
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Save mask to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_file:
                mask_path = mask_file.name
                cv2.imwrite(mask_path, mask_binary)

            # Estimate pose using the pose estimator
            pose_matrix = self.pose_est.estimate_pose(
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
                cad_name=object_name,
            )

            # Convert pose matrix to position and orientation
            if pose_matrix is not None:
                position = pose_matrix[:3, 3]
                rotation_matrix = pose_matrix[:3, :3]
                quat = R.from_matrix(rotation_matrix).as_quat()  # [qx, qy, qz, qw]
                pose = np.concatenate([position, quat])  # [x, y, z, qx, qy, qz, qw]
            else:
                print(f"Pose estimation failed for {object_name}")
                return None

            # Transform pose from camera frame to robot frame
            position_wrt_robot, orientation_wrt_robot = self.environment.camera_to_robot_frame(
                pose[:3], pose[3:]
            )
            
            return position_wrt_robot, orientation_wrt_robot
            
        finally:
            # Clean up temporary files
            for file_path in [rgb_path, depth_path, mask_path]:
                try:
                    os.unlink(file_path)
                except (OSError, NameError):
                    pass  # File might not exist or variable not defined