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
    def __init__(self, environment: IOAIEnv):
        super().__init__(environment)

    def estimate_pose(self, object_name: str, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass