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
# Description: A basic pipeline for IOAI environment
# Author: Chenyu Cao, Herman Ye@Galbot
#
######################################################################################

from abc import ABC, abstractmethod
import numpy as np
from ioai_env import IOAIEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Tuple, List
import copy


#####################################################################################
# NOTE for Competition Participants:
#   The official baseline implementation provided below is intended as a starting point
#   and will not result in lower scores, but more sophisticated predictors may
#   improve grasping performance and reduce execution time.
#####################################################################################


class BaseGraspPosePredictor(ABC):
    """Abstract base class for grasp pose predictors in the IOAI environment.

    All grasp pose predictor implementations should inherit from this class and
    implement the predict_grasp method.

    This base class is designed to be flexible and not impose strict constraints
    on input parameters or return values, allowing for various implementation approaches.
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the grasp pose predictor with a reference to the IOAI environment.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def predict_grasp(self, *args, **kwargs):
        """Predict the optimal grasp pose for an object.

        This method should be implemented by subclasses to predict grasp poses.
        The input parameters and return values are flexible to accommodate different
        implementation approaches.

        Args:
            *args: Variable length argument list for flexible input parameters.
            **kwargs: Arbitrary keyword arguments for flexible input parameters.

        Returns:
            Any: The predicted grasp pose in any format suitable for the implementation.
                Common formats include:
                - np.ndarray: Grasp pose [x, y, z, qx, qy, qz, qw]
                - Tuple[np.ndarray, np.ndarray]: (position, orientation)
                - Dict: Dictionary containing grasp information
                - Any other format that suits the implementation

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement predict_grasp().")


class OfficialGraspPosePredictor(BaseGraspPosePredictor):
    """Official baseline grasp pose predictor for the IOAI environment.

    This class provides a basic implementation of grasp pose prediction using
    predefined grasp strategies for different object types. It serves as a
    baseline for evaluation and can be used as a starting point for custom
    implementations.

    The predictor uses simple geometric transformations to predict grasp poses
    based on object type and pose. While functional, more advanced methods
    considering object geometry, physics, and task requirements may provide
    better performance.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the official grasp pose predictor.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        super().__init__(environment)
        from grasp_reg.grasp_reg import GraspRegistration

        self.grasp_registration = GraspRegistration()

    def predict_grasp(self, object_name: str, object_pose: np.ndarray) -> np.ndarray:
        """Predict the optimal grasp pose for a given object using the official baseline.

        This method uses the GraspRegistration class to predict grasp poses based on
        predefined strategies for different object types. The grasp pose is returned
        in the robot's coordinate frame.

        Args:
            object_name (str): The name of the object to grasp (e.g., "cube", "power_drill").
            object_pose (np.ndarray): The 6D pose of the object in quaternion format [x, y, z, qx, qy, qz, qw].

        Returns:
            np.ndarray: Grasp pose of shape (7,) containing [x, y, z, qx, qy, qz, qw]

        Raises:
            ValueError: If the object_name is not supported or object_pose has invalid shape.
            RuntimeError: If grasp pose generation fails.
        """
        if not isinstance(object_pose, np.ndarray) or object_pose.shape != (7,):
            raise ValueError("object_pose must be a numpy array of shape (7,).")

        try:
            # Use the GraspRegistration class to predict grasp pose
            grasp_result = self.grasp_registration.predict_grasp(
                object_name, object_pose
            )

            # Extract the grasp pose from the result
            # The result contains additional information, but for this baseline example,
            # we only use the grasp pose
            grasp_pose = grasp_result["grasp_pose"]
            return grasp_pose

        except Exception as e:
            raise RuntimeError(
                f"Failed to predict grasp pose for object '{object_name}': {str(e)}"
            )
