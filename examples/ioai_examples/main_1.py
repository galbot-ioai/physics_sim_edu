# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
#####################################################################################
#          _____             _   _       _   _
#         / ____|           | | | |     | \ | |
#        | (___  _   _ _ __ | |_| |__   |  \| | _____   ____ _
#         \___ \| | | | '_ \| __| '_ \  | . ` |/ _ \ \ / / _` |
#         ____) | |_| | | | | |_| | | | | |\  | (_) \ V / (_| |
#        |_____/ \__, |_| |_|\__|_| |_| |_| \_|\___/ \_/ \__,_|
#                 __/ |
#                |___/
#
#####################################################################################
#
# Description: A basic pipeline for IOAI environment
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-07-28
#
#####################################################################################

from abc import ABC, abstractmethod
import numpy as np
from ioai_env import IOAIEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Tuple, List
import copy


#####################################################################################
# NOTE for Competition Participants:
#   You will receive a higher score if you use vision-based methods (e.g., camera
#   images, depth maps, point clouds) to estimate object poses. Using ground-truth
#   information directly from the simulator is only intended as a baseline and is
#   not permitted for final competition submissions. Please implement your pose
#   estimation using perception and vision techniques whenever possible!
#####################################################################################


class BaseObjectPoseEstimator(ABC):
    """Abstract base class for object pose estimators in the IOAI environment.

    All pose estimator implementations should inherit from this class and
    implement the estimate_pose method.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the pose estimator with a reference to the IOAI environment.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def estimate_pose(self, object_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the 6D pose (position and orientation) of a given object.

        Args:
            object_name (str): The name of the object whose pose is to be estimated.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the position (3,) and
                orientation (4,) as numpy arrays, both in the robot's coordinate frame.
                The position is ordered as [x, y, z], and the orientation quaternion is
                ordered as [qx, qy, qz, qw].

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

    def estimate_pose(self, object_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ground-truth pose of the specified object.

        This method queries the simulator for the true position and orientation
        of the object, and transforms them into the robot's coordinate frame.

        Args:
            object_name (str): The name of the object whose pose is to be estimated.

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


#####################################################################################
# NOTE for Competition Participants:
#   The official baseline implementation provided below is intended as a starting point
#   and will not result in lower scores, but more sophisticated predictors may
#   improve performance and reduce execution time.
#####################################################################################


class BaseGraspPosePredictor(ABC):
    """Abstract base class for grasp pose predictors in the IOAI environment.

    All grasp pose predictor implementations should inherit from this class and
    implement the predict_grasp method.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the grasp pose predictor with a reference to the IOAI environment.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def predict_grasp(
        self, object_name: str, object_pose: np.ndarray
    ) -> Dict[str, Any]:
        """Predict the optimal grasp pose for a given object.

        Args:
            object_name (str): The name of the object to grasp.
            object_pose (np.ndarray): The 6D pose of the object in quaternion format [x, y, z, qx, qy, qz, qw].

        Returns:
            Dict[str, Any]: A dictionary containing grasp information with the following keys:
                - "grasp_pose": np.ndarray of shape (7,) containing [x, y, z, qx, qy, qz, qw]
                - "gripper_width": float indicating the required gripper width
                - "part_id": str indicating the object type
                - "object_pose": np.ndarray containing the input object pose
                - "grasp_se3": np.ndarray of shape (4, 4) containing the SE(3) transformation matrix

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
        from grasp_reg import GraspRegistration

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


def example():
    """Example usage of pose and grasp pose estimators."""
    # Initialize IOAI simulation environment
    env = IOAIEnv(headless=False)

    # Initialize object pose estimator (you can use GroundTruthObjectPoseEstimator for baseline)
    # For competition, implement your own object pose estimator using vision-based methods
    object_pose_estimator = GroundTruthObjectPoseEstimator(env)

    # Initialize grasp pose predictor (you can use OfficialGraspPosePredictor for baseline)
    # For better performance, implement your own custom grasp pose predictor
    grasp_pose_predictor = OfficialGraspPosePredictor(env)

    # Estimate object pose
    object_pose = object_pose_estimator.estimate_pose("cube")
    print("Object pose:", object_pose)

    # Predict grasp pose
    object_pose_combined = np.concatenate([object_pose[0], object_pose[1]])
    grasp_pose = grasp_pose_predictor.predict_grasp(
        object_name="cube", object_pose=object_pose_combined
    )
    print("Grasp pose:", grasp_pose)

if __name__ == "__main__":
    pass