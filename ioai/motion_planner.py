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
# Description: Base class for motion planners
# Author: Chenyu Cao, Herman Ye@Galbot
#
######################################################################################

from abc import ABC, abstractmethod
import numpy as np
from ioai_env import IOAIEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Tuple, List, Optional
import copy
import math


#####################################################################################
# NOTE for Competition Participants:
#   The official baseline implementation provided below is intended as a starting point
#   and will not result in lower scores, but more sophisticated planners may
#   improve motion performance and reduce execution time.
#####################################################################################


class BaseMotionPlanner(ABC):
    """Abstract base class for motion planners in the IOAI environment.

    All motion planner implementations should inherit from this class and
    implement the plan_motion method.

    This base class is designed to be flexible and not impose strict constraints
    on input parameters or return values, allowing for various implementation approaches.
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the motion planner with a reference to the IOAI environment.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def plan_motion(self, *args, **kwargs):
        """Plan motion from current state to target state.

        This method should be implemented by subclasses to plan motion trajectories.
        The input parameters and return values are flexible to accommodate different
        implementation approaches.

        Args:
            *args: Variable length argument list for flexible input parameters.
            **kwargs: Arbitrary keyword arguments for flexible input parameters.

        Returns:
            Any: The planned motion trajectory in any format suitable for the implementation.
                Common formats include:
                - List[np.ndarray]: List of waypoints [x, y, z, qx, qy, qz, qw]
                - List[List[float]]: List of joint positions [joint1, joint2, ..., jointN]
                - Dict: Dictionary containing trajectory information
                - Any other format that suits the implementation

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement plan_motion().")


class InterpolationMotionPlanner(BaseMotionPlanner):
    """Official baseline motion planner for the IOAI environment.

    This class provides a basic implementation of motion planning using
    simple interpolation strategies for different motion types. It serves as a
    baseline for evaluation and can be used as a starting point for custom
    implementations.

    The planner uses linear interpolation for joint space motion and
    simple geometric transformations for Cartesian space motion. While functional,
    more advanced methods considering dynamics, obstacles, and optimization may
    improve motion performance and reduce execution time.

    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
        - Joint positions: [joint1, joint2, ..., jointN]
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the interpolation motion planner.

        Args:
            environment (IOAIEnv): The IOAI simulation environment instance.
        """
        super().__init__(environment)

    def plan_motion(
        self,
        start_joint_positions: List[float],
        end_joint_positions: List[float],
        interpolation_steps: int = 100,
    ) -> List[List[float]]:
        """Plan motion from start joint positions to end joint positions using interpolation.

        This method takes the start and end joint positions, along with the number of
        interpolation steps, to generate a motion trajectory.

        Args:
            start_joint_positions (List[float]): Starting joint positions [joint1, joint2, ..., jointN]
            end_joint_positions (List[float]): Target joint positions [joint1, joint2, ..., jointN]
            interpolation_steps (int): Number of interpolation steps (default: 100)

        Returns:
            List[List[float]]: List of joint positions representing the trajectory.
                Each element is a list of joint positions for one time step.

        Raises:
            ValueError: If joint positions have invalid format or different lengths.
            RuntimeError: If motion planning fails.
        """
        try:
            # Validate input parameters
            self._validate_joint_positions(
                start_joint_positions, "start_joint_positions"
            )
            self._validate_joint_positions(end_joint_positions, "end_joint_positions")

            if len(start_joint_positions) != len(end_joint_positions):
                raise ValueError(
                    "start_joint_positions and end_joint_positions must have the same length"
                )

            if interpolation_steps <= 0:
                raise ValueError("interpolation_steps must be positive")

            # Convert to numpy arrays for easier manipulation
            start_joints = np.array(start_joint_positions)
            end_joints = np.array(end_joint_positions)

            # Generate trajectory using linear interpolation
            trajectory = self._interpolate_joint_trajectory_simple(
                start_joints, end_joints, interpolation_steps
            )

            return trajectory

        except Exception as e:
            raise RuntimeError(f"Failed to plan motion: {str(e)}")

    def _validate_joint_positions(self, joint_positions: List[float], param_name: str):
        """Validate joint positions format."""
        if not isinstance(joint_positions, (list, np.ndarray)):
            raise ValueError(f"{param_name} must be a list or numpy array")

        if len(joint_positions) == 0:
            raise ValueError(f"{param_name} cannot be empty")

        # Check if all elements are numbers
        for i, pos in enumerate(joint_positions):
            if not isinstance(pos, (int, float, np.number)):
                raise ValueError(f"{param_name}[{i}] must be a number, got {type(pos)}")

    def _interpolate_joint_trajectory_simple(
        self, start_joints: np.ndarray, end_joints: np.ndarray, steps: int
    ) -> List[List[float]]:
        """Generate interpolated joint trajectory using linear interpolation."""
        return np.linspace(start_joints, end_joints, steps).tolist()
