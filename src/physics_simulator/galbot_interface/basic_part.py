#####################################################################################
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
# Description: Basic components for Galbot interface (camera, limb, gripper)
# Author: Chenyu Cao@Galbot
# Date: 2025-05-01
#
#####################################################################################

from typing import List
from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface.config import GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
import numpy as np


class BasicCamera:
    def __init__(
        self, galbot_interface_config: GalbotInterfaceConfig, simulator: PhysicsSimulator
    ):
        self.module_name = None
        self.module_config = None
        self.initialized = False
        self.simulator = simulator
        self.logger = simulator.logger
        self.galbot_interface_config = galbot_interface_config
        self.rgb_camera = None
        self.depth_camera = None

    def initialize(self):
        self.robot = self.simulator.get_robot(
            self.galbot_interface_config.robot.prim_path
        )
        
        if self.module_name is None:
            raise RuntimeError(
                "module_name is not set, please set it in the __init__ method"
            )
        self.module_config = getattr(self.galbot_interface_config, self.module_name)
        
        # Get the camera by the prim path
        if self.module_config.prim_path_rgb is not None:
            self.rgb_camera = self.simulator.get_sensor(self.module_config.prim_path_rgb)
        else:
            self.rgb_camera = None
            
        if self.module_config.prim_path_depth is not None:
            self.depth_camera = self.simulator.get_sensor(self.module_config.prim_path_depth)
        else:
            self.depth_camera = None

        if self.rgb_camera is None and self.depth_camera is None:
            raise RuntimeError(
                f"At least one camera (RGB or depth) must be configured for {self.module_name}"
            )

        self.initialized = True
        self.logger.log_debug(f"{self.__class__.__name__} initialized")

    def get_rgb(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return self.rgb_camera.get_data()

    def get_depth(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return self.depth_camera.get_data()

    def get_segmentation(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return self.rgb_camera.get_segmentation()

    def get_point_cloud_wrt_robot(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            if self.depth_camera.get_data() is None:
                self.logger.log_warning("Depth image is None, cannot get point cloud")
                return None
            return self.depth_camera.get_point_cloud_wrt_robot(self.robot)

    def get_pose_wrt_robot(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            # Default to rgb camera
            return self.rgb_camera.get_pose_wrt_robot(self.robot)

    def get_parameters(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return {
                "rgb": self.rgb_camera.get_parameters(),
                "depth": self.depth_camera.get_parameters(),
            }


class BasicLimb:
    def __init__(
        self, galbot_interface_config: GalbotInterfaceConfig, simulator: PhysicsSimulator
    ):
        self.module_name = None
        self.module_config = None
        self.initialized = False
        self.simulator = simulator
        self.logger = simulator.logger
        self.galbot_interface_config = galbot_interface_config
        self.joint_names = []
        self.joint_indexes = []
        self.robot = None
        self.robot_controller = None
        self.future_actions = []
        self.current_trajectory = None
        self.trajectory_start_time = 0.0
        self.motion_started = False

    def initialize(self):
        self.robot = self.simulator.get_robot(
            self.galbot_interface_config.robot.prim_path
        )
        
        self.module_config = getattr(self.galbot_interface_config, self.module_name)
        self.joint_names = self.module_config.joint_names
        self.robot_controller = self.robot.get_articulation_controller()
        self.joint_indexes = [self.robot.get_dof_index(x) for x in self.joint_names]
        self.initialized = True
        self.logger.log_debug(f"{self.__class__.__name__} initialized")

    def get_joint_positions(self):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return self.simulator.get_joint_positions(
                robot=self.robot, joint_names=self.joint_names
            )

    def set_joint_positions(self, joint_positions: List[float], immediate=False):
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            self.simulator.set_joint_positions(
                robot=self.robot,
                joint_names=self.joint_names,
                joint_positions=joint_positions,
                immediate=immediate,
            )

    def _create_actions_mujoco(self, joint_trajectory: JointTrajectory):
        """
        Create a list of actions from the joint trajectory.
        
        Args:
            joint_trajectory: JointTrajectory object with positions
            
        Returns:
            List of actions for the robot controller
        """
        actions = []
        # JointTrajectory positions have shape (n_points, n_joints)
        if joint_trajectory.positions is None:
            raise ValueError("Joint trajectory must have positions")
            
        for i in range(joint_trajectory.positions.shape[0]):
            # Extract the joint positions for this waypoint
            position = joint_trajectory.positions[i].tolist()
            action = {
                "joint_positions": position,
                "joint_indices": self.joint_indexes,
            }
            actions.append(action)
        return actions

    def _follow_trajectory_mujoco(self, joint_trajectory: JointTrajectory):
        """
        Execute a joint trajectory using MuJoCo.
        
        Args:
            joint_trajectory: JointTrajectory object with positions
        """
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        
        callback_name = f"{self.__class__.__name__}_follow_trajectory_callback"
        
        # Reset callback state
        self.callback_active = False
        self.callback_should_remove = False
        
        # Remove the callback if it already exists
        if self.simulator.physics_callback_exists(
            f"{self.__class__.__name__}_follow_trajectory_callback"
        ):

            self.simulator.remove_physics_callback(
                f"{self.__class__.__name__}_follow_trajectory_callback"
            )
            self.future_actions = []
            self.logger.log_warning(
                f"Break the previous follow_trajectory_callback of {self.__class__.__name__}"
            )

        actions = self._create_actions_mujoco(joint_trajectory)
        self.future_actions.extend(actions)
        # Add the callback
        self.simulator.add_physics_callback(
            f"{self.__class__.__name__}_follow_trajectory_callback",
            callback_fn=self.follow_trajectory_callback_mujoco,
        )
        self.logger.log_debug(
            f"Add follow_trajectory_callback of {self.__class__.__name__}"
        )

    def follow_trajectory(self, joint_trajectory: JointTrajectory):
        """
        Execute a joint trajectory on the robot.
        
        Args:
            joint_trajectory: JointTrajectory object with positions
        """
        self._follow_trajectory_mujoco(joint_trajectory)

    def follow_trajectory_callback_mujoco(self, *args):
        """
        Callback function executed during physics steps to follow a trajectory.
        
        Args:
            *args: Additional arguments passed by the physics engine, typically the simulator instance
        """
        if len(self.future_actions) > 0:
            action_to_apply = self.future_actions.pop(0)
            self.set_joint_positions(action_to_apply["joint_positions"])
        else:
            self.simulator.remove_physics_callback(
                f"{self.__class__.__name__}_follow_trajectory_callback"
            )
            self.logger.log_debug(
                f"Remove follow_trajectory_callback of {self.__class__.__name__}"
            )

    def get_joint_names(self):
        """
        Get the joint names of the limb.
        """
        return self.joint_names

    def interpolate_joint_positions(self, start_pos, end_pos, alpha):
        """Interpolate between two joint positions using cubic interpolation.
        
        Args:
            start_pos: Starting joint positions
            end_pos: Target joint positions
            alpha: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Interpolated joint positions
        """
        # Use cubic interpolation for smoother motion
        alpha = alpha * alpha * (3 - 2 * alpha)  # Smoothstep interpolation
        return [s + (e - s) * alpha for s, e in zip(start_pos, end_pos)]


class BasicGripper(BasicLimb):
    def __init__(
        self, galbot_interface_config: GalbotInterfaceConfig, simulator: PhysicsSimulator
    ):
        super().__init__(galbot_interface_config, simulator)
        self._is_open = True  # Track if gripper is open
        self._is_close = False  # Track if gripper is closed

    @property
    def is_open(self) -> bool:
        """Check if the gripper is in open state.
        
        Returns:
            bool: True if gripper is open, False otherwise
        """
        return self._is_open

    @property
    def is_close(self) -> bool:
        """Check if the gripper is in closed state.
        
        Returns:
            bool: True if gripper is closed, False otherwise
        """
        return self._is_close

    def set_gripper_close(self):
        """Set the gripper to closed state."""
        self.set_gripper_width(0.0)
        self._is_open = False
        self._is_close = True

    def set_gripper_open(self):
        """Set the gripper to open state."""
        self.set_gripper_width(1.0)
        self._is_open = True
        self._is_close = False

    # TODO@Chenyu: fix the width logic
    def get_gripper_width(self):
        raise NotImplementedError(
            f"Function {self.get_gripper_width.__name__} is not implemented"
        )

    # def set_gripper_width(self, normalized_width: float):
    #     """
    #     Set the gripper width based on a normalized value between 0.0 (closed) and 1.0 (open).
        
    #     Args:
    #         normalized_width: Float between 0.0 and 1.0 representing the desired gripper width
    #     """
    #     self._set_gripper_width_robotiq(normalized_width)

    def set_gripper_width(self, normalized_width: float):
        """
        Set the Robotiq gripper width based on a normalized value.
        
        Args:
            normalized_width: Float between 0.0 and 1.0 representing the desired gripper width
        """
        def map_range(
            value: float, in_min: float, in_max: float, out_min: float, out_max: float
        ) -> float:
            """
            Linearly maps input value from [in_min, in_max] to [out_min, out_max].

            Args:
                value: Input value
                in_min: Input range minimum
                in_max: Input range maximum
                out_min: Output range minimum
                out_max: Output range maximum
            
            Returns:
                Mapped value
            """
            return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

        value_min, value_max = 0.0, 1.0  # Input range [0,1]
        width_min, width_max = (
            0,
            1.703,
        )  # Mapped range [0, 1.703] m
        # Inverse the value
        normalized_width = 1 - normalized_width

        # Calculate mapped gripper width
        real_gripper_width = map_range(
            normalized_width, value_min, value_max, width_min, width_max
        )

        joint_positions = [real_gripper_width]
        self.set_joint_positions(joint_positions)

    def _set_gripper_width_robotiq(self, normalized_width: float):
        """
        Set the Robotiq gripper width based on a normalized value.
        
        Args:
            normalized_width: Float between 0.0 and 1.0 representing the desired gripper width
        """
        def map_range(
            value: float, in_min: float, in_max: float, out_min: float, out_max: float
        ) -> float:
            """
            Linearly maps input value from [in_min, in_max] to [out_min, out_max].

            Args:
                value: Input value
                in_min: Input range minimum
                in_max: Input range maximum
                out_min: Output range minimum
                out_max: Output range maximum
            
            Returns:
                Mapped value
            """
            return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

        value_min, value_max = 0.0, 1.0  # Input range [0,1]
        width_min, width_max = (
            0,
            0.82,
        )  # Mapped range [0, 0.82] m
        # Inverse the value
        normalized_width = 1 - normalized_width

        # Calculate mapped gripper width
        real_gripper_width = map_range(
            normalized_width, value_min, value_max, width_min, width_max
        )

        joint_positions = [real_gripper_width]
        self.set_joint_positions(joint_positions)
