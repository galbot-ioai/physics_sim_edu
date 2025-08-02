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
# Description: IOAI main environment
# Author: Chenyu Cao, Herman Ye@Galbot
# Warning: Competition participants should not edit this file.
#
######################################################################################

"""IOAI Environment for physics simulation and robot control.

This module provides a comprehensive interface for controlling the Galbot robot
in a physics simulation environment, including inverse kinematics, motion planning,
and sensor data acquisition.

Typical usage example:
    env = IOAIEnv(headless=False)
    env.run()
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import mink
import numpy as np
from auro_utils import wxyz_to_xyzw, xyzw_to_wxyz
from loop_rate_limiters import RateLimiter
from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from physics_simulator.utils.state_machine import SimpleStateMachine
from synthnova_config import (
    CuboidConfig,
    DepthCameraConfig,
    MeshConfig,
    MujocoConfig,
    PhysicsSimulatorConfig,
    RealsenseD436DepthSensorConfig,
    RealsenseD436RgbSensorConfig,
    RobotConfig,
    RgbCameraConfig,
    ScenarioConfig,
)

def interpolate_joint_positions(
    start_positions: np.ndarray,
    end_positions: np.ndarray,
    steps: int
) -> List[np.ndarray]:
    """Interpolate between start and end joint positions.
    
    Args:
        start_positions: Initial joint positions as numpy array.
        end_positions: Target joint positions as numpy array.
        steps: Number of interpolation steps.
        
    Returns:
        List of interpolated joint position arrays.
        
    Example:
        >>> start = np.array([0.0, 0.0, 0.0])
        >>> end = np.array([1.0, 1.0, 1.0])
        >>> result = interpolate_joint_positions(start, end, 5)
        >>> len(result)
        5
    """
    return np.linspace(start_positions, end_positions, steps).tolist()

class IOAIEnv:
    """IOAI Environment for physics simulation and robot control.
    
    This class provides a comprehensive interface for controlling the Galbot robot
    in a physics simulation environment, including inverse kinematics, motion planning,
    and sensor data acquisition.
    
    The environment supports:
    - Robot arm control with inverse kinematics
    - Chassis movement and path following
    - Camera data acquisition
    - Coordinate frame transformations
    - Joint trajectory planning
    
    Attributes:
        simulator: Physics simulator instance.
        robot: Robot model instance.
        interface: Galbot interface for robot control.
        mink_config: Mink inverse kinematics configuration.
        tasks: Dictionary of IK tasks for different robot parts.
        velocity_limit: Velocity limits for joint movements.
        solver: IK solver type.
        damping: Damping parameter for IK solver.
        rate_limiter: Rate limiter for control loops.
        robot_initial_position: Initial robot position in world frame.
        robot_initial_orientation: Initial robot orientation in world frame.
        front_head_rgb_camera_path: Path to front head RGB camera.
        front_head_depth_camera_path: Path to front head depth camera.
        right_wrist_rgb_camera_path: Path to right wrist RGB camera.
        right_wrist_depth_camera_path: Path to right wrist depth camera.
        left_wrist_rgb_camera_path: Path to left wrist RGB camera.
        left_wrist_depth_camera_path: Path to left wrist depth camera.
    """
    
    def __init__(self, headless: bool = False) -> None:
        """Initialize the IOAI environment.
        
        Args:
            headless: Whether to run the simulator in headless mode.
            
        Raises:
            RuntimeError: If simulator initialization fails.
            
        Example:
            >>> env = IOAIEnv(headless=True)
            >>> env.simulator is not None
            True
        """
        # Initialize component references
        self.simulator: Optional[PhysicsSimulator] = None
        self.robot: Optional[Any] = None
        self.interface: Optional[GalbotInterface] = None
        self.mink_config: Optional[Any] = None
        self.tasks: dict[str, Any] = {}
        self.velocity_limit: Optional[Any] = None
        self.solver: Optional[str] = None
        self.damping: Optional[float] = None
        self.rate_limiter: Optional[RateLimiter] = None
        
        # Robot initial configuration in world frame
        self.robot_initial_position: List[float] = [0, 4, 0]
        self.robot_initial_orientation: List[float] = [0, 0, 0.70711, -0.70711]
        
        # Camera paths in the scene hierarchy
        self.front_head_rgb_camera_path: Optional[str] = None
        self.front_head_depth_camera_path: Optional[str] = None
        self.right_wrist_rgb_camera_path: Optional[str] = None
        self.right_wrist_depth_camera_path: Optional[str] = None
        self.left_wrist_rgb_camera_path: Optional[str] = None
        self.left_wrist_depth_camera_path: Optional[str] = None

        # Setup components in order
        self._setup_simulator(headless=headless)
        self._setup_interface()
        self._setup_mink()

    def _setup_simulator(self, headless: bool = False) -> None:
        """Setup the physics simulator with robot and camera configuration.
        
        This method initializes the physics simulator, loads the scenario configuration,
        and sets up camera paths for the robot's sensors.
        
        Args:
            headless: Whether to run the simulator in headless mode.
            
        Raises:
            FileNotFoundError: If scenario configuration file is not found.
            RuntimeError: If simulator initialization fails.
            
        Note:
            The scenario configuration file must be located at 'ioai_scenario.json'
            relative to this module's directory.
        """
        # Create simulator configuration
        scenario_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "ioai_scenario.json"
        )
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless),
            scenario_config=ScenarioConfig.load_from_file(scenario_config_path)
        )

        # Initialize the simulator
        self.simulator = PhysicsSimulator(sim_config)
        self.simulator.initialize()
        
        # Get robot reference from the scene
        self.robot = self.simulator.get_robot(prim_path="/World/Galbot")
        
        # Define camera paths in the scene hierarchy
        # These paths correspond to the camera sensors mounted on the robot
        self.front_head_rgb_camera_path = (
            "/World/Galbot/head_link2/head_end_effector_mount_link/front_head_rgb_camera"
        )
        self.front_head_depth_camera_path = (
            "/World/Galbot/head_link2/head_end_effector_mount_link/front_head_depth_camera"
        )
        self.right_wrist_rgb_camera_path = (
            "/World/Galbot/right_arm_link7/right_arm_end_effector_mount_link/right_wrist_rgb_camera"
        )
        self.right_wrist_depth_camera_path = (
            "/World/Galbot/right_arm_link7/right_arm_end_effector_mount_link/right_wrist_depth_camera"
        )
        self.left_wrist_rgb_camera_path = (
            "/World/Galbot/left_arm_link7/left_arm_end_effector_mount_link/left_wrist_rgb_camera"
        )
        self.left_wrist_depth_camera_path = (
            "/World/Galbot/left_arm_link7/left_arm_end_effector_mount_link/left_wrist_depth_camera"
        )

    def _setup_interface(self) -> None:
        """Setup the Galbot interface for robot control.
        
        This method configures the Galbot interface with all robot modules including
        arms, legs, head, chassis, grippers, and cameras. It sets up joint names
        and camera paths for each module.
        
        Raises:
            RuntimeError: If interface initialization fails.
            ValueError: If robot name cannot be determined.
            
        Note:
            The interface enables all major robot modules for comprehensive control.
            Joint names are constructed using the robot's namespace prefix.
        """
        galbot_interface_config = GalbotInterfaceConfig()
        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        
        # Enable all major robot modules for comprehensive control
        enabled_modules = [
            "right_arm", "left_arm", "leg", "head", "chassis", 
            "left_gripper", "right_gripper", "front_head_camera"
        ]
        galbot_interface_config.modules_manager.enabled_modules.extend(enabled_modules)

        # Configure joint names for each module using robot namespace
        # Right arm: 7-DOF manipulator
        galbot_interface_config.right_arm.joint_names = [
            f"{robot_name}/right_arm_joint1",
            f"{robot_name}/right_arm_joint2",
            f"{robot_name}/right_arm_joint3",
            f"{robot_name}/right_arm_joint4",
            f"{robot_name}/right_arm_joint5",
            f"{robot_name}/right_arm_joint6",
            f"{robot_name}/right_arm_joint7",
        ]

        # Left arm: 7-DOF manipulator
        galbot_interface_config.left_arm.joint_names = [
            f"{robot_name}/left_arm_joint1",
            f"{robot_name}/left_arm_joint2",
            f"{robot_name}/left_arm_joint3",
            f"{robot_name}/left_arm_joint4",
            f"{robot_name}/left_arm_joint5",
            f"{robot_name}/left_arm_joint6",
            f"{robot_name}/left_arm_joint7",
        ]

        # Leg: 4-DOF leg mechanism
        galbot_interface_config.leg.joint_names = [
            f"{robot_name}/leg_joint1",
            f"{robot_name}/leg_joint2",
            f"{robot_name}/leg_joint3",
            f"{robot_name}/leg_joint4",
        ]
        
        # Head: 2-DOF head mechanism
        galbot_interface_config.head.joint_names = [
            f"{robot_name}/head_joint1",
            f"{robot_name}/head_joint2"
        ]

        # Chassis: 3-DOF omnidirectional base
        galbot_interface_config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint",
            f"{robot_name}/mobile_yaw_joint",
        ]

        # Left gripper: Single joint for opening/closing
        galbot_interface_config.left_gripper.joint_names = [
            f"{robot_name}/left_gripper_r_knuckle_joint",
        ]

        # Right gripper: Single joint for opening/closing
        galbot_interface_config.right_gripper.joint_names = [
            f"{robot_name}/right_gripper_r_knuckle_joint",
        ]

        # Configure front head camera paths for RGB and depth sensors
        galbot_interface_config.front_head_camera.prim_path_rgb = self.front_head_rgb_camera_path
        galbot_interface_config.front_head_camera.prim_path_depth = self.front_head_depth_camera_path

        # Initialize interface with simulator
        self.interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        self.interface.initialize()

    def _setup_mink(self) -> None:
        """Initialize Mink inverse kinematics solver configuration.
        
        This method sets up the Mink IK solver with tasks for different robot parts
        including torso, posture, chassis, and arms. It configures velocity limits
        and solver parameters for optimal IK performance.
        
        Raises:
            RuntimeError: If Mink configuration fails to initialize.
            
        Note:
            The IK tasks are configured with different cost weights to prioritize
            certain objectives over others. Velocity limits are set to ensure
            safe and smooth robot movements.
        """
        model = self.simulator.model._model
        self.mink_config = mink.Configuration(model)
        
        # Create IK tasks for different robot parts with appropriate cost weights
        # Torso task: Maintains torso orientation and position
        self.tasks = {
            "torso": mink.FrameTask(
                frame_name=self.robot.namespace + "torso_base_link",
                frame_type="body",
                position_cost=1e6,
                orientation_cost=1e6,
            ),
            # Posture task: Maintains natural robot posture
            "posture": mink.PostureTask(model, cost=1.0),
            # Chassis task: Maintains chassis position and orientation
            "chassis": mink.FrameTask(
                frame_name=self.robot.namespace + "omni_chassis_base_link",
                frame_type="body",
                position_cost=1e6,
                orientation_cost=1e6,
            ),
            # Left arm task: Controls left gripper TCP pose
            "left_arm": mink.FrameTask(
                frame_name=self.robot.namespace + "left_gripper_tcp",
                frame_type="site",
                position_cost=50.0,
                orientation_cost=50.0,
                lm_damping=1.0,
            ),
            # Right arm task: Controls right gripper TCP pose
            "right_arm": mink.FrameTask(
                frame_name=self.robot.namespace + "right_gripper_tcp",
                frame_type="site",
                position_cost=50.0,
                orientation_cost=50.0,
                lm_damping=1.0,
            )
        }

        # Set velocity limits for arm joints to ensure safe movements
        self.velocity_limit = mink.VelocityLimit(
            model, 
            velocities={
                name: 2.0 for name in self.interface.left_arm.joint_names
                + self.interface.right_arm.joint_names
            }
        )
        
        # Solver configuration for optimal IK performance
        self.solver = "daqp"  # DAQP solver for better convergence
        self.damping = 1e-3   # Damping parameter for numerical stability
        self.rate_limiter = RateLimiter(frequency=1000, warn=False)

    def world_to_robot_frame(
        self, 
        world_position: np.ndarray, 
        world_orientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pose from world frame to robot base frame.
        
        This method transforms a pose (position and orientation) from the world
        coordinate frame to the robot's base coordinate frame using the current
        robot base pose.
        
        Args:
            world_position: Position in world frame as [x, y, z] array.
            world_orientation: Orientation in world frame as [qx, qy, qz, qw] array.
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame.
            robot_position: Position in robot base frame as [x, y, z] array.
            robot_orientation: Orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> world_pos = np.array([1.0, 2.0, 0.5])
            >>> world_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> robot_pos, robot_ori = env.world_to_robot_frame(world_pos, world_ori)
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        base_rotation = Rotation.from_quat(base_orientation)
        world_rotation = Rotation.from_quat(world_orientation)
        
        # Transform position: subtract base position and rotate
        relative_position = world_position - base_position
        robot_position = base_rotation.inv().apply(relative_position)
        
        # Transform orientation: compose rotations
        robot_orientation = (base_rotation.inv() * world_rotation).as_quat()
        
        return robot_position, robot_orientation

    def robot_to_world_frame(
        self, 
        robot_position: np.ndarray, 
        robot_orientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pose from robot base frame to world frame.
        
        This method transforms a pose (position and orientation) from the robot's
        base coordinate frame to the world coordinate frame using the current
        robot base pose.
        
        Args:
            robot_position: Position in robot base frame as [x, y, z] array.
            robot_orientation: Orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Returns:
            Tuple of (world_position, world_orientation) in world frame.
            world_position: Position in world frame as [x, y, z] array.
            world_orientation: Orientation in world frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> robot_pos = np.array([0.5, 0.0, 0.3])
            >>> robot_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> world_pos, world_ori = env.robot_to_world_frame(robot_pos, robot_ori)
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        base_rotation = Rotation.from_quat(base_orientation)
        robot_rotation = Rotation.from_quat(robot_orientation)
        
        # Transform position: rotate and add base position
        world_position = base_rotation.apply(robot_position) + base_position
        
        # Transform orientation: compose rotations
        world_orientation = (base_rotation * robot_rotation).as_quat()
        
        return world_position, world_orientation

    def world_to_robot_initial_frame(
        self, 
        world_position: np.ndarray, 
        world_orientation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform pose from world frame to robot initial frame.
        
        This method transforms a pose from the world coordinate frame to the robot's
        initial coordinate frame using the robot's initial pose configuration.
        
        Args:
            world_position: Position in world frame as [x, y, z] array.
            world_orientation: Orientation in world frame as [qx, qy, qz, qw] array (optional).
            
        Returns:
            Tuple of (robot_initial_position, robot_initial_orientation) in robot initial frame.
            robot_initial_position: Position in robot initial frame as [x, y, z] array.
            robot_initial_orientation: Orientation in robot initial frame as [qx, qy, qz, qw] array,
                or None if world_orientation was not provided.
            
        Example:
            >>> world_pos = np.array([1.0, 2.0, 0.5])
            >>> robot_pos, robot_ori = env.world_to_robot_initial_frame(world_pos)
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot initial pose in world frame
        initial_position = self.robot_initial_position
        initial_orientation = self.robot_initial_orientation
        
        # Create transformation matrix for initial pose
        initial_rotation = Rotation.from_quat(initial_orientation)
        
        # Transform position: subtract initial position and rotate
        relative_position = world_position - initial_position
        robot_initial_position = initial_rotation.inv().apply(relative_position)
        
        # Transform orientation if provided
        if world_orientation is not None:
            world_rotation = Rotation.from_quat(world_orientation)
            robot_initial_orientation = (initial_rotation.inv() * world_rotation).as_quat()
        else:
            robot_initial_orientation = None
        
        return robot_initial_position, robot_initial_orientation

    def robot_initial_to_world_frame(
        self, 
        robot_initial_position: np.ndarray, 
        robot_initial_orientation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform pose from robot initial frame to world frame.
        
        This method transforms a pose from the robot's initial coordinate frame to the
        world coordinate frame using the robot's initial pose configuration.
        
        Args:
            robot_initial_position: Position in robot initial frame as [x, y, z] array.
            robot_initial_orientation: Orientation in robot initial frame as [qx, qy, qz, qw] array (optional).
            
        Returns:
            Tuple of (world_position, world_orientation) in world frame.
            world_position: Position in world frame as [x, y, z] array.
            world_orientation: Orientation in world frame as [qx, qy, qz, qw] array,
                or None if robot_initial_orientation was not provided.
            
        Example:
            >>> robot_pos = np.array([0.5, 0.0, 0.3])
            >>> world_pos, world_ori = env.robot_initial_to_world_frame(robot_pos)
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot initial pose in world frame
        initial_position = self.robot_initial_position
        initial_orientation = self.robot_initial_orientation
        
        # Create transformation matrix for initial pose
        initial_rotation = Rotation.from_quat(initial_orientation)
        
        # Transform position: rotate and add initial position
        world_position = initial_rotation.apply(robot_initial_position) + initial_position
        
        # Transform orientation if provided
        if robot_initial_orientation is not None:
            robot_initial_rot = Rotation.from_quat(robot_initial_orientation)
            world_orientation = (initial_rotation * robot_initial_rot).as_quat()
        else:
            world_orientation = None
        
        return world_position, world_orientation

    def camera_to_robot_frame(
        self, 
        camera_position: np.ndarray, 
        camera_orientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pose from camera frame to robot base frame.
        
        This method transforms a pose from the front head camera coordinate frame to the
        robot's base coordinate frame using the camera's current pose in world frame.
        
        Args:
            camera_position: Position in camera frame as [x, y, z] array.
            camera_orientation: Orientation in camera frame as [qx, qy, qz, qw] array.
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame.
            robot_position: Position in robot base frame as [x, y, z] array.
            robot_orientation: Orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> camera_pos = np.array([0.1, 0.0, 0.5])
            >>> camera_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> robot_pos, robot_ori = env.camera_to_robot_frame(camera_pos, camera_ori)
        """
        from scipy.spatial.transform import Rotation
        
        # Get camera pose in world frame
        camera_prim_path = self.front_head_rgb_camera_path
        camera_state = self.simulator.get_sensor_state(camera_prim_path)
        camera_world_position = camera_state["transform_to_base_link"]["position"]
        camera_world_orientation = camera_state["transform_to_base_link"]["orientation"]
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        camera_world_rotation = Rotation.from_quat(camera_world_orientation)
        camera_local_rotation = Rotation.from_quat(camera_orientation)
        base_rotation = Rotation.from_quat(base_orientation)
        
        # Transform position: camera frame -> world frame -> robot frame
        world_position = camera_world_rotation.apply(camera_position) + camera_world_position
        relative_position = world_position - base_position
        robot_position = base_rotation.inv().apply(relative_position)
        
        # Transform orientation: camera frame -> world frame -> robot frame
        world_orientation = (camera_world_rotation * camera_local_rotation).as_quat()
        robot_orientation = (base_rotation.inv() * Rotation.from_quat(world_orientation)).as_quat()
        
        return robot_position, robot_orientation

    def robot_to_camera_frame(
        self, 
        robot_position: np.ndarray, 
        robot_orientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pose from robot base frame to camera frame.
        
        This method transforms a pose from the robot's base coordinate frame to the
        front head camera coordinate frame using the camera's current pose in world frame.
        
        Args:
            robot_position: Position in robot base frame as [x, y, z] array.
            robot_orientation: Orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Returns:
            Tuple of (camera_position, camera_orientation) in camera frame.
            camera_position: Position in camera frame as [x, y, z] array.
            camera_orientation: Orientation in camera frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> robot_pos = np.array([0.5, 0.0, 0.3])
            >>> robot_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> camera_pos, camera_ori = env.robot_to_camera_frame(robot_pos, robot_ori)
        """
        from scipy.spatial.transform import Rotation
        
        # Get camera pose in world frame
        camera_prim_path = self.front_head_rgb_camera_path
        camera_state = self.simulator.get_sensor_state(camera_prim_path)
        camera_world_position = camera_state["transform_to_base_link"]["position"]
        camera_world_orientation = camera_state["transform_to_base_link"]["orientation"]
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        camera_world_rotation = Rotation.from_quat(camera_world_orientation)
        base_rotation = Rotation.from_quat(base_orientation)
        robot_rotation = Rotation.from_quat(robot_orientation)
        
        # Transform position: robot frame -> world frame -> camera frame
        world_position = base_rotation.apply(robot_position) + base_position
        relative_position = world_position - camera_world_position
        camera_position = camera_world_rotation.inv().apply(relative_position)
        
        # Transform orientation: robot frame -> world frame -> camera frame
        world_orientation = (base_rotation * robot_rotation).as_quat()
        camera_orientation = (camera_world_rotation.inv() * Rotation.from_quat(world_orientation)).as_quat()
        
        return camera_position, camera_orientation

    def world_to_robot_initial_frame_2d(
        self, 
        world_position_2d: np.ndarray
    ) -> np.ndarray:
        """Transform 2D position from world frame to robot initial frame.
        
        This method transforms a 2D position from the world coordinate frame to the
        robot's initial coordinate frame using the robot's initial pose configuration.
        The z-coordinate is assumed to be 0 for the 2D transformation.
        
        Args:
            world_position_2d: Position in world frame as [x, y] array.
            
        Returns:
            robot_initial_position_2d: Position in robot initial frame as [x, y] array.
            
        Example:
            >>> world_pos_2d = np.array([1.0, 2.0])
            >>> robot_pos_2d = env.world_to_robot_initial_frame_2d(world_pos_2d)
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot initial pose in world frame
        initial_position = self.robot_initial_position
        initial_orientation = self.robot_initial_orientation
        
        # Create transformation matrix for initial pose
        initial_rotation = Rotation.from_quat(initial_orientation)
        
        # Convert 2D to 3D by adding z=0 and ensure numpy arrays
        world_position_3d = np.array([world_position_2d[0], world_position_2d[1], 0.0])
        initial_position_array = np.array(initial_position, dtype=float)
        
        # Transform position: subtract initial position and rotate
        relative_position = world_position_3d - initial_position_array
        robot_initial_position_3d = initial_rotation.inv().apply(relative_position)
        
        # Return only 2D coordinates
        return robot_initial_position_3d[:2]

    def compute_inverse_kinematics(
        self, 
        start_joint_config: np.ndarray, 
        target_pose: np.ndarray, 
        arm_id: str = "left_arm"
    ) -> np.ndarray:
        """Compute inverse kinematics using Mink solver.
        
        This method solves the inverse kinematics problem to find joint configurations
        that achieve a desired end-effector pose. It uses the Mink IK solver with
        multiple tasks including torso, posture, chassis, and arm control.
        
        Args:
            start_joint_config: Initial joint configuration (not used in current implementation).
            target_pose: Target pose as [x, y, z, qx, qy, qz, qw] array in robot base frame.
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            
        Returns:
            Target joint configuration for the specified arm as numpy array.
            
        Raises:
            ValueError: If arm_id is invalid.
            RuntimeError: If IK solver fails to converge.
            
        Example:
            >>> target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
            >>> joint_config = env.compute_inverse_kinematics(start_config, target_pose, "left_arm")
        """
        # Transform target pose from robot frame to world frame for IK
        target_position = target_pose[:3]
        target_orientation = target_pose[3:7]
        world_position, world_orientation = self.robot_to_world_frame(target_position, target_orientation)
        
        # Set target for chassis to maintain current pose
        chassis_target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=xyzw_to_wxyz(self.robot.get_orientation())),
            translation=self.robot.get_position()
        )
        self.tasks["chassis"].set_target(chassis_target)

        # Set target for torso to maintain current pose
        import mujoco
        torso_body_id = mujoco.mj_name2id(
            self.simulator.model._model, 
            mujoco.mjtObj.mjOBJ_BODY, 
            self.robot.namespace + "torso_base_link"
        )
        torso_target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=self.simulator.data.xquat[torso_body_id]),
            translation=self.simulator.data.xpos[torso_body_id]
        )
        self.tasks["torso"].set_target(torso_target)

        # Set target for posture to maintain natural configuration
        self.tasks["posture"].set_target_from_configuration(self.mink_config)

        # Set target for the specified arm using world frame pose
        if arm_id == "left_arm":
            target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=xyzw_to_wxyz(world_orientation)),
                translation=world_position
            )
            self.tasks["left_arm"].set_target(target)
            tasks = [self.tasks["torso"], self.tasks["posture"], self.tasks["chassis"], self.tasks["left_arm"]]
        elif arm_id == "right_arm":
            target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=xyzw_to_wxyz(world_orientation)),
                translation=world_position
            )
            self.tasks["right_arm"].set_target(target)
            tasks = [self.tasks["torso"], self.tasks["posture"], self.tasks["chassis"], self.tasks["right_arm"]]
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 'left_arm' or 'right_arm'.")
        
        # Iterative IK solving to get final positions
        dt = 1e-3
        max_iterations = 50
        position_tolerance = 1e-2
        orientation_tolerance = 1e-2
        
        converged = False
        for iteration in range(max_iterations):
            # Solve IK for velocity
            velocity = mink.solve_ik(
                self.mink_config,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=[self.velocity_limit] if False else None
            )
            
            # Integrate to update configuration
            self.mink_config.integrate_inplace(velocity, dt)
            
            # Check convergence for the specified arm
            error = self.tasks[arm_id].compute_error(self.mink_config)
            position_error = np.linalg.norm(error[:3])
            orientation_error = np.linalg.norm(error[3:])
            if position_error < position_tolerance and orientation_error < orientation_tolerance:
                converged = True
                break

        # Check if IK converged and warn if not
        if not converged:
            error = self.tasks[arm_id].compute_error(self.mink_config)
            position_error = np.linalg.norm(error[:3])
            orientation_error = np.linalg.norm(error[3:])
            warnings.warn(
                f"IK did not fully converge for {arm_id} after {max_iterations} iterations. "
                f"Final position error: {position_error:.6f} (tolerance: {position_tolerance}), "
                f"orientation error: {orientation_error:.6f} (tolerance: {orientation_tolerance}). "
                f"Proceeding with current solution."
            )

        # Get final joint positions
        joint_positions = self.mink_config.q

        # Extract joint positions for the specified arm
        if arm_id == "left_arm":
            arm_joint_indexes = self.interface.left_arm.joint_indexes
        else:  # right_arm
            arm_joint_indexes = self.interface.right_arm.joint_indexes
        
        arm_joint_positions = joint_positions[arm_joint_indexes]
        return arm_joint_positions

    def compute_forward_kinematics(self, arm_id: str = "left_arm") -> np.ndarray:
        """Compute forward kinematics using ground truth from simulator.
        
        This method computes the forward kinematics by getting the current TCP pose
        from the simulator and transforming it to the robot base frame.
        
        Args:
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            
        Returns:
            TCP pose in base link frame as [x, y, z, qx, qy, qz, qw] array.
            
        Raises:
            ValueError: If arm_id is invalid.
            
        Example:
            >>> tcp_pose = env.compute_forward_kinematics("left_arm")
            >>> print(f"TCP position: {tcp_pose[:3]}")
            >>> print(f"TCP orientation: {tcp_pose[3:]}")
        """
        if arm_id == "left_arm":
            # Get left gripper TCP pose from simulator
            position, quaternion = self.get_left_gripper_pose()
        elif arm_id == "right_arm":
            # Get right gripper TCP pose from simulator
            position, quaternion = self.get_right_gripper_pose()
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 'left_arm' or 'right_arm'.")
        
        # Transform from world frame to base link frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        from scipy.spatial.transform import Rotation

        # World to base transformation
        base_rotation = Rotation.from_quat(base_orientation)
        
        # TCP in world frame
        tcp_rotation = Rotation.from_quat(quaternion)
        
        # Transform position: subtract base position and rotate
        relative_position = position - base_position
        tcp_position_base = base_rotation.inv().apply(relative_position)
        
        # Transform orientation: compose rotations
        tcp_orientation_base = (base_rotation.inv() * tcp_rotation).as_quat()
        
        # Return pose in base link frame [x, y, z, qx, qy, qz, qw]
        return np.concatenate([tcp_position_base, tcp_orientation_base])

    def get_left_gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get left gripper TCP pose in world frame.
        
        This method retrieves the current pose of the left gripper's TCP (Tool Center Point)
        from the simulator in world coordinates.
        
        Returns:
            Tuple of (position, quaternion) in world frame.
            position: TCP position as [x, y, z] array in world frame.
            quaternion: TCP orientation as [qx, qy, qz, qw] array in world frame.
            
        Example:
            >>> position, quaternion = env.get_left_gripper_pose()
            >>> print(f"Left gripper position: {position}")
        """
        site_data = self.simulator.data.site(self.robot.namespace + "left_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion
    
    def get_right_gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get right gripper TCP pose in world frame.
        
        This method retrieves the current pose of the right gripper's TCP (Tool Center Point)
        from the simulator in world coordinates.
        
        Returns:
            Tuple of (position, quaternion) in world frame.
            position: TCP position as [x, y, z] array in world frame.
            quaternion: TCP orientation as [qx, qy, qz, qw] array in world frame.
            
        Example:
            >>> position, quaternion = env.get_right_gripper_pose()
            >>> print(f"Right gripper position: {position}")
        """
        site_data = self.simulator.data.site(self.robot.namespace + "right_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion
    
    def move_arm_to_pose(
        self, 
        arm_id: str, 
        target_position: np.ndarray, 
        target_orientation: np.ndarray
    ) -> None:
        """Move arm to target pose using inverse kinematics.
        
        This method moves the specified arm to a target pose by solving inverse kinematics
        and executing a smooth joint trajectory to reach the target.
        
        Args:
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            target_position: Target position in robot base frame as [x, y, z] array.
            target_orientation: Target orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Raises:
            ValueError: If arm_id is invalid.
            RuntimeError: If IK solution fails or trajectory execution fails.
            
        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> env.move_arm_to_pose("left_arm", target_pos, target_ori)
        """
        # Prepare target pose in robot frame
        target_pose = np.concatenate([target_position, target_orientation])
        
        # Solve IK and start motion
        current_joints = self.mink_config.q
        arm_joints = self.compute_inverse_kinematics(current_joints, target_pose, arm_id)
        arm_module = getattr(self.interface, arm_id)
        self._move_joints_to_target(arm_module, arm_joints)

    def move_left_arm_to_pose(
        self, 
        target_position: np.ndarray, 
        target_orientation: np.ndarray
    ) -> None:
        """Move left arm to target pose.
        
        Convenience method to move the left arm to a specific pose using inverse kinematics.
        
        Args:
            target_position: Target position in robot base frame as [x, y, z] array.
            target_orientation: Target orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> env.move_left_arm_to_pose(target_pos, target_ori)
        """
        return self.move_arm_to_pose("left_arm", target_position, target_orientation)
    
    def move_right_arm_to_pose(
        self, 
        target_position: np.ndarray, 
        target_orientation: np.ndarray
    ) -> None:
        """Move right arm to target pose.
        
        Convenience method to move the right arm to a specific pose using inverse kinematics.
        
        Args:
            target_position: Target position in robot base frame as [x, y, z] array.
            target_orientation: Target orientation in robot base frame as [qx, qy, qz, qw] array.
            
        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_ori = np.array([0.0, 0.0, 0.0, 1.0])
            >>> env.move_right_arm_to_pose(target_pos, target_ori)
        """
        return self.move_arm_to_pose("right_arm", target_position, target_orientation)

    def move_chassis_follow_path(self, waypoints: List[Tuple[float, float]]) -> None:
        """Move chassis to follow a path defined by waypoints in world coordinates.
        
        This method moves the robot chassis along a path defined by 2D waypoints in world
        coordinates. The robot follows the path using a path following algorithm with
        velocity control.
        
        Args:
            waypoints: List of 2D waypoints as [(x1, y1), (x2, y2), ...] in world frame.
                Each waypoint is a tuple of (x, y) coordinates.
                
        Raises:
            ValueError: If waypoints list is empty or has fewer than 2 points.
            
        Example:
            >>> path = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
            >>> env.move_chassis_follow_path(path)
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints: need at least 2 points for path following")
        
        # Initialize path following components
        from physics_simulator.utils.control_utils import BasicPathFollower
        path_follower = BasicPathFollower(velocity=0.8)
        waypoint_tolerance = 0.1
        current_target_index = 0
        path = waypoints

        def follow_path_callback() -> None:
            """Callback function for path following control loop."""
            nonlocal current_target_index
            
            # Check if path is complete
            if current_target_index >= len(path):
                self.interface.chassis.set_joint_velocities([0.0, 0.0, 0.0])
                self.simulator.remove_physics_callback("follow_path_callback")
                return
                
            # Get current state
            current_pos = self.interface.chassis.get_joint_positions()[:2]
            current_heading = self.interface.chassis.get_joint_positions()[2]
            
            # Update target waypoint
            if current_target_index < len(path):
                target_pos = path[current_target_index]
                target_pos = self.world_to_robot_initial_frame_2d(target_pos)
                
                distance = math.sqrt(
                    (target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2
                )
                if distance < waypoint_tolerance:
                    current_target_index += 1
                    return
            else:
                target_pos = path[-1]
                
            # Calculate control commands
            forward_vel, side_vel, yaw_vel = path_follower.calculate_control(
                current_pos, current_heading, target_pos
            )
            
            # Set chassis velocities
            self.interface.chassis.set_joint_velocities([forward_vel, side_vel, yaw_vel])
            
        # Add physics callback for path following
        self.simulator.add_physics_callback("follow_path_callback", follow_path_callback)

    def move_chassis_xy(
        self, 
        waypoints: List[Tuple[float, float]], 
        velocity: float = 0.8
    ) -> None:
        """Move chassis to follow waypoints in world coordinates.
        
        This method moves the robot chassis to follow a sequence of 2D waypoints in world
        coordinates. The robot moves directly towards each waypoint with constant velocity
        until all waypoints are reached.
        
        Args:
            waypoints: List of 2D waypoints as [(x1, y1), (x2, y2), ...] in world frame.
                Each waypoint is a tuple of (x, y) coordinates.
            velocity: Chassis velocity in m/s, default 0.8.
                
        Raises:
            ValueError: If waypoints list is empty.
            
        Example:
            >>> waypoints = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
            >>> env.move_chassis_xy(waypoints, velocity=1.0)
        """
        if not waypoints or len(waypoints) < 1:
            raise ValueError("Invalid waypoints: need at least 1 point for xy movement")
        
        # Convert waypoints from world frame to robot initial frame
        waypoints_robot = [self.world_to_robot_initial_frame_2d(wp) for wp in waypoints]
        
        current_waypoint_index = 0
        
        def move_xy_callback() -> None:
            """Callback function for xy movement control loop."""
            nonlocal current_waypoint_index
            
            # Check if all waypoints are reached
            if current_waypoint_index >= len(waypoints_robot):
                self.interface.chassis.set_joint_velocities([0.0, 0.0, 0.0])
                self.simulator.remove_physics_callback("move_xy_callback")
                return
            
            # Get current position in robot frame
            current_pos = self.interface.chassis.get_joint_positions()[:2]
            
            # Get current target waypoint
            target_pos_robot = waypoints_robot[current_waypoint_index]
            
            # Calculate distance to current waypoint
            distance = math.sqrt(
                (target_pos_robot[0] - current_pos[0])**2 + (target_pos_robot[1] - current_pos[1])**2
            )
            
            # Check if current waypoint is reached
            if distance < 0.01:  # 1cm tolerance
                current_waypoint_index += 1
                return
            
            # Calculate direction vector
            direction_x = target_pos_robot[0] - current_pos[0]
            direction_y = target_pos_robot[1] - current_pos[1]
            
            # Normalize direction vector
            if distance > 0:
                direction_x /= distance
                direction_y /= distance
            
            # Set velocities based on direction and target velocity
            forward_vel = direction_x * velocity
            side_vel = direction_y * velocity
            
            # Set chassis velocities (forward, side, yaw)
            self.interface.chassis.set_joint_velocities([forward_vel, side_vel, 0.0])
        
        # Add physics callback for xy movement
        self.simulator.add_physics_callback("move_xy_callback", move_xy_callback)

    def move_chassis_rotate(
        self, 
        target_angle_world: float, 
        angular_velocity: float = 1.0
    ) -> None:
        """Rotate chassis to face a specific angle in world coordinates.
        
        This method rotates the robot chassis to face a specific angle in world coordinates.
        The rotation is performed around the z-axis (yaw) with constant angular velocity.
        
        Args:
            target_angle_world: Target angle in world frame in radians, where 0 is along positive x-axis.
            angular_velocity: Angular velocity for rotation in rad/s, default 1.0.
                
        Example:
            >>> # Rotate to face 90 degrees (π/2 radians) from positive x-axis
            >>> env.move_chassis_rotate(math.pi / 2, angular_velocity=1.5)
        """
        
        # Calculate target heading in robot frame
        from scipy.spatial.transform import Rotation
        initial_rotation = Rotation.from_quat(self.robot_initial_orientation)
        initial_z_angle = initial_rotation.as_euler('xyz')[2]  # Extract z-axis rotation
        target_heading = -initial_z_angle + target_angle_world
        
        def rotate_callback() -> None:
            """Callback function for rotation control loop."""
            current_heading = self.interface.chassis.get_joint_positions()[2]
            heading_error = target_heading - current_heading
            
            # Normalize error to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            if abs(heading_error) < 0.05:  # 5 degrees tolerance
                self.interface.chassis.set_joint_velocities([0.0, 0.0, 0.0])
                self.simulator.remove_physics_callback("rotate_callback")
            else:
                yaw_vel = angular_velocity if heading_error > 0 else -angular_velocity
                self.interface.chassis.set_joint_velocities([0.0, 0.0, yaw_vel])
        
        self.simulator.add_physics_callback("rotate_callback", rotate_callback)

    def get_camera_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get RGB and depth images from the front head camera.
        
        This method retrieves RGB and depth images from the front head camera.
        If depth image is not available, it returns None for the depth image.
        
        Returns:
            Tuple of (rgb_image, depth_image).
            rgb_image: RGB image as numpy array, or None if not available.
            depth_image: Depth image as numpy array, or None if not available.
            
        Example:
            >>> rgb_img, depth_img = env.get_camera_images()
            >>> if rgb_img is not None:
            ...     print(f"RGB image shape: {rgb_img.shape}")
            >>> if depth_img is not None:
            ...     print(f"Depth image shape: {depth_img.shape}")
        """
        try:
            # Get RGB image
            rgb_image = self.interface.front_head_camera.get_rgb()
            
            # Get depth image if available
            depth_image = None
            try:
                depth_image = self.interface.front_head_camera.get_depth()
            except:
                pass  # Depth image not available
                
            return rgb_image, depth_image
        except Exception as e:
            print(f"Error getting camera images: {e}")
            return None, None

    def _move_joints_to_target(
        self, 
        module: Any, 
        target_positions: np.ndarray, 
        steps: int = 500
    ) -> None:
        """Move joints from current position to target position smoothly.
        
        This method creates a smooth trajectory from current joint positions to target
        positions and executes it using the robot module's trajectory following capability.
        
        Args:
            module: Robot module to control (e.g., left_arm, right_arm).
            target_positions: Target joint positions as numpy array.
            steps: Number of interpolation steps for smooth motion, default 500.
            
        Example:
            >>> target_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            >>> env._move_joints_to_target(env.interface.left_arm, target_joints)
        """
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _is_joint_positions_reached(
        self, 
        module: Any, 
        target_positions: np.ndarray, 
        atol: float = 0.01
    ) -> bool:
        """Check if joint positions are reached within tolerance.
        
        This method checks whether the current joint positions of a module are within
        the specified tolerance of the target positions.
        
        Args:
            module: Robot module to check (e.g., left_arm, right_arm).
            target_positions: Target joint positions as numpy array.
            atol: Absolute tolerance for position comparison in radians, default 0.01.
            
        Returns:
            True if positions are reached within tolerance, False otherwise.
            
        Example:
            >>> target_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            >>> reached = env._is_joint_positions_reached(env.interface.left_arm, target_joints)
        """
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=atol)

    def run(self) -> None:
        """Start the simulation loop.
        
        This method starts the main simulation loop that runs the physics simulation
        and executes any registered physics callbacks.
        
        Example:
            >>> env = IOAIEnv(headless=False)
            >>> env.run()  # This will start the simulation and run indefinitely
        """
        self.simulator.loop()

if __name__ == "__main__":
    env = IOAIEnv(headless=False)
    #TODO: Define your callbacks here
    # def demo_callback():
    #     print("demo callback")
    # env.simulator.add_physics_callback("demo_callback", demo_callback)

    env.run()