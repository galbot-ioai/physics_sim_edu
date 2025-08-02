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
# Description: IOAI Environment
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-07-28
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import (
    MujocoConfig,
    PhysicsSimulatorConfig,
    RobotConfig,
    MeshConfig,
    CuboidConfig,
    RgbCameraConfig,
    RealsenseD436RgbSensorConfig,
    DepthCameraConfig,
    RealsenseD436DepthSensorConfig,
    ScenarioConfig
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
import mink
from loop_rate_limiters import RateLimiter
from auro_utils import xyzw_to_wxyz, wxyz_to_xyzw
from pathlib import Path
import numpy as np
from physics_simulator.utils.data_types import JointTrajectory
import time
import os
import math
import warnings
from physics_simulator.utils.state_machine import SimpleStateMachine

def interpolate_joint_positions(start_positions, end_positions, steps):
    """Interpolate between start and end joint positions.
    
    Args:
        start_positions: Initial joint positions.
        end_positions: Target joint positions.
        steps: Number of interpolation steps.
        
    Returns:
        List of interpolated joint position arrays.
    """
    return np.linspace(start_positions, end_positions, steps).tolist()

class IOAIEnv:
    """IOAI Environment for physics simulation and robot control.
    
    This class provides a comprehensive interface for controlling the Galbot robot
    in a physics simulation environment, including inverse kinematics, motion planning,
    and sensor data acquisition.
    """
    
    def __init__(self, headless=False):
        """Initialize the IOAI environment.
        
        Args:
            headless: Whether to run the simulator in headless mode.
        """
        self.simulator = None
        self.robot = None
        self.interface = None
        self.mink_config = None
        self.tasks = {}
        self.velocity_limit = None
        self.solver = None
        self.damping = None
        self.rate_limiter = None
        
        # Robot initial configuration
        self.robot_initial_position = [0, 4, 0]
        self.robot_initial_orientation = [0, 0, 0.70711, -0.70711]
        
        # Camera paths
        self.front_head_rgb_camera_path = None
        self.front_head_depth_camera_path = None
        self.right_wrist_rgb_camera_path = None
        self.right_wrist_depth_camera_path = None
        self.left_wrist_rgb_camera_path = None
        self.left_wrist_depth_camera_path = None

        # Setup components
        self._setup_simulator(headless=headless)
        self._setup_interface()
        self._setup_mink()

    def _setup_simulator(self, headless=False):
        """Setup the physics simulator.
        
        Args:
            headless: Whether to run the simulator in headless mode.
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
        
        # Get robot reference
        self.robot = self.simulator.get_robot(prim_path="/World/Galbot")
        
        # Define camera paths
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

    def _setup_interface(self):
        """Setup the Galbot interface for robot control."""
        galbot_interface_config = GalbotInterfaceConfig()
        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        
        # Enable robot modules
        enabled_modules = [
            "right_arm", "left_arm", "leg", "head", "chassis", 
            "left_gripper", "right_gripper", "front_head_camera"
        ]
        galbot_interface_config.modules_manager.enabled_modules.extend(enabled_modules)

        # Configure joint names for each module
        galbot_interface_config.right_arm.joint_names = [
            f"{robot_name}/right_arm_joint1",
            f"{robot_name}/right_arm_joint2",
            f"{robot_name}/right_arm_joint3",
            f"{robot_name}/right_arm_joint4",
            f"{robot_name}/right_arm_joint5",
            f"{robot_name}/right_arm_joint6",
            f"{robot_name}/right_arm_joint7",
        ]

        galbot_interface_config.left_arm.joint_names = [
            f"{robot_name}/left_arm_joint1",
            f"{robot_name}/left_arm_joint2",
            f"{robot_name}/left_arm_joint3",
            f"{robot_name}/left_arm_joint4",
            f"{robot_name}/left_arm_joint5",
            f"{robot_name}/left_arm_joint6",
            f"{robot_name}/left_arm_joint7",
        ]

        galbot_interface_config.leg.joint_names = [
            f"{robot_name}/leg_joint1",
            f"{robot_name}/leg_joint2",
            f"{robot_name}/leg_joint3",
            f"{robot_name}/leg_joint4",
        ]
        
        galbot_interface_config.head.joint_names = [
            f"{robot_name}/head_joint1",
            f"{robot_name}/head_joint2"
        ]

        galbot_interface_config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint",
            f"{robot_name}/mobile_yaw_joint",
        ]

        galbot_interface_config.left_gripper.joint_names = [
            f"{robot_name}/left_gripper_r_knuckle_joint",
        ]

        galbot_interface_config.right_gripper.joint_names = [
            f"{robot_name}/right_gripper_r_knuckle_joint",
        ]

        # Configure front head camera
        galbot_interface_config.front_head_camera.prim_path_rgb = self.front_head_rgb_camera_path
        galbot_interface_config.front_head_camera.prim_path_depth = self.front_head_depth_camera_path

        # Initialize interface
        self.interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        self.interface.initialize()

    def _setup_mink(self):
        """Initialize Mink inverse kinematics solver configuration."""
        model = self.simulator.model._model
        self.mink_config = mink.Configuration(model)
        
        # Create IK tasks
        self.tasks = {
            "torso": mink.FrameTask(
                frame_name=self.robot.namespace + "torso_base_link",
                frame_type="body",
                position_cost=1e6,
                orientation_cost=1e6,
            ),
            "posture": mink.PostureTask(model, cost=1.0),
            "chassis": mink.FrameTask(
                frame_name=self.robot.namespace + "omni_chassis_base_link",
                frame_type="body",
                position_cost=1e6,
                orientation_cost=1e6,
            ),
            "left_arm": mink.FrameTask(
                frame_name=self.robot.namespace + "left_gripper_tcp",
                frame_type="site",
                position_cost=50.0,
                orientation_cost=50.0,
                lm_damping=1.0,
            ),
            "right_arm": mink.FrameTask(
                frame_name=self.robot.namespace + "right_gripper_tcp",
                frame_type="site",
                position_cost=50.0,
                orientation_cost=50.0,
                lm_damping=1.0,
            )
        }

        self.velocity_limit = mink.VelocityLimit(
            model, 
            velocities={
                name: 2.0 for name in self.interface.left_arm.joint_names
                + self.interface.right_arm.joint_names
            }
        )
        
        # Solver configuration
        self.solver = "daqp"
        self.damping = 1e-3
        self.rate_limiter = RateLimiter(frequency=1000, warn=False)

    def world_to_robot_frame(self, world_position, world_orientation):
        """Transform pose from world frame to robot base frame.
        
        Args:
            world_position: Position in world frame [x, y, z].
            world_orientation: Orientation in world frame [qx, qy, qz, qw].
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame.
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

    def robot_to_world_frame(self, robot_position, robot_orientation):
        """Transform pose from robot base frame to world frame.
        
        Args:
            robot_position: Position in robot base frame [x, y, z].
            robot_orientation: Orientation in robot base frame [qx, qy, qz, qw].
            
        Returns:
            Tuple of (world_position, world_orientation) in world frame.
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

    def world_to_robot_initial_frame(self, world_position, world_orientation=None):
        """Transform pose from world frame to robot initial frame.
        
        Args:
            world_position: Position in world frame [x, y, z].
            world_orientation: Orientation in world frame [qx, qy, qz, qw] (optional).
            
        Returns:
            Tuple of (robot_initial_position, robot_initial_orientation) in robot initial frame.
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

    def robot_initial_to_world_frame(self, robot_initial_position, robot_initial_orientation=None):
        """Transform pose from robot initial frame to world frame.
        
        Args:
            robot_initial_position: Position in robot initial frame [x, y, z].
            robot_initial_orientation: Orientation in robot initial frame [qx, qy, qz, qw] (optional).
            
        Returns:
            Tuple of (world_position, world_orientation) in world frame.
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

    def camera_to_robot_frame(self, camera_position, camera_orientation):
        """Transform pose from camera frame to robot base frame.
        
        Args:
            camera_position: Position in camera frame [x, y, z].
            camera_orientation: Orientation in camera frame [qx, qy, qz, qw].
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame.
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

    def robot_to_camera_frame(self, robot_position, robot_orientation):
        """Transform pose from robot base frame to camera frame.
        
        Args:
            robot_position: Position in robot base frame [x, y, z].
            robot_orientation: Orientation in robot base frame [qx, qy, qz, qw].
            
        Returns:
            Tuple of (camera_position, camera_orientation) in camera frame.
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

    def world_to_robot_initial_frame_2d(self, world_position_2d):
        """Transform 2D position from world frame to robot initial frame.
        
        Args:
            world_position_2d: Position in world frame [x, y].
            
        Returns:
            robot_initial_position_2d: Position in robot initial frame [x, y].
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

    def compute_inverse_kinematics(self, start_joint_config, target_pose, arm_id="left_arm"):
        """Compute inverse kinematics using Mink solver.
        
        Args:
            start_joint_config: Initial joint configuration (not used in current implementation).
            target_pose: Target pose [x, y, z, qx, qy, qz, qw] in robot base frame.
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            
        Returns:
            Target joint configuration for the specified arm.
            
        Raises:
            ValueError: If arm_id is invalid.
        """
        # Transform target pose from robot frame to world frame for IK
        target_position = target_pose[:3]
        target_orientation = target_pose[3:7]
        world_position, world_orientation = self.robot_to_world_frame(target_position, target_orientation)
        
        # Set target for chassis
        chassis_target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=xyzw_to_wxyz(self.robot.get_orientation())),
            translation=self.robot.get_position()
        )
        self.tasks["chassis"].set_target(chassis_target)

        # Set target for torso
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

        # Set target for posture
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
            raise ValueError(f"Invalid arm_id: {arm_id}")
        
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

        # Check if IK converged
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

    def compute_forward_kinematics(self, arm_id="left_arm"):
        """Compute forward kinematics using ground truth from simulator.
        
        Args:
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            
        Returns:
            TCP pose in base link frame [x, y, z, qx, qy, qz, qw].
            
        Raises:
            ValueError: If arm_id is invalid.
        """
        if arm_id == "left_arm":
            # Get left gripper TCP pose from simulator
            position, quaternion = self.get_left_gripper_pose()
        elif arm_id == "right_arm":
            # Get right gripper TCP pose from simulator
            position, quaternion = self.get_right_gripper_pose()
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}")
        
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

    def get_left_gripper_pose(self):
        """Get left gripper TCP pose in world frame.
        
        Returns:
            Tuple of (position, quaternion) in world frame.
        """
        site_data = self.simulator.data.site(self.robot.namespace + "left_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion
    
    def get_right_gripper_pose(self):
        """Get right gripper TCP pose in world frame.
        
        Returns:
            Tuple of (position, quaternion) in world frame.
        """
        site_data = self.simulator.data.site(self.robot.namespace + "right_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion
    
    def move_arm_to_pose(self, arm_id, target_position, target_orientation):
        """Move arm to target pose using inverse kinematics.
        
        Args:
            arm_id: The ID of the arm, either "left_arm" or "right_arm".
            target_position: Target position in robot base frame [x, y, z].
            target_orientation: Target orientation in robot base frame [qx, qy, qz, qw].
        """
        # Prepare target pose in robot frame
        target_pose = np.concatenate([target_position, target_orientation])
        
        # Solve IK and start motion
        current_joints = self.mink_config.q
        arm_joints = self.compute_inverse_kinematics(current_joints, target_pose, arm_id)
        arm_module = getattr(self.interface, arm_id)
        self._move_joints_to_target(arm_module, arm_joints)

    def move_left_arm_to_pose(self, target_position, target_orientation):
        """Move left arm to target pose.
        
        Args:
            target_position: Target position in robot base frame [x, y, z].
            target_orientation: Target orientation in robot base frame [qx, qy, qz, qw].
        """
        return self.move_arm_to_pose("left_arm", target_position, target_orientation)
    
    def move_right_arm_to_pose(self, target_position, target_orientation):
        """Move right arm to target pose.
        
        Args:
            target_position: Target position in robot base frame [x, y, z].
            target_orientation: Target orientation in robot base frame [qx, qy, qz, qw].
        """
        return self.move_arm_to_pose("right_arm", target_position, target_orientation)

    def move_chassis_follow_path(self, waypoints):
        """Move chassis to follow a path defined by waypoints in world coordinates.
        
        Args:
            waypoints: List of 2D waypoints [(x1, y1), (x2, y2), ...] in world frame.
        """
        if not waypoints or len(waypoints) < 2:
            print("Invalid waypoints: need at least 2 points")
            return
        
        # Initialize path following components
        from physics_simulator.utils.control_utils import BasicPathFollower
        path_follower = BasicPathFollower(velocity=0.8)
        waypoint_tolerance = 0.1
        current_target_index = 0
        path = waypoints

        def follow_path_callback():
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

    def move_chassis_xy(self, waypoints, velocity=0.8):
        """Move chassis to follow waypoints in world coordinates.
        
        Args:
            waypoints: List of 2D waypoints [(x1, y1), (x2, y2), ...] in world frame.
            velocity: Chassis velocity (m/s), default 0.8.
        """
        if not waypoints or len(waypoints) < 1:
            print("Invalid waypoints: need at least 1 point")
            return
        
        # Convert waypoints from world frame to robot initial frame
        waypoints_robot = [self.world_to_robot_initial_frame_2d(wp) for wp in waypoints]
        
        current_waypoint_index = 0
        
        def move_xy_callback():
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

    def move_chassis_rotate(self, target_angle_world, angular_velocity=1.0):
        """Rotate chassis to face a specific angle in world coordinates.
        
        Args:
            target_angle_world: Target angle in world frame (radians), 0 is along positive x-axis.
            angular_velocity: Angular velocity for rotation (rad/s), default 1.0.
        """
        
        # Calculate target heading in robot frame
        from scipy.spatial.transform import Rotation
        initial_rotation = Rotation.from_quat(self.robot_initial_orientation)
        initial_z_angle = initial_rotation.as_euler('xyz')[2]  # Extract z-axis rotation
        target_heading = -initial_z_angle + target_angle_world
        
        def rotate_callback():
            current_heading = self.interface.chassis.get_joint_positions()[2]
            heading_error = target_heading - current_heading
            
            # Normalize error to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            if abs(heading_error) < 0.05:
                self.interface.chassis.set_joint_velocities([0.0, 0.0, 0.0])
                self.simulator.remove_physics_callback("rotate_callback")
            else:
                yaw_vel = angular_velocity if heading_error > 0 else -angular_velocity
                self.interface.chassis.set_joint_velocities([0.0, 0.0, yaw_vel])
        
        self.simulator.add_physics_callback("rotate_callback", rotate_callback)

    def get_camera_images(self):
        """Get RGB and depth images from the front head camera.
        
        Returns:
            Tuple of (rgb_image, depth_image) or (rgb_image, None) if depth not available.
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

    def _move_joints_to_target(self, module, target_positions, steps=500):
        """Move joints from current position to target position smoothly.
        
        Args:
            module: Robot module to control.
            target_positions: Target joint positions.
            steps: Number of interpolation steps.
        """
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _is_joint_positions_reached(self, module, target_positions, atol=0.01):
        """Check if joint positions are reached within tolerance.
        
        Args:
            module: Robot module to check.
            target_positions: Target joint positions.
            atol: Absolute tolerance for position comparison.
            
        Returns:
            True if positions are reached, False otherwise.
        """
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=atol)

    def run(self):
        """Start the simulation loop."""
        self.simulator.loop()

if __name__ == "__main__":
    env = IOAIEnv(headless=False)
    #TODO: Define your callbacks here
    def demo_callback():
        print("demo callback")
    env.simulator.add_physics_callback("demo_callback", demo_callback)

    env.run()