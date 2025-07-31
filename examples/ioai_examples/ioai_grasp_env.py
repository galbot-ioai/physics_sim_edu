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
# Description: Grasp env setup using Galbot
# Author: Chenyu Cao@Galbot
# Date: 2025-05-31
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
    RealsenseD436DepthSensorConfig
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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from physics_simulator.utils.state_machine import SimpleStateMachine

from grasp_reg import GraspRegistration
grasp_reg = GraspRegistration()

@dataclass
class PoseEstimationResult:
    """Data class for pose estimation result"""
    class_name: str
    position: np.ndarray  # [x, y, z] in camera frame
    orientation: np.ndarray  # [qx, qy, qz, qw] in camera frame
    confidence: float
    segmentation_mask: Optional[np.ndarray] = None  # Segmentation mask if available
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2] if available

class PoseEstimationModel:
    """Interface for pose estimation model using YOLO segmentation"""
    
    def __init__(self):
        """Initialize the pose estimation model interface"""
        pass
    
    def estimate_poses(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None) -> List[PoseEstimationResult]:
        """
        Estimate object poses using YOLO segmentation and pose estimation
        
        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (optional)
            
        Returns:
            List of pose estimation results in camera frame
        """
        raise NotImplementedError("Subclass must implement estimate_poses method")

class DummyPoseEstimationModel(PoseEstimationModel):
    """Dummy pose estimation model using ground truth from simulator"""
    
    def __init__(self, simulator, robot):
        super().__init__()
        self.simulator = simulator
        self.robot = robot
        self.supported_objects = ["cube", "bin"]
    
    def estimate_poses(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None) -> List[PoseEstimationResult]:
        """Estimate poses using ground truth from simulator"""
        pose_results = []
        
        for obj_class in self.supported_objects:
            # Get ground truth pose from simulator
            obj_state = self.simulator.get_object_state(f"/World/{obj_class.capitalize()}")
            world_position = obj_state["position"]
            world_orientation = obj_state["orientation"]
            
            # Transform to camera frame
            camera_position, camera_orientation = self._transform_to_camera_frame(
                world_position, world_orientation
            )
            
            # Create pose estimation result
            pose_result = PoseEstimationResult(
                class_name=obj_class,
                position=camera_position,
                orientation=camera_orientation,
                confidence=0.95,  # High confidence for ground truth
                bbox=np.array([100, 100, 200, 200])  # Dummy bbox
            )
            pose_results.append(pose_result)
        
        return pose_results
    
    def _transform_to_camera_frame(self, world_position, world_orientation):
        """Transform pose from world frame to camera frame"""
        from scipy.spatial.transform import Rotation
        
        # Get camera pose in world frame
        camera_prim_path = "/World/Galbot/head_link2/head_end_effector_mount_link/front_head_rgb_camera"
        camera_state = self.simulator.get_sensor_state(camera_prim_path)
        camera_world_position = camera_state["transform_to_base_link"]["position"]
        camera_world_orientation = camera_state["transform_to_base_link"]["orientation"]
        
        # Create transformation matrices
        camera_world_rot = Rotation.from_quat(camera_world_orientation)
        world_rot = Rotation.from_quat(world_orientation)
        
        # Transform position: subtract camera position and rotate
        relative_position = world_position - camera_world_position
        camera_position = camera_world_rot.inv().apply(relative_position)
        
        # Transform orientation: compose rotations
        camera_orientation = (camera_world_rot.inv() * world_rot).as_quat()
        
        return camera_position, camera_orientation

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

class IoaiGraspEnv:
    def __init__(self, headless=False, pose_estimation_model: Optional[PoseEstimationModel] = None):
        """
        Initialize the Olympic environment.
        
        Args:
            headless: Whether to run in headless mode (without visualization)
            pose_estimation_model: Pose estimation model for object detection (optional)
        """
        self.simulator = None
        self.robot = None
        
        # Initialize pose estimation model
        self.pose_estimation_model = pose_estimation_model if pose_estimation_model is not None else None
        
        # Pose estimation related variables
        self.pose_estimation_results = []
        self.last_estimation_time = 0
        self.estimation_interval = 0.1  # Estimation frequency in seconds

        # Setup the simulator
        self._setup_simulator(headless=headless)
        
        # Initialize pose estimation model after simulator setup
        if self.pose_estimation_model is None:
            self.pose_estimation_model = DummyPoseEstimationModel(self.simulator, self.robot)
        
        # Setup the interface
        self._setup_interface()
        self._init_pose()
        # Setup the Mink for solving the inverse kinematics
        self._setup_mink()
        self.state_machine = SimpleStateMachine(max_states=8)
        self.last_state_transition_time = time.time()
        self.state_first_entry = False
        
        # Motion control variables
        self.motion_in_progress = False
        self.target_joint_positions = {}
        self.grasp_start_time = None
        self.pick_pos = None  # Store pick position for move_to_pick_state

    def _setup_simulator(self, headless=False):
        """
        Setup the simulator.
        """
        # Create simulator config
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )
        
        # Initialize the simulator
        self.simulator = PhysicsSimulator(sim_config)

        # Add default scene (default ground plane)
        self.simulator.add_default_scene()

        # Add robot
        robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_foxtrot",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robots")
            .joinpath("galbot_one_foxtrot_description_simplified")
            .joinpath("galbot_one_foxtrot.xml"),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )
        self.simulator.add_robot(robot_config)
        self.robot = self.simulator.get_robot("/World/Galbot")

        # Add front head RGB camera (RealSense D405)
        front_head_rgb_camera_config = RgbCameraConfig(
            name="front_head_rgb_camera",
            prim_path=os.path.join(
                self.robot.prim_path,
                "head_link2",
                "head_end_effector_mount_link",
                "front_head_rgb_camera",
            ),
            translation=[
                0.10084319533055261,
                -0.059042081352783105,
                0.03184978861787491
            ],
            rotation=[
                -0.1654571792421115, 
                0.6935589352367344,
                0.16457378953789606,
                0.6815536611211676
            ],
            camera_axes="ros",
            sensor_config=RealsenseD436RgbSensorConfig(),
            parent_entity_name="galbot_one_foxtrot/head_end_effector_mount_link"
        )
        self.front_head_rgb_camera_path = self.simulator.add_sensor(front_head_rgb_camera_config)

        # Add front head depth camera (RealSense D436)
        front_head_depth_camera_config = DepthCameraConfig(
            name="front_head_depth_camera",
            prim_path=os.path.join(
                self.robot.prim_path,
                "head_link2",
                "head_end_effector_mount_link",
                "front_head_depth_camera",
            ),
            translation=[
                0.10084319533055261,# grasp power_drill
                -0.059042081352783105,
                0.03184978861787491
            ],
            rotation=[
                -0.1654571792421115, 
                0.6935589352367344,
                0.16457378953789606,
                0.6815536611211676
            ],
            camera_axes="ros",
            sensor_config=RealsenseD436DepthSensorConfig(),
            parent_entity_name="galbot_one_foxtrot/head_end_effector_mount_link"
        )
        self.front_head_depth_camera_path = self.simulator.add_sensor(front_head_depth_camera_config)
        
        # Store camera intrinsic parameters for pose estimation
        self.camera_intrinsics = [637.7254326533274, 637.7254326533274, 640.0, 360.0]  # [fx, fy, cx, cy]

        # Add table
        table_config = MeshConfig(
            prim_path="/World/Table",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("table")
            .joinpath("table.xml"),
            position=[0.65, 0, 0],
            orientation=[0, 0, 0.70711, 0.70711],
            # scale=[0.5, 0.7, 0.5]
        )
        self.simulator.add_object(table_config)

        # Add bin
        bin_config = MeshConfig(
            prim_path="/World/Bin",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("bin")
            .joinpath("bin.xml"),
            position=[0.7, -0.1, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(bin_config)

        # Add cube
        cube_config = CuboidConfig(
            prim_path="/World/Cube",
            position=[0.6, 0.2, 0.56],
            orientation=[0, 0, 0, 1],
            scale=[0.05, 0.05, 0.05],
            color=[0.5, 0.5, 0.5],  # Gray color
        )
        self.simulator.add_object(cube_config)

        # Initialize the simulator
        self.simulator.initialize()

        bin_state = self.simulator.get_object_state("/World/Bin")
        self.bin_position = bin_state["position"]
        self.bin_orientation = bin_state["orientation"]


    def _setup_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        galbot_interface_config.modules_manager.enabled_modules.append("chassis")
        galbot_interface_config.modules_manager.enabled_modules.append("left_gripper")
        galbot_interface_config.modules_manager.enabled_modules.append("right_gripper")

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

        # Enable the front camera modules
        galbot_interface_config.modules_manager.enabled_modules.append("front_head_camera")
        galbot_interface_config.front_head_camera.prim_path_rgb = self.front_head_rgb_camera_path
        galbot_interface_config.front_head_camera.prim_path_depth = self.front_head_depth_camera_path

        galbot_interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        galbot_interface.initialize()

        self.interface = galbot_interface

    def _setup_mink(self):
        """
        Initialize Mink IK solver configuration.
        """
        model = self.simulator.model._model
        self.mink_config = mink.Configuration(model)
        
        # Create tasks as dictionary
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
        
        self.solver = "daqp"
        self.damping = 1e-3
        self.rate_limiter = RateLimiter(frequency=1000, warn=False)

    def world_to_robot_frame(self, world_position, world_orientation):
        """Transform pose from world frame to robot base frame.
        
        Args:
            world_position: Position in world frame [x, y, z]
            world_orientation: Orientation in world frame [qx, qy, qz, qw]
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        base_rot = Rotation.from_quat(base_orientation)
        world_rot = Rotation.from_quat(world_orientation)
        
        # Transform position: subtract base position and rotate
        relative_position = world_position - base_position
        robot_position = base_rot.inv().apply(relative_position)
        
        # Transform orientation: compose rotations
        robot_orientation = (base_rot.inv() * world_rot).as_quat()
        
        return robot_position, robot_orientation

    def robot_to_world_frame(self, robot_position, robot_orientation):
        """Transform pose from robot base frame to world frame.
        
        Args:
            robot_position: Position in robot base frame [x, y, z]
            robot_orientation: Orientation in robot base frame [qx, qy, qz, qw]
            
        Returns:
            Tuple of (world_position, world_orientation) in world frame
        """
        from scipy.spatial.transform import Rotation
        
        # Get robot base pose in world frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        # Create transformation matrices
        base_rot = Rotation.from_quat(base_orientation)
        robot_rot = Rotation.from_quat(robot_orientation)
        
        # Transform position: rotate and add base position
        world_position = base_rot.apply(robot_position) + base_position
        
        # Transform orientation: compose rotations
        world_orientation = (base_rot * robot_rot).as_quat()
        
        return world_position, world_orientation

    def camera_to_robot_frame(self, camera_position, camera_orientation):
        """Transform pose from camera frame to robot base frame.
        
        Args:
            camera_position: Position in camera frame [x, y, z]
            camera_orientation: Orientation in camera frame [qx, qy, qz, qw]
            
        Returns:
            Tuple of (robot_position, robot_orientation) in robot base frame
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
        camera_world_rot = Rotation.from_quat(camera_world_orientation)
        camera_local_rot = Rotation.from_quat(camera_orientation)
        base_rot = Rotation.from_quat(base_orientation)
        
        # Transform position: camera frame -> world frame -> robot frame
        world_position = camera_world_rot.apply(camera_position) + camera_world_position
        relative_position = world_position - base_position
        robot_position = base_rot.inv().apply(relative_position)
        
        # Transform orientation: camera frame -> world frame -> robot frame
        world_orientation = (camera_world_rot * camera_local_rot).as_quat()
        robot_orientation = (base_rot.inv() * Rotation.from_quat(world_orientation)).as_quat()
        
        return robot_position, robot_orientation

    def robot_to_camera_frame(self, robot_position, robot_orientation):
        """Transform pose from robot base frame to camera frame.
        
        Args:
            robot_position: Position in robot base frame [x, y, z]
            robot_orientation: Orientation in robot base frame [qx, qy, qz, qw]
                        # orientation=[0, 0, 0.7071, 0.7071]
            Tuple of (camera_position, camera_orientation) in camera frame
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
        camera_world_rot = Rotation.from_quat(camera_world_orientation)
        base_rot = Rotation.from_quat(base_orientation)
        robot_rot = Rotation.from_quat(robot_orientation)
        
        # Transform position: robot frame -> world frame -> camera frame
        world_position = base_rot.apply(robot_position) + base_position
        relative_position = world_position - camera_world_position
        camera_position = camera_world_rot.inv().apply(relative_position)
        
        # Transform orientation: robot frame -> world frame -> camera frame
        world_orientation = (base_rot * robot_rot).as_quat()
        camera_orientation = (camera_world_rot.inv() * Rotation.from_quat(world_orientation)).as_quat()
        
        return camera_position, camera_orientation

    def get_camera_images(self):
        """Get RGB and depth images from the front head camera.
        
        Returns:
            Tuple of (rgb_image, depth_image) or (rgb_image, None) if depth not available
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

    def estimate_object_poses(self) -> List[PoseEstimationResult]:
        """Estimate object poses using pose estimation model"""
        current_time = time.time()
        
        # Check estimation frequency
        if current_time - self.last_estimation_time < self.estimation_interval:
            return self.pose_estimation_results
        
        # Get camera images
        rgb_image, depth_image = self.get_camera_images()
        
        if rgb_image is None:
            return self.pose_estimation_results
        
        # Run pose estimation
        pose_results = self.pose_estimation_model.estimate_poses(rgb_image, depth_image)
        
        # Update estimation results
        self.pose_estimation_results = pose_results
        self.last_estimation_time = current_time
        
        return pose_results

    def get_object_pose_from_estimation(self, target_class: str = "cube") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get object pose from pose estimation in robot base frame"""
        # Estimate object poses
        pose_results = self.estimate_object_poses()

        # Find target object
        target_result = None
        for result in pose_results:
            if result.class_name.lower() == target_class.lower():
                target_result = result
                break
        
        if target_result is None:
            print(f"Target object '{target_class}' not found in pose estimation")
            return None
        
        target_pose = np.concatenate([target_result.position, target_result.orientation])
        grasp_pose = grasp_reg.predict_grasp(target_class, target_pose)["grasp_pose"]

        gripper_position = grasp_pose[:3]
        gripper_orientation = grasp_pose[3:7]

        # Transform from camera frame to robot base frame
        robot_position, robot_orientation = self.camera_to_robot_frame(
            gripper_position, gripper_orientation
        )
        
        return robot_position, robot_orientation

    def compute_simple_ik(self, start_joint, target_pose, arm_id="left_arm"):
        """Compute inverse kinematics using Mink.
        
        Args:
            start_joint: Initial joint configuration (not used in current implementation)
            target_pose: Target pose [x, y, z, qx, qy, qz, qw] in robot base frame
            arm_id: The ID of the arm, either "left_arm" or "right_arm"
            
        Returns:
            Target joint configuration for the specified arm
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
        torso_body_id = mujoco.mj_name2id(self.simulator.model._model, mujoco.mjtObj.mjOBJ_BODY, self.robot.namespace + "torso_base_link")
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
        position_tolerance = 1e-4
        orientation_tolerance = 1e-4
        
        for iteration in range(max_iterations):
            # Solve IK for velocity
            vel = mink.solve_ik(
                self.mink_config,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=[self.velocity_limit] if False else None
            )
            
            # Integrate to update configuration
            self.mink_config.integrate_inplace(vel, dt)
            
            # Check convergence for the specified arm
            error = self.tasks[arm_id].compute_error(self.mink_config)
            pos_error = np.linalg.norm(error[:3])
            ori_error = np.linalg.norm(error[3:])
            if pos_error < position_tolerance and ori_error < orientation_tolerance:
                break

        # Get final joint positions
        joint_positions = self.mink_config.q

        # Extract joint positions for the specified arm
        if arm_id == "left_arm":
            arm_joint_indexes = self.interface.left_arm.joint_indexes
        else:  # right_arm
            arm_joint_indexes = self.interface.right_arm.joint_indexes
        
        arm_joint_positions = joint_positions[arm_joint_indexes]
        return arm_joint_positions

    def compute_simple_fk(self, arm_id="left_arm"):
        """Compute forward kinematics using ground truth from simulator.
        
        Args:
            arm_id: The ID of the arm, either "left_arm" or "right_arm"
            
        Returns:
            TCP pose in robot base frame [x, y, z, qx, qy, qz, qw]
        """
        if arm_id == "left_arm":
            position, quaternion = self.get_left_gripper_pose()
        elif arm_id == "right_arm":
            position, quaternion = self.get_right_gripper_pose()
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}")
        
        # Transform from world frame to robot base frame
        base_position = self.robot.get_position()
        base_orientation = self.robot.get_orientation()
        
        from scipy.spatial.transform import Rotation
        base_rot = Rotation.from_quat(base_orientation)
        tcp_rot = Rotation.from_quat(quaternion)
        
        # Transform position and orientation
        relative_position = position - base_position
        tcp_position_base = base_rot.inv().apply(relative_position)
        tcp_orientation_base = (base_rot.inv() * tcp_rot).as_quat()
        
        return np.concatenate([tcp_position_base, tcp_orientation_base])

    def _init_pose(self):
        # Initialize robot pose
        poses = {
            self.interface.head: [0.0, 0.26],
            self.interface.leg: [0.0821758285164833, 0.6340972781181335,0.5227039456367493, -0.00001198422432935331],
            self.interface.left_arm: [-0.4654513936071508, 1.4785659313201904, -0.6235712173907869, 2.097979784011841, 1.3999720811843872, -0.009971064515411854, 1.0999830961227417],
            self.interface.right_arm: [0.4654513936071508, -1.4785659313201904, 0.6235712173907869, -2.097979784011841, -1.3999720811843872, 0.009971064515411854, -1.0999830961227417]
        }
        
        for module, pose in poses.items():
            module.set_joint_positions(pose, immediate=True)

    def _move_joints_to_target(self, module, target_positions, steps=500):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _is_joint_positions_reached(self, module, target_positions, atol=0.01):
        """Check if joint positions are reached within tolerance."""
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=atol)
    
    def _is_arm_motion_complete(self, atol=0.01):
        """Check if arm has reached its target position."""
        for module_name, target_positions in self.target_joint_positions.items():
            module = getattr(self.interface, module_name)
            if not self._is_joint_positions_reached(module, target_positions, atol):
                return False
        return True
    
    def _move_arm_to_pose(self, target_position, target_orientation, arm_id="left_arm"):
        """Move arm to target pose with IK solving and motion control.
        
        Args:
            target_position: Target position [x, y, z] in robot base frame
            target_orientation: Target orientation [qx, qy, qz, qw] in robot base frame
            arm_id: The ID of the arm, either "left_arm" or "right_arm"
            
        Returns:
            True if motion is complete, False otherwise
        """
        if not self.motion_in_progress:
            # Prepare target pose in robot frame
            target_pose = np.concatenate([target_position, target_orientation])
            
            # Solve IK and start motion
            current_joints = self.mink_config.q
            arm_joints = self.compute_simple_ik(current_joints, target_pose, arm_id)
            arm_module = getattr(self.interface, arm_id)
            self._move_joints_to_target(arm_module, arm_joints)
            
            # Store target positions for completion check
            self.target_joint_positions = {arm_id: arm_joints}
            self.motion_in_progress = True
        
        # Check if motion is complete
        if self._is_arm_motion_complete():
            self.motion_in_progress = False
            return True
        return False
    
    def _move_left_arm_to_pose(self, target_position, target_orientation):
        """Move left arm to target pose"""
        return self._move_arm_to_pose(target_position, target_orientation, "left_arm")
    
    def _move_right_arm_to_pose(self, target_position, target_orientation):
        """Move right arm to target pose"""
        return self._move_arm_to_pose(target_position, target_orientation, "right_arm")

    def get_left_gripper_pose(self):
        """Get left gripper TCP pose in world frame"""
        site_data = self.simulator.data.site(self.robot.namespace + "left_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion
    
    def get_right_gripper_pose(self):
        """Get right gripper TCP pose in world frame"""
        site_data = self.simulator.data.site(self.robot.namespace + "right_gripper_tcp")
        position = site_data.xpos
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_matrix(site_data.xmat.reshape((3, 3))).as_quat()
        return position, quaternion

    def pick_and_place_callback(self):
        """Callback function for pick and place task using state machine"""

        def init_state():
            """Move to initial pose"""
            robot_pos = np.array([0.5, 0.3, 0.7])
            robot_ori = np.array([0, 0.7071, 0, 0.7071])
            return self._move_left_arm_to_pose(robot_pos, robot_ori)
        
        def move_to_pre_pick_state():
            """Move to pre-pick position"""
            # Get cube position from estimation or cached value
            if self.state_first_entry:
                # Estimate object position on first entry
                pose_result = self.get_object_pose_from_estimation("cube")
                if pose_result is None:
                    print("Failed to estimate cube pose, cannot move to pick state.")
                    return False
                robot_pos, robot_ori = pose_result
                # Cache for subsequent use
                self.cube_position = robot_pos.copy()
                self.cube_orientation = robot_ori.copy()
                self.state_first_entry = False
            else:
                # Use cached cube position
                robot_pos = getattr(self, "cube_position", None)
                robot_ori = getattr(self, "cube_orientation", None)
                if robot_pos is None or robot_ori is None:
                    print("Cube position/orientation not set, cannot move to pick state.")
                    return False

            # Move to pre-pick position (offset above cube)
            pre_pick_pos = robot_pos + np.array([0, 0, 0.2])
            return self._move_left_arm_to_pose(pre_pick_pos, robot_ori)
        

        def move_to_pick_state():
            """Move to pick position"""
            # Get cube position from estimation or cached value
            if self.state_first_entry:
                # Estimate object position on first entry
                pose_result = self.get_object_pose_from_estimation("cube")
                if pose_result is None:
                    print("Failed to estimate cube pose, cannot move to pick state.")
                    return False
                robot_pos, robot_ori = pose_result
                # Cache for subsequent use
                self.cube_position = robot_pos.copy()
                self.cube_orientation = robot_ori.copy()
                self.state_first_entry = False
            else:
                # Use cached cube position
                robot_pos = getattr(self, "cube_position", None)
                robot_ori = getattr(self, "cube_orientation", None)
                if robot_pos is None or robot_ori is None:
                    print("Cube position/orientation not set, cannot move to pick state.")
                    return False

            # Move to pick position
            return self._move_left_arm_to_pose(robot_pos, robot_ori)
        
        def grasp_state():
            """Grasp the object"""
            if self.state_first_entry:
                self.interface.left_gripper.set_gripper_close()
                self.grasp_start_time = time.time()
                self.state_first_entry = False
            
            # Stay in grasp state for 2 seconds
            if time.time() - self.grasp_start_time >= 2.0:
                return True
            return False

        def move_to_pre_place_state():
            """Move to pre-place position"""
            # Move to pre-place position relative to cube
            pre_place_pos = self.cube_position + np.array([-0.1, 0, 0.4])
            pre_place_ori = np.array([0, 0.7071, 0, 0.7071])
            return self._move_left_arm_to_pose(pre_place_pos, pre_place_ori)

        def move_to_place_state():
            """Move to place position"""
            if self.state_first_entry:
                # Use pose estimation to get bin pose
                pose_result = self.get_object_pose_from_estimation("bin")
                if pose_result is None:
                    print("Failed to estimate bin pose, cannot move to place state.")
                    return False
                robot_pos, robot_ori = pose_result
                self.bin_position = robot_pos.copy()
                self.bin_orientation = robot_ori.copy()
                print(f"Pose estimation detected bin at position: {robot_pos}")
                self.state_first_entry = False

            # Move to place position above bin
            place_pos = self.bin_position + np.array([0, 0, 0.3])
            place_ori = np.array([0, 0.7071, 0, 0.7071])  # Fixed orientation for placing
            return self._move_left_arm_to_pose(place_pos, place_ori)
        
        def release_state():
            """Release the object"""
            if self.state_first_entry:
                self.interface.left_gripper.set_gripper_open()
                self.grasp_start_time = time.time()
                self.state_first_entry = False
            
            # Stay in release state for 1 second
            if time.time() - self.grasp_start_time >= 1.0:
                return True
            return False
        
        def return_to_init_state():
            """Return to initial pose"""
            return init_state()

        # Add states to state machine
        self.state_machine.add_state(0, "Init", init_state)
        self.state_machine.add_state(1, "MoveToPrePick", move_to_pre_pick_state)
        self.state_machine.add_state(2, "MoveToPick", move_to_pick_state)
        self.state_machine.add_state(3, "Grasp", grasp_state)
        self.state_machine.add_state(4, "MoveToPrePlace", move_to_pre_place_state)
        self.state_machine.add_state(5, "MoveToPlace", move_to_place_state)
        self.state_machine.add_state(6, "Release", release_state)
        self.state_machine.add_state(7, "ReturnToInit", return_to_init_state)

        # Execute current state
        if self.state_machine.trigger():
            self.state_first_entry = True
            self.motion_in_progress = False
            print(f"Current state: {self.state_machine.get_state_name()}")
        
        # Execute current state and move to next when complete
        if self.state_machine.execute_current_state():
            # Check if we can move to next state
            if not self.state_machine.next():
                # Task completed, reset state machine for next cycle
                print("Pick and place task completed!")
                self.state_machine.reset()
            self.state_first_entry = True

if __name__ == "__main__":
    env = IoaiGraspEnv(headless=False)
    env.simulator.add_physics_callback("pick_and_place", env.pick_and_place_callback)
    env.simulator.loop()
    env.simulator.close()
