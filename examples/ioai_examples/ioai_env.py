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
# Description: IOAI env
# Author: Chenyu Cao@Galbot
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

from physics_simulator.utils.state_machine import SimpleStateMachine

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

class IOAIEnv:
    def __init__(self, headless=False):
        """
        Init the IOAI env
        """
        self.simulator = None
        self.robot = None

        # Setup the simulator
        self._setup_simulator(headless=headless)

        # Setup the interface
        self._setup_interface()

        # Init pose
        self._init_pose()

        # Setup the mink
        self._setup_mink()

    def _setup_simulator(self, headless=False):
        """
        Setup the simulator
        """
        # Create simulator config
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )

        # Initilize the simulator
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
            position=[0, -0.2, 0],
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
            sensor_config=RealsenseD436DepthSensorConfig(),
            parent_entity_name="galbot_one_foxtrot/head_end_effector_mount_link"
        )
        self.front_head_depth_camera_path = self.simulator.add_sensor(front_head_depth_camera_config)

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
        )
        self.simulator.add_object(table_config)

        # Add bin
        bin_config = MeshConfig(
            prim_path="/World/bin",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("bin")
            .joinpath("bin.xml"),
            position=[0.65, 0.3, 0.55],
            orientation=[0, 0, 0, 1],
        )
        self.simulator.add_object(bin_config)

        # Add cube
        cube_config = CuboidConfig(
            prim_path="/World/Cube",
            position=[0.5, -0.3, 0.56],
            orientation=[0, 0, 0, 1],
            scale=[0.05, 0.05, 0.05],
            color=[0.5, 0.5, 0.5],  # Gray color
        )
        self.simulator.add_object(cube_config)

        # Add toy
        toy_config = MeshConfig(
            prim_path="/World/toy",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("toy")
            .joinpath("toy.xml"),
            position=[0.65, -0.3, 0.55],
            orientation=[0, 0, 0, -1],
        )
        self.simulator.add_object(toy_config)

        # Add extrusion
        extrusion_config = MeshConfig(
            prim_path="/World/Extrusion",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("extrusion")
            .joinpath("extrusion.xml"),
            position=[0.6, -0.2, 0.55],
            orientation=[0, 0, 0, 1],
        )   
        self.simulator.add_object(extrusion_config)

        # Add power drill
        power_drill_config = MeshConfig(
            prim_path="/World/PowerDrill",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("power_drill")
            .joinpath("power_drill.xml"),
            position=[0.65, -0.1, 0.55],
            orientation=[0, 0, 0, 1],
        )
        self.simulator.add_object(power_drill_config)

        # Add mug
        mug_config = MeshConfig(
            prim_path="/World/Mug",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("mug")
            .joinpath("mug.xml"),
            position=[0.5, 0, 0.55],
            orientation=[0, 0, 0, 1],
        )
        self.simulator.add_object(mug_config)

        center_x = 2
        center_y = 2
        wall_width = 5.5
        wall_height = 0.5
        wall_depth = 0.05

        # Add walls
        wall_1_config = CuboidConfig(
            prim_path="/World/Wall1",
            position=[center_x, center_y+wall_width/2, wall_height / 2],
            orientation=[0, 0, 0, 1],
            scale=[wall_width, wall_depth, wall_height],
            color=[0.6, 0.8, 1.0],  # Light blue color
            interaction_type="static"
        )
        self.simulator.add_object(wall_1_config)

        wall_2_config = CuboidConfig(
            prim_path="/World/Wall2",
            position=[center_x, center_y-wall_width/2, wall_height / 2],
            orientation=[0, 0, 0, 1],
            scale=[wall_width, wall_depth, wall_height],
            color=[0.6, 0.8, 1.0],  # Light blue color
            interaction_type="static"
        )
        self.simulator.add_object(wall_2_config)

        wall_3_config = CuboidConfig(
            prim_path="/World/Wall3",
            position=[center_x+wall_width/2, center_y, wall_height / 2],
            orientation=[0, 0, 0, 1],
            scale=[wall_depth, wall_width, wall_height],
            color=[0.6, 0.8, 1.0],  # Light blue color
            interaction_type="static"
        )
        self.simulator.add_object(wall_3_config)

        wall_4_config = CuboidConfig(
            prim_path="/World/Wall4",
            position=[center_x-wall_width/2, center_y, wall_height / 2],
            orientation=[0, 0, 0, 1],
            scale=[wall_depth, wall_width, wall_height],
            color=[0.6, 0.8, 1.0],  # Light blue color
            interaction_type="static"
        )
        self.simulator.add_object(wall_4_config)

        # Add shelf
        shelf_config = MeshConfig(
            prim_path="/World/Shelf",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("shelf")
            .joinpath("shelf.xml"),
            position=[4, 4, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(shelf_config)

        # Add cones
        cone_1_config = MeshConfig(
            prim_path="/World/Cone1",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("cone")
            .joinpath("cone.xml"),
            position=[3, 3, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(cone_1_config)

        cone_2_config = MeshConfig(
            prim_path="/World/Cone2",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("cone")
            .joinpath("cone.xml"),
            position=[2, 2, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(cone_2_config)

        cone_3_config = MeshConfig(
            prim_path="/World/Cone3",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("cone")
            .joinpath("cone.xml"),
            position=[1, 1, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(cone_3_config)

        # Initialize the simulator
        self.simulator.initialize()

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
            TCP pose in base link frame [x, y, z, qx, qy, qz, qw]
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
        base_rot = Rotation.from_quat(base_orientation)
        base_rot_matrix = base_rot.as_matrix()
        
        # TCP in world frame
        tcp_rot = Rotation.from_quat(quaternion)
        tcp_rot_matrix = tcp_rot.as_matrix()
        
        # Transform position: subtract base position and rotate
        relative_position = position - base_position
        tcp_position_base = base_rot.inv().apply(relative_position)
        
        # Transform orientation: compose rotations
        tcp_orientation_base = (base_rot.inv() * tcp_rot).as_quat()
        
        # Return pose in base link frame [x, y, z, qx, qy, qz, qw]
        return np.concatenate([tcp_position_base, tcp_orientation_base])

    def _init_pose(self):
        # Initialize robot pose
        poses = {
            self.interface.head: [0.0, 0.26],
            self.interface.leg: [0.0821758285164833, 0.6340972781181335,0.5227039456367493, -0.00001198422432935331],
            self.interface.left_arm: [2.0020599365234375,-1.5977126359939575,-0.5948255658149719,-1.694089651107788,-0.0002879792882595211,-0.7909831404685974,-0.00016755158139858395],
            self.interface.right_arm: [-2.001628875732422,1.6029852628707886,0.6024474501609802,1.6955766677856445,-0.0002391100861132145,0.7967827916145325,-0.00014311698032543063]
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
    
    def _is_left_arm_motion_complete(self, atol=0.01):
        """Check if left arm has reached its target position."""
        for module_name, target_positions in self.target_joint_positions.items():
            module = getattr(self.interface, module_name)
            if not self._is_joint_positions_reached(module, target_positions, atol):
                return False
        return True
    
    def _is_right_arm_motion_complete(self, atol=0.01):
        """Check if right arm has reached its target position."""
        for module_name, target_positions in self.target_joint_positions.items():
            module = getattr(self.interface, module_name)
            if not self._is_joint_positions_reached(module, target_positions, atol):
                return False
        return True
    
    def _move_left_arm_to_pose(self, target_position, target_orientation):
        """Move left arm to target pose with IK solving and motion control.
        
        Args:
            target_position: Target position [x, y, z] in robot base frame
            target_orientation: Target orientation [qx, qy, qz, qw] in robot base frame
            
        Returns:
            True if motion is complete, False otherwise
        """
        if not self.motion_in_progress:
            # Prepare target pose in robot frame
            target_pose = np.concatenate([target_position, target_orientation])
            
            # Solve IK and start motion
            current_joints = self.mink_config.q
            left_arm_joints = self.compute_simple_ik(current_joints, target_pose, "left_arm")
            self._move_joints_to_target(self.interface.left_arm, left_arm_joints)
            
            # Store target positions for completion check
            self.target_joint_positions = {"left_arm": left_arm_joints}
            self.motion_in_progress = True
        
        # Check if motion is complete
        if self._is_left_arm_motion_complete():
            self.motion_in_progress = False
            return True
        return False
    
    def _move_right_arm_to_pose(self, target_position, target_orientation):
        """Move right arm to target pose with IK solving and motion control.
        
        Args:
            target_position: Target position [x, y, z] in robot base frame
            target_orientation: Target orientation [qx, qy, qz, qw] in robot base frame
            
        Returns:
            True if motion is complete, False otherwise
        """
        if not self.motion_in_progress:
            # Prepare target pose in robot frame
            target_pose = np.concatenate([target_position, target_orientation])
            
            # Solve IK and start motion
            current_joints = self.mink_config.q
            right_arm_joints = self.compute_simple_ik(current_joints, target_pose, "right_arm")
            self._move_joints_to_target(self.interface.right_arm, right_arm_joints)
            
            # Store target positions for completion check
            self.target_joint_positions = {"right_arm": right_arm_joints}
            self.motion_in_progress = True
        
        # Check if motion is complete
        if self._is_right_arm_motion_complete():
            self.motion_in_progress = False
            return True
        return False


    def run(self):
        self.simulator.loop()

if __name__ == "__main__":
    env = IOAIEnv(headless=False)
    #TODO: Define your callbacks here
    def demo_callback():
        print("demo callback")
    env.simulator.add_physics_callback("demo_callback", demo_callback)

    env.run()