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
    CuboidConfig
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
import mink
from loop_rate_limiters import RateLimiter
from auro_utils import xyzw_to_wxyz, wxyz_to_xyzw
from pathlib import Path
import numpy as np
from physics_simulator.utils.data_types import JointTrajectory
import time

from physics_simulator.utils.state_machine import SimpleStateMachine

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

class IoaiGraspEnv:
    def __init__(self, headless=False):
        """
        Initialize the Olympic environment.
        
        Args:
            headless: Whether to run in headless mode (without visualization)
        """
        self.simulator = None
        self.robot = None

        # Setup the simulator
        self._setup_simulator(headless=headless)
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

        # Add bucket
        bucket_config = MeshConfig(
            prim_path="/World/bucket",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("bucket")
            .joinpath("bucket.xml"),
            position=[0.7, -0.1, 0.55],
            orientation=[0, 0, 0.70711, 0.70711],
        )
        self.simulator.add_object(bucket_config)

        # Add cube
        cube_config = CuboidConfig(
            prim_path="/World/Cube",
            position=[0.65, 0.2, 0.56],
            orientation=[0, 0, 0, 1],
            scale=[0.05, 0.05, 0.05],
            color=[0, 1, 0],
        )
        self.simulator.add_object(cube_config)

        # Initialize the simulator
        self.simulator.initialize()

        bucket_state = self.simulator.get_object_state("/World/bucket")
        self.bucket_position = bucket_state["position"]
        self.bucket_orientation = bucket_state["orientation"]


    def _setup_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        # galbot_interface_config.modules_manager.enabled_modules.append("chassis")
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

        # galbot_interface_config.chassis.joint_names = [
        #     f"{robot_name}/mobile_forward_joint",
        #     f"{robot_name}/mobile_side_joint",
        #     f"{robot_name}/mobile_yaw_joint",
        # ]

        galbot_interface_config.left_gripper.joint_names = [
            f"{robot_name}/left_gripper_r_knuckle_joint",
        ]

        galbot_interface_config.right_gripper.joint_names = [
            f"{robot_name}/right_gripper_r_knuckle_joint",
        ]

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

    def solve_ik(self,
                 left_target_position=None,
                 left_target_orientation=None,
                 right_target_position=None,
                 right_target_orientation=None,
                 limit_velocity=False
                 ):
        """
        Solve IK for specified arm(s) and return final joint positions
        
        Args:
            left_target_position: Target position for left arm [x, y, z]
            left_target_orientation: Target orientation for left arm as quaternion [x, y, z, w]
            right_target_position: Target position for right arm [x, y, z]
            right_target_orientation: Target orientation for right arm as quaternion [x, y, z, w]
            limit_velocity: Whether to apply velocity limits
            
        Returns:
            Dictionary containing final joint positions for each module
        """
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

        # Set targets for left and right arm
        if left_target_position is not None and left_target_orientation is not None:
            target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=xyzw_to_wxyz(left_target_orientation)),
                translation=left_target_position
            )
            self.tasks["left_arm"].set_target(target)
        if right_target_position is not None and right_target_orientation is not None:
            target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=xyzw_to_wxyz(right_target_orientation)),
                translation=right_target_position
            )
            self.tasks["right_arm"].set_target(target)

        # Collect all tasks
        tasks = [self.tasks["torso"], self.tasks["posture"], self.tasks["chassis"]]
        if left_target_position is not None and left_target_orientation is not None:
            tasks.append(self.tasks["left_arm"])
        if right_target_position is not None and right_target_orientation is not None:
            tasks.append(self.tasks["right_arm"])
        
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
                limits=[self.velocity_limit] if limit_velocity else None
            )
            
            # Integrate to update configuration
            self.mink_config.integrate_inplace(vel, dt)
            
            # Check convergence for left arm
            if left_target_position is not None and left_target_orientation is not None:
                error = self.tasks["left_arm"].compute_error(self.mink_config)
                pos_error = np.linalg.norm(error[:3])
                ori_error = np.linalg.norm(error[3:])
                if pos_error < position_tolerance and ori_error < orientation_tolerance:
                    break
            
            # Check convergence for right arm
            if right_target_position is not None and right_target_orientation is not None:
                error = self.tasks["right_arm"].compute_error(self.mink_config)
                pos_error = np.linalg.norm(error[:3])
                ori_error = np.linalg.norm(error[3:])
                if pos_error < position_tolerance and ori_error < orientation_tolerance:
                    break

        # Get final joint positions
        joint_positions = self.mink_config.q

        # Extract joint positions for each module
        left_arm_joint_indexes = self.interface.left_arm.joint_indexes
        left_arm_joint_positions = joint_positions[left_arm_joint_indexes]
        right_arm_joint_indexes = self.interface.right_arm.joint_indexes
        right_arm_joint_positions = joint_positions[right_arm_joint_indexes]
        head_joint_indexes = self.interface.head.joint_indexes
        head_joint_positions = joint_positions[head_joint_indexes]
        leg_joint_indexes = self.interface.leg.joint_indexes
        leg_joint_positions = joint_positions[leg_joint_indexes]

        return {
            "left_arm": left_arm_joint_positions,
            "right_arm": right_arm_joint_positions,
            "head": head_joint_positions,
            "leg": leg_joint_positions
        }

    def _init_pose(self):
        # Initialize robot pose
        poses = {
            self.interface.head: [0.0, 0.0],
            self.interface.leg: [0.2, 0.756, 0.53, 0.0],
            self.interface.left_arm: [-0.4654513936071508, 1.4785659313201904, -0.6235712173907869, 2.097979784011841, 1.3999720811843872, -0.009971064515411854, 1.0999830961227417],
            self.interface.right_arm: [0.4654513936071508, -1.4785659313201904, 0.6235712173907869, -2.097979784011841, -1.3999720811843872, 0.009971064515411854, -1.0999830961227417]
        }
        
        for module, pose in poses.items():
            module.set_joint_positions(pose, immediate=True)

    def _move_joints_to_target(self, module, target_positions, steps=1000):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _is_joint_positions_reached(self, module, target_positions, atol=0.01):
        """Check if joint positions are reached within tolerance."""
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=atol)
    
    def _is_arms_motion_complete(self, atol=0.01):
        """Check if both arms have reached their target positions."""
        for module_name, target_positions in self.target_joint_positions.items():
            module = getattr(self.interface, module_name)
            if not self._is_joint_positions_reached(module, target_positions, atol):
                return False
        return True

    def get_left_gripper_pose(self):
        tmat = np.eye(4)
        tmat[:3,:3] = self.simulator.data.site(self.robot.namespace + "left_gripper_tcp").xmat.reshape((3,3))
        tmat[:3,3] = self.simulator.data.site(self.robot.namespace + "left_gripper_tcp").xpos
        
        # Extract position
        position = tmat[:3, 3]
        
        # Extract orientation as quaternion (x, y, z, w)
        from scipy.spatial.transform import Rotation
        rotation_matrix = tmat[:3, :3]
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        
        return position, quaternion
    
    def get_right_gripper_pose(self):
        tmat = np.eye(4)
        tmat[:3,:3] = self.simulator.data.site(self.robot.namespace + "right_gripper_tcp").xmat.reshape((3,3))
        tmat[:3,3] = self.simulator.data.site(self.robot.namespace + "right_gripper_tcp").xpos
        
        # Extract position
        position = tmat[:3, 3]

        # Extract orientation as quaternion (x, y, z, w)
        from scipy.spatial.transform import Rotation
        rotation_matrix = tmat[:3, :3]
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        
        return position, quaternion

    def pick_and_place_callback(self):
        """Callback function for pick and place task using state machine"""

        def init_state():
            """Move to initial pose"""
            if not self.motion_in_progress:
                joint_positions = self.solve_ik(
                    left_target_position=np.array([0.5, 0.3, 0.7]),
                    left_target_orientation=np.array([0, 0.7071, 0, 0.7071]),
                    right_target_position=np.array([0.5, -0.3, 0.7]),
                    right_target_orientation=np.array([0, 0.7071, 0, 0.7071])
                )
                
                # Start motion for arms only
                self._move_joints_to_target(self.interface.left_arm, joint_positions["left_arm"])
                self._move_joints_to_target(self.interface.right_arm, joint_positions["right_arm"])
                
                # Store target positions for completion check
                self.target_joint_positions = {
                    "left_arm": joint_positions["left_arm"],
                    "right_arm": joint_positions["right_arm"]
                }
                self.motion_in_progress = True
            
            # Check if motion is complete
            if self._is_arms_motion_complete():
                self.motion_in_progress = False
                return True
            return False
        
        def move_to_pre_pick_state():
            """Move to pre-pick position"""
            if not self.motion_in_progress:
                if self.state_first_entry:
                    cube_state = self.simulator.get_object_state("/World/Cube")
                    self.cube_position = cube_state["position"].copy()
                    self.state_first_entry = False
                    
                joint_positions = self.solve_ik(
                    left_target_position=self.cube_position + np.array([0, 0, 0.15]),
                    left_target_orientation=np.array([0, 0.7071, 0, 0.7071]),
                    right_target_position=np.array([0.5, -0.3, 0.7]),
                    right_target_orientation=np.array([0, 0.7071, 0, 0.7071])
                )
                
                self._move_joints_to_target(self.interface.left_arm, joint_positions["left_arm"])
                self._move_joints_to_target(self.interface.right_arm, joint_positions["right_arm"])
                
                self.target_joint_positions = {
                    "left_arm": joint_positions["left_arm"],
                    "right_arm": joint_positions["right_arm"]
                }
                self.motion_in_progress = True
            
            if self._is_arms_motion_complete():
                self.motion_in_progress = False
                return True
            return False

        def move_to_pick_state():
            """Move to pick position"""
            if not self.motion_in_progress:
                joint_positions = self.solve_ik(
                    left_target_position=self.cube_position + np.array([0, 0, 0.03]),
                    left_target_orientation=np.array([0, 0.7071, 0, 0.7071]),
                    right_target_position=np.array([0.5, -0.3, 0.7]),
                    right_target_orientation=np.array([0, 0.7071, 0, 0.7071])
                )
                
                self._move_joints_to_target(self.interface.left_arm, joint_positions["left_arm"])
                self._move_joints_to_target(self.interface.right_arm, joint_positions["right_arm"])
                
                self.target_joint_positions = {
                    "left_arm": joint_positions["left_arm"],
                    "right_arm": joint_positions["right_arm"]
                }
                self.motion_in_progress = True
            
            if self._is_arms_motion_complete():
                self.motion_in_progress = False
                return True
            return False
        
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
            if not self.motion_in_progress:
                joint_positions = self.solve_ik(
                    left_target_position=self.cube_position + np.array([-0.1, 0, 0.4]),
                    left_target_orientation=np.array([0, 0.7071, 0, 0.7071]),
                    right_target_position=np.array([0.5, -0.3, 0.7]),
                    right_target_orientation=np.array([0, 0.7071, 0, 0.7071])
                )
                
                self._move_joints_to_target(self.interface.left_arm, joint_positions["left_arm"])
                self._move_joints_to_target(self.interface.right_arm, joint_positions["right_arm"])
                
                self.target_joint_positions = {
                    "left_arm": joint_positions["left_arm"],
                    "right_arm": joint_positions["right_arm"]
                }
                self.motion_in_progress = True
            
            if self._is_arms_motion_complete():
                self.motion_in_progress = False
                return True
            return False

        def move_to_place_state():
            """Move to place position"""
            if not self.motion_in_progress:
                if self.state_first_entry:
                    bucket_state = self.simulator.get_object_state("/World/bucket")
                    self.bucket_position = bucket_state["position"].copy()
                    self.state_first_entry = False

                joint_positions = self.solve_ik(
                    left_target_position=self.bucket_position + np.array([0, 0, 0.3]),
                    left_target_orientation=np.array([0, 0.7071, 0, 0.7071]),
                    right_target_position=np.array([0.5, -0.3, 0.7]),
                    right_target_orientation=np.array([0, 0.7071, 0, 0.7071])
                )
                
                self._move_joints_to_target(self.interface.left_arm, joint_positions["left_arm"])
                self._move_joints_to_target(self.interface.right_arm, joint_positions["right_arm"])
                
                self.target_joint_positions = {
                    "left_arm": joint_positions["left_arm"],
                    "right_arm": joint_positions["right_arm"]
                }
                self.motion_in_progress = True
            
            if self._is_arms_motion_complete():
                self.motion_in_progress = False
                return True
            return False
        
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
            self.state_machine.next()
            self.state_first_entry = True

if __name__ == "__main__":
    env = IoaiGraspEnv(headless=False)
    env.simulator.add_physics_callback("pick_and_place", env.pick_and_place_callback)
    env.simulator.loop()
    env.simulator.close()
