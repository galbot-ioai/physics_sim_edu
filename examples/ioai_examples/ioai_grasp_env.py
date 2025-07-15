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
        # Setup the Mink for solving the inverse kinematics
        self._setup_mink()
        self.state_machine = SimpleStateMachine(max_states=8)
        self.last_state_transition_time = time.time()
        self.state_first_entry = False

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
            .joinpath("galbot_one_foxtrot_description")
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
            orientation=[0, 0, 0.70711, -0.70711],
            scale=[0.5, 0.7, 0.5]
        )
        self.simulator.add_object(table_config)

        # Add closet
        closet_config = MeshConfig(
            prim_path="/World/Closet",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("closet")
            .joinpath("closet.xml"),
            position=[0.65, -0.1, 0.55],
            orientation=[0, 0, 0.70711, -0.70711],
            scale=[0.2, 0.2, 0.2]
        )
        self.simulator.add_object(closet_config)

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

        closet_state = self.simulator.get_object_state("/World/Closet")
        self.closet_position = closet_state["position"]
        self.closet_orientation = closet_state["orientation"]


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
        
        # Create tasks
        self.tasks = [
            mink.FrameTask(
                frame_name=self.robot.namespace + "torso_base_link",
                frame_type="body",
                position_cost=0.0,
                orientation_cost=10.0,
            ),
            mink.PostureTask(model, cost=1.0),
            mink.FrameTask(
                frame_name=self.robot.namespace + "omni_chassis_base_link",
                frame_type="body",
                position_cost=100.0,
                orientation_cost=100.0,
            ),
        ]
        
        # Create arm tasks
        self.arm_tasks = {
            "left": mink.FrameTask(
                frame_name=self.robot.namespace + "left_gripper_tcp",
                frame_type="site",
                position_cost=50.0,
                orientation_cost=50.0,
                lm_damping=1.0,
            ),
            "right": mink.FrameTask(
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
        self.rate_limiter = RateLimiter(frequency=1000, warn=False)
        
        for task in self.tasks:
            task.set_target_from_configuration(self.mink_config)

    def solve_ik(self,
                 left_target_position=None,
                 left_target_orientation=None,
                 right_target_position=None,
                 right_target_orientation=None,
                 limit_velocity=False
                 ):
        """
        Solve IK for specified arm(s)
        
        Args:
            left_target_position: Target position for left arm [x, y, z]
            left_target_orientation: Target orientation for left arm as quaternion [x, y, z, w]
            right_target_position: Target position for right arm [x, y, z]
            right_target_orientation: Target orientation for right arm as quaternion [x, y, z, w]
        """
        active_tasks = self.tasks.copy()
        
        if left_target_position is not None:
            if left_target_orientation is not None:
                target = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=xyzw_to_wxyz(left_target_orientation)),
                    translation=left_target_position
                )
                self.arm_tasks["left"].set_target(target)
                active_tasks.append(self.arm_tasks["left"])
            
        if right_target_position is not None:
            if right_target_orientation is not None:
                target = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=xyzw_to_wxyz(right_target_orientation)),
                    translation=right_target_position
                )
                self.arm_tasks["right"].set_target(target)
                active_tasks.append(self.arm_tasks["right"])
            
        # Solve IK
        vel = mink.solve_ik(
            self.mink_config,
            active_tasks,
            self.rate_limiter.dt,
            self.solver,
            0.01,
            limits=[self.velocity_limit] if limit_velocity else None
        )

        self.mink_config.integrate_inplace(vel, self.rate_limiter.dt * 0.1)
         
        # Update robot joint positions
        joint_positions = self.mink_config.q

        # Use joint_positions to update leg, head, left_arm, and right_arm
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
        # Init head pose
        head = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, head)

        # Init leg pose
        leg = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, leg)

        # Init left arm pose
        left_arm = [
            -0.716656506061554,
            -1.538102626800537,
            -0.03163932263851166,
            -1.379408597946167,
            -1.4995604753494263,
            0.0332450270652771,
            -1.0637063884735107
        ]
        self._move_joints_to_target(self.interface.left_arm, left_arm)

        # Init right arm pose
        right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417
        ]
        self._move_joints_to_target(self.interface.right_arm, right_arm)

    def _move_joints_to_target(self, module, target_positions, steps=100):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _is_joint_positions_reached(self, module, target_positions, atol=0.01):
        """Check if joint positions are reached within tolerance."""
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=atol)

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
        """
        Callback function for pick and place task using state machine
        
        Args:
            env: IoaiGraspEnv instance
        """

        def init_state():
            """Ready to pick"""
            left_target_position = np.array([0.4, 0.3, 0.8])
            left_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            right_target_position = np.array([0.4, -0.3, 0.8])
            right_target_orientation = np.array([0, 0.7071, 0, 0.7071])

            # Solve IK for left arm
            joint_positions = self.solve_ik(
                left_target_position=left_target_position,
                left_target_orientation=left_target_orientation,
                right_target_position=right_target_position,
                right_target_orientation=right_target_orientation
            )

            # Set joints to target positions
            self.interface.left_arm.set_joint_positions(joint_positions["left_arm"])
            self.interface.right_arm.set_joint_positions(joint_positions["right_arm"])
            self.interface.head.set_joint_positions(joint_positions["head"])
            self.interface.leg.set_joint_positions(joint_positions["leg"])

            left_gripper_pose = self.get_left_gripper_pose()
            right_gripper_pose = self.get_right_gripper_pose()

            # Check if left gripper is close to cube
            if not np.allclose(left_gripper_pose[0], left_target_position, atol=2e-2):
                return False
            # Check if right gripper is close to cube
            if not np.allclose(right_gripper_pose[0], right_target_position, atol=2e-2):
                return False
            return True
        
        def move_to_pre_pick_state():
            """Move to pre-pick position"""
            if self.state_first_entry:
                cube_state = self.simulator.get_object_state("/World/Cube")
                self.cube_position = cube_state["position"].copy()
                self.cube_orientation = cube_state["orientation"].copy()
                self.state_first_entry = False
            left_target_position = self.cube_position + np.array([0, 0, 0.15])
            left_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            right_target_position = np.array([0.4, -0.3, 0.8])
            right_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            
            # Solve IK for left arm
            joint_positions = self.solve_ik(
                left_target_position=left_target_position,
                left_target_orientation=left_target_orientation,
                right_target_position=right_target_position,
                right_target_orientation=right_target_orientation,
            )
            
            # Move left arm to pick position
            self.interface.left_arm.set_joint_positions(joint_positions["left_arm"])
            self.interface.right_arm.set_joint_positions(joint_positions["right_arm"])
            self.interface.head.set_joint_positions(joint_positions["head"])
            self.interface.leg.set_joint_positions(joint_positions["leg"])
            
            left_gripper_pose = self.get_left_gripper_pose()

            # Check if left gripper is close to target position
            if not np.allclose(left_gripper_pose[0], left_target_position, atol=5e-2):
                return False
            if not np.allclose(left_gripper_pose[1], left_target_orientation, atol=5e-2):
                return False
            return True

        def move_to_pick_state():
            """Move to pick position"""
            if self.state_first_entry:
                cube_state = self.simulator.get_object_state("/World/Cube")
                self.cube_position = cube_state["position"].copy()
                self.cube_orientation = cube_state["orientation"].copy()

                self.state_first_entry = False
            left_target_position = self.cube_position + np.array([0, 0, 0.03])
            left_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            right_target_position = np.array([0.4, -0.3, 0.8])
            right_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            
            # Solve IK for left arm
            joint_positions = self.solve_ik(
                left_target_position=left_target_position,
                left_target_orientation=left_target_orientation,
                right_target_position=right_target_position,
                right_target_orientation=right_target_orientation,
            )
            
            # Move left arm to pick position
            self.interface.left_arm.set_joint_positions(joint_positions["left_arm"])
            self.interface.right_arm.set_joint_positions(joint_positions["right_arm"])
            self.interface.head.set_joint_positions(joint_positions["head"])
            self.interface.leg.set_joint_positions(joint_positions["leg"])
            
            left_gripper_pose = self.get_left_gripper_pose()

            # Check if left gripper is close to target position
            if not np.allclose(left_gripper_pose[0], left_target_position, atol=5e-2):
                return False
            if not np.allclose(left_gripper_pose[1], left_target_orientation, atol=5e-2):
                return False
            return True
        
        def grasp_state():
            """Grasp the object"""
            # Close gripper
            self.interface.left_gripper.set_gripper_close()
            return True

        def move_to_pre_place_state():
            """Move to place position"""

            left_target_position = self.cube_position + np.array([0, 0, 0.4])
            left_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            right_target_position = np.array([0.4, -0.3, 0.7])
            right_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            
            # Solve IK for left arm 
            joint_positions = self.solve_ik(
                left_target_position=left_target_position,
                left_target_orientation=left_target_orientation,
                right_target_position=right_target_position,
                right_target_orientation=right_target_orientation,
            )
            
            # Move left arm to pick position
            self.interface.left_arm.set_joint_positions(joint_positions["left_arm"])
            self.interface.right_arm.set_joint_positions(joint_positions["right_arm"])
            self.interface.head.set_joint_positions(joint_positions["head"])
            self.interface.leg.set_joint_positions(joint_positions["leg"])
            
            left_gripper_pose = self.get_left_gripper_pose()

            # Check if left gripper is close to target position
            if not np.allclose(left_gripper_pose[0], left_target_position, atol=5e-2):
                return False
            if not np.allclose(left_gripper_pose[1], left_target_orientation, atol=5e-2):
                return False
            return True

        def move_to_place_state():
            """Move to place position"""
            if self.state_first_entry:
                closet_state = self.simulator.get_object_state("/World/Closet")
                self.closet_position = closet_state["position"].copy()
                self.state_first_entry = False

            left_target_position = self.closet_position + np.array([0, 0, 0.3])
            left_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            right_target_position = np.array([0.4, -0.3, 0.7])
            right_target_orientation = np.array([0, 0.7071, 0, 0.7071])
            
            # Solve IK for left arm
            joint_positions = self.solve_ik(
                left_target_position=left_target_position,
                left_target_orientation=left_target_orientation,
                right_target_position=right_target_position,
                right_target_orientation=right_target_orientation,
            )
            
            # Move left arm to pick position
            self.interface.left_arm.set_joint_positions(joint_positions["left_arm"])
            self.interface.right_arm.set_joint_positions(joint_positions["right_arm"])
            self.interface.head.set_joint_positions(joint_positions["head"])
            self.interface.leg.set_joint_positions(joint_positions["leg"])
            
            left_gripper_pose = self.get_left_gripper_pose()

            # Check if left gripper is close to target position
            if not np.allclose(left_gripper_pose[0], left_target_position, atol=5e-2):
                return False
            if not np.allclose(left_gripper_pose[1], left_target_orientation, atol=5e-2):
                return False
            return True
        
        def release_state():
            """Release the object"""
            # Open gripper
            self.interface.left_gripper.set_gripper_open()
            return True
        
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
            print(f"Current state: {self.state_machine.get_state_name()}")
        
        # Check if current state is complete and move to next state
        if self.state_machine.execute_current_state():
            current_time = time.time()
            if current_time - self.last_state_transition_time >= 1.0:
                self.state_machine.next()
                self.last_state_transition_time = current_time

if __name__ == "__main__":
    env = IoaiGraspEnv(headless=False)
    env.simulator.add_physics_callback("pick_and_place", env.pick_and_place_callback)
    env.simulator.loop()
    env.simulator.close()
