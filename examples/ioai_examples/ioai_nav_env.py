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

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
from physics_simulator.utils.data_types import JointTrajectory
from pathlib import Path
from physics_simulator.galbot_interface import (
    GalbotInterface,
    GalbotInterfaceConfig
)
from synthnova_config import CuboidConfig
import numpy as np
import random
import math

from physics_simulator.utils.path_planner import AStarPathPlanner
from physics_simulator.utils.control_utils import BasicPathFollower

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

class OlympicNavEnv:
    def __init__(self, headless=False):
        self.simulator = None
        self.robot = None
        self.interface = None

        # Path planning
        self.planner = AStarPathPlanner(grid_size=1, obstacle_radius=0.5)
        
        # Basic path following
        self.path_follower = BasicPathFollower(velocity=0.8)
        self.waypoint_tolerance = 0.1
        self.current_target_index = 0

        self._setup_simulator(headless)
        self._setup_interface()
        self._init_pose()

        # Get actual starting position after initialization
        current_pos, _ = self._get_current_state()
        self.start_pos = (current_pos[0], current_pos[1])
        self.goal_pos = (10, 10)
        
        self.path = self.planner.find_path(self.start_pos, self.goal_pos)
        print(f"Robot actual start position: {self.start_pos}")
        print(f"Path first few points: {self.path[:5]}")

    def _setup_simulator(self, headless):
        """
        Initialize the physics simulator with basic configuration.
        
        Args:
            headless: Whether to run in headless mode
        """
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )
        
        self.simulator = PhysicsSimulator(sim_config)
        self.simulator.add_default_scene()

        robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
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

        self._add_random_obstacles()
        self.simulator.initialize()
        
        self.robot = self.simulator.get_robot("/World/Galbot")

    def _setup_interface(self):
        config = GalbotInterfaceConfig()
        config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        config.modules_manager.enabled_modules.extend([
            "right_arm", "left_arm", "leg", "head", "chassis"
        ])

        # Joint configurations
        config.right_arm.joint_names = [f"{robot_name}/right_arm_joint{i}" for i in range(1, 8)]
        config.left_arm.joint_names = [f"{robot_name}/left_arm_joint{i}" for i in range(1, 8)]
        config.leg.joint_names = [f"{robot_name}/leg_joint{i}" for i in range(1, 5)]
        config.head.joint_names = [f"{robot_name}/head_joint{i}" for i in range(1, 3)]
        config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint", 
            f"{robot_name}/mobile_yaw_joint",
        ]

        self.interface = GalbotInterface(galbot_interface_config=config, simulator=self.simulator)
        self.interface.initialize()

    def _init_pose(self):
        # Initialize robot pose
        poses = {
            self.interface.head: [0.0, 0.0],
            self.interface.leg: [0.43, 1.48, 1.07, 0.0],
            self.interface.left_arm: [-0.4654513936071508, 1.4785659313201904, -0.6235712173907869, 2.097979784011841, 1.3999720811843872, -0.009971064515411854, 1.0999830961227417],
            self.interface.right_arm: [0.4654513936071508, -1.4785659313201904, 0.6235712173907869, -2.097979784011841, -1.3999720811843872, 0.009971064515411854, -1.0999830961227417]
        }
        
        for module, pose in poses.items():
            module.set_joint_positions(pose, immediate=True)

    def _add_random_obstacles(self):        
        obstacle_points = self.planner.generate_obstacles(
            probability=0.3, exclusion_zones=[(0, 0, 2), (10, 10, 2)]
        )
        for i, point in enumerate(obstacle_points):
            self.simulator.add_object(
                CuboidConfig(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=[point[0], point[1], 0.4],
                    scale=[0.6, 0.6, 0.6],
                    color=[random.random(), random.random(), random.random()]
                )
            )

    def _move_joints_to_target(self, module, target_positions, steps=200):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def _get_current_state(self):
        """Get current chassis position and heading"""
        joints = self.interface.chassis.get_joint_positions()
        return (joints[0], joints[1]), joints[2]  # (x, y), yaw

    def _update_target_index(self, current_pos):
        """Update target waypoint index based on distance"""
        if self.current_target_index < len(self.path):
            target = self.path[self.current_target_index]
            distance = math.sqrt(
                (target[0] - current_pos[0])**2 + (target[1] - current_pos[1])**2
            )
            if distance < self.waypoint_tolerance:
                self.current_target_index += 1
                return True
        return False

    def follow_path_callback(self):
        """Basic path following callback"""
        if self.simulator.get_step_count() < 3000:
            return

        # Check if path is complete
        if self.current_target_index >= len(self.path):
            self.interface.chassis.set_joint_velocities([0.0, 0.0, 0.0])
            self.simulator.remove_physics_callback("follow_path_callback")
            print("Navigation completed!")
            return

        # Get current state from interface
        current_pos, current_heading = self._get_current_state()
        
        # Update target waypoint
        self._update_target_index(current_pos)
        
        # Get current target position
        if self.current_target_index < len(self.path):
            target_pos = self.path[self.current_target_index]
        else:
            target_pos = self.path[-1]
        
        # Calculate control commands using basic PID
        forward_vel, side_vel, yaw_vel = self.path_follower.calculate_control(
            current_pos, current_heading, target_pos
        )
        
        # Apply velocities
        self.interface.chassis.set_joint_velocities([forward_vel, side_vel, yaw_vel])
        
        # Debug info every 1000 steps
        if self.simulator.get_step_count() % 1000 == 0:
            print(f"Current: ({current_pos[0]:.1f}, {current_pos[1]:.1f}), "
                  f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}), "
                  f"Waypoint: {self.current_target_index}/{len(self.path)}")

if __name__ == "__main__":
    env = OlympicNavEnv(headless=False)
    env.simulator.play()
    env.simulator.add_physics_callback("follow_path_callback", env.follow_path_callback)
    env.simulator.loop()
    env.simulator.close()