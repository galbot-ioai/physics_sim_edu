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
# Description: A test env code for setting up ioai
# Author: Chenyu Cao@Galbot
# Date: 2025-07-24
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, ScenarioConfig, MujocoConfig
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from pathlib import Path

class IoaiTestEnv:
    def __init__(self, headless=False):
        self.simulator = None
        self.robot = None
        self.interface = None

        self._setup_simulator(headless)
        self._setup_interface()
        self._init_pose()

    def _setup_simulator(self, headless=False):
        sn_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )

        scenario_config = ScenarioConfig.load_from_file(
            Path()
            .joinpath(PhysicsSimulator.get_root_directory())
            .joinpath("assets")
            .joinpath("synthnova_assets")
            .joinpath("scenarios")
            .joinpath("ioai_test_scenario_grasp.json")
        )
        sn_config.scenario_config = scenario_config

        self.simulator = PhysicsSimulator(sn_config)
        self.simulator.initialize()

        self.robot = self.simulator.get_robot("/World/Galbot")

    def _setup_interface(self):
        self.interface = GalbotInterface(
            galbot_interface_config=self.galbot_interface_config,
            simulator=self.simulator
        )
        self.interface.initialize()

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
            self.interface.leg: [0.0835, 0.635, 0.523, 0.0],
            self.interface.left_arm: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            self.interface.right_arm: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        for module, pose in poses.items():
            module.set_joint_positions(pose, immediate=True)

    def run(self):
        self.simulator.loop()

    def close(self):
        self.simulator.close()

if __name__ == "__main__":
    env = IoaiTestEnv(headless=False)
    env.run()
    env.close()