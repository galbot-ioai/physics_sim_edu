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
# Description: Example of using the galbot interface to control the leg
# Author: Herman Ye@Galbot
# Date: 2025-05-27
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig
import numpy as np

from pathlib import Path

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)


def main():
    # Create sim config
    my_config = PhysicsSimulatorConfig()

    # Instantiate the simulator
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default ground plane if you need
    synthnova_physics_simulator.add_default_scene()

    # Add robot
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("robots")
        .joinpath("galbot_one_foxtrot_description_simplified")
        .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Initialize the galbot interface
    galbot_interface_config = GalbotInterfaceConfig()
    # Enable the modules
    galbot_interface_config.modules_manager.enabled_modules.append("leg")
    galbot_interface_config.leg.joint_names = [
        f"{robot_config.name}/leg_joint1",
        f"{robot_config.name}/leg_joint2",
        f"{robot_config.name}/leg_joint3",
        f"{robot_config.name}/leg_joint4",
    ]
    # Bind the simulation entity prim path to the interface config
    galbot_interface_config.robot.prim_path = robot_path
    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config,
        simulator=synthnova_physics_simulator,
    )
    galbot_interface.initialize()

    # Start the simulation
    synthnova_physics_simulator.step()

    # Get current joint positions
    current_joint_positions = galbot_interface.leg.get_joint_positions()

    # Define target joint positions
    target_joint_positions = [0.5, 0.5, 0.5, 0.5]

    # Interpolate joint positions
    positions = interpolate_joint_positions(
        current_joint_positions, target_joint_positions, 500
    )
    # Create a joint trajectory
    joint_trajectory = JointTrajectory(positions=positions)

    # Follow the trajectory
    galbot_interface.leg.follow_trajectory(joint_trajectory)

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()
