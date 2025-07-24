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
# Description: Example of adding basic geom objects to the simulation
# Author: Chenyu Cao@Galbot
# Date: 2025-05-28
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
    galbot_interface_config.modules_manager.enabled_modules.append("chassis")
    galbot_interface_config.chassis.joint_names = [
        f"{robot_config.name}/mobile_forward_joint",
        f"{robot_config.name}/mobile_side_joint",
        f"{robot_config.name}/mobile_yaw_joint",
    ]
    # Bind the simulation entity prim path to the interface config
    galbot_interface_config.robot.prim_path = robot_path
    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config,
        simulator=synthnova_physics_simulator
    )
    galbot_interface.initialize()

    # Start the simulation
    synthnova_physics_simulator.step(10)

    # Example 1: Direct velocity control
    print("Example 1: Direct velocity control")
    
    # Move forward for 2 seconds
    print("Moving forward...")
    for _ in range(2000):  # 2 seconds at 1000Hz
        galbot_interface.chassis.set_joint_positions([0.5, 0.0, 0.0])  # forward, side, yaw
        synthnova_physics_simulator.step(10)

    # Stop for 1 second
    print("Stopping...")
    for _ in range(100):
        galbot_interface.chassis.set_joint_positions([0.0, 0.0, 0.0])
        synthnova_physics_simulator.step(10)

    # Move sideways for 2 seconds
    print("Moving sideways...")
    for _ in range(2000):
        galbot_interface.chassis.set_joint_positions([0.0, 0.3, 0.0])  # forward, side, yaw
        synthnova_physics_simulator.step(10)

    # Stop for 1 second
    print("Stopping...")
    for _ in range(100):
        galbot_interface.chassis.set_joint_positions([0.0, 0.0, 0.0])
        synthnova_physics_simulator.step(10)

    # Rotate for 2 seconds
    print("Rotating...")
    for _ in range(2000):
        galbot_interface.chassis.set_joint_positions([0.0, 0.0, 0.5])  # forward, side, yaw
        synthnova_physics_simulator.step(10)

    # Stop
    print("Final stop...")
    for _ in range(100):
        galbot_interface.chassis.set_joint_positions([0.0, 0.0, 0.0])
        synthnova_physics_simulator.step(10)

    # Example 2: Combined movement - diagonal with rotation
    print("Example 2: Combined movement - diagonal with rotation")
    for _ in range(3000):
        galbot_interface.chassis.set_joint_positions([0.3, 0.2, 0.1])  # forward, side, yaw
        synthnova_physics_simulator.step(10)

    # Final stop
    for _ in range(100):
        galbot_interface.chassis.set_joint_positions([0.0, 0.0, 0.0])
        synthnova_physics_simulator.step(10)

    print("Demo completed!")

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()