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
#
#####################################################################################
#
# Description: Export a scenario to a config file
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-04-02
#
#####################################################################################

from synthnova_config import PhysicsSimulatorConfig, CuboidConfig, RobotConfig, MeshConfig
import os
from physics_simulator import PhysicsSimulator
from pathlib import Path

def main():
    # Instantiate the simulator
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default ground plane if you need
    synthnova_physics_simulator.add_default_scene()

    # Add cube
    cube_1_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_1"),
        position=[2, 2, 2],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
        color=[1.0, 0.0, 0.0],
    )
    cube_1_path = synthnova_physics_simulator.add_object(cube_1_config)

    # Add cube
    cube_2_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_2"),
        position=[0, 0, 2],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
        color=[0.0, 1.0, 0.0],
    )
    cube_2_path = synthnova_physics_simulator.add_object(cube_2_config)

    # Add cube
    cube_3_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_3"),
        position=[-2, -2, 2],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
        color=[0.0, 0.0, 1.0],
    )
    cube_3_path = synthnova_physics_simulator.add_object(cube_3_config)

    # Add a robot
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("robots")
        .joinpath("galbot_one_foxtrot_description")
        .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    synthnova_physics_simulator.add_robot(robot_config)

    # Add a mug mesh
    mug_config = MeshConfig(
        prim_path="/World/Mug",
        name="mug",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("objects")
        .joinpath("mug")
        .joinpath("mug.xml"),
        position=[0.55, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    synthnova_physics_simulator.add_object(mug_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Export the scenario to a config file
    target_file_path = (
        Path()
        .joinpath(PhysicsSimulator.get_root_directory())
        .joinpath("assets")
        .joinpath("exported_scenario.json")
    )
    synthnova_physics_simulator.export_scenario(file_path=target_file_path)

    # Run the step loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()
