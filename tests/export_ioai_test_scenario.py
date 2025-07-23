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
# Description: Example of adding mesh objects to the simulation
# Author: Chenyu Cao@Galbot
# Date: 2025-07-23
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig
from synthnova_config import MeshConfig, CuboidConfig, RobotConfig
from pathlib import Path
import os


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
        position=[-0.5, -1.4, 0],
        orientation=[0, 0, 1, 0],
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)

    # Add a shelf
    shelf_config = MeshConfig(
        prim_path="/World/Shelf",
        name="shelf",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("shelf")
            .joinpath("shelf.xml"),
        position=[-0.8, -4.5, 0],
        orientation=[0, 0, 0.707, 0.707],
        scale=[1.1312, 1.1312, 1.1312]
    )
    synthnova_physics_simulator.add_object(shelf_config)

    # Add a table
    table_config = MeshConfig(
        prim_path="/World/Table",
        name="table",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("table")
            .joinpath("table.xml"),
        position=[-4.4, -0.4, 0],
        orientation=[0, 0, 0.707, 0.707],
        scale=[0.5747, 0.5747, 0.5747]
    )
    synthnova_physics_simulator.add_object(table_config)

    # Add a mug
    mug_config = MeshConfig(
        prim_path="/World/Mug",
        name="mug",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("mug")
            .joinpath("mug.xml"),
        position=[-4.7, -0.4, 0.45],
        orientation=[0, 0, -0.707, 0.707],
        scale=[1.0, 1.0, 1.0]
    )
    synthnova_physics_simulator.add_object(mug_config)

    # Add a power drill
    power_drill_config = MeshConfig(
        prim_path="/World/Power_drill",
        name="power_drill",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("power_drill")
            .joinpath("power_drill.xml"),
        position=[-4.2, -0.4, 0.53],
        orientation=[0.5, -0.5, -0.5, 0.5],
        scale=[1.0, 1.0, 1.0]
    )
    synthnova_physics_simulator.add_object(power_drill_config)

    # Add cone 1
    cone_1_config = MeshConfig(
        name="cone_1",
        prim_path=os.path.join("/World", "cone_1"),
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("cone")
            .joinpath("cone.xml"),
        position=[-4.5, -3.4, 0.25],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
    )
    cone_1_path = synthnova_physics_simulator.add_object(cone_1_config)

    # Add cone 2
    cone_2_config = MeshConfig(
        name="cone_2",
        prim_path=os.path.join("/World", "cone_2"),
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("cone")
            .joinpath("cone.xml"),
        position=[-2.5, -2.5, 0.25],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
    )
    cone_2_path = synthnova_physics_simulator.add_object(cone_2_config)

    # Add cone 3
    cone_3_config = MeshConfig(
        name="cone_3",
        prim_path=os.path.join("/World", "cone_3"),
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("cone")
            .joinpath("cone.xml"),
        position=[-3, -0.7, 0.25],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
    )
    cone_3_path = synthnova_physics_simulator.add_object(cone_3_config)

    # Add bucket 1
    bucket_1_config = MeshConfig(
        name="bucket_1",
        prim_path=os.path.join("/World", "bucket_1"),
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("bucket")
            .joinpath("bucket.xml"),
        position=[-2.2, -4.4, 0.25],
        orientation=[0, 0, 0, 1],
        scale=[0.5, 0.5, 0.5],
    )
    bucket_1_path = synthnova_physics_simulator.add_object(bucket_1_config)

    # Add left wall (x=-5)
    wall_left_config = CuboidConfig(
        name="wall_left",
        prim_path="/World/Wall_Left",
        position=[-5, -2.5, 1],
        orientation=[0, 0, 0, 1],
        scale=[0.1, 5, 2],
        color=[1.0, 1.0, 1.0],
        interaction_type="static",
    )
    synthnova_physics_simulator.add_object(wall_left_config)

    # Add right wall (x=0)
    wall_right_config = CuboidConfig(
        name="wall_right",
        prim_path="/World/Wall_Right",
        position=[0, -2.5, 1],
        orientation=[0, 0, 0, 1],
        scale=[0.1, 5, 2],
        color=[1.0, 1.0, 1.0],
        interaction_type="static",
    )
    synthnova_physics_simulator.add_object(wall_right_config)

    # Add bottom wall (y=-5)
    wall_bottom_config = CuboidConfig(
        name="wall_bottom",
        prim_path="/World/Wall_Bottom",
        position=[-2.5, -5, 1],
        orientation=[0, 0, 0, 1],
        scale=[5, 0.1, 2],
        color=[1.0, 1.0, 1.0],
        interaction_type="static",
    )
    synthnova_physics_simulator.add_object(wall_bottom_config)

    # Add top wall (y=0)
    wall_top_config = CuboidConfig(
        name="wall_top",
        prim_path="/World/Wall_Top",
        position=[-2.5, 0, 1],
        orientation=[0, 0, 0, 1],
        scale=[5, 0.1, 2],
        color=[1.0, 1.0, 1.0],
        interaction_type="static",
    )
    synthnova_physics_simulator.add_object(wall_top_config)

    # Add floot
    floor_config = CuboidConfig(
        name="floor",
        prim_path="/World/Floor",
        position=[-2.5, -2.5, -0.001],
        orientation=[0, 0, 0, 1],
        scale=[5, 5, 0.002],
        color=[1.0, 1.0, 1.0],
        interaction_type="static",
    )
    synthnova_physics_simulator.add_object(floor_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Export to a json file
    synthnova_physics_simulator.export_scenario(
        Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("scenarios")
        .joinpath("ioai_test_scenario.json")
    )

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()