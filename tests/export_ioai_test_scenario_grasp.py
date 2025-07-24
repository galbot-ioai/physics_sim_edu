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
# Description: Example of exporting ioai test scenario (7*7)
# Author: Chenyu Cao@Galbot
# Date: 2025-07-23
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterfaceConfig, GalbotInterface
from synthnova_config import PhysicsSimulatorConfig
from synthnova_config import MeshConfig, CuboidConfig, RobotConfig
from synthnova_config import RgbCameraConfig, DepthCameraConfig, RealsenseD436RgbSensorConfig, RealsenseD436DepthSensorConfig
from pathlib import Path
import os
import numpy as np


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
        orientation=[0, 0, 1, 0],
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)

    # Add front head camera
    # Add front head RGB camera (RealSense D405)
    front_head_rgb_camera_config = RgbCameraConfig(
        name="front_head_rgb_camera",
        prim_path=os.path.join(
            robot_path,
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
    front_head_rgb_camera_path = synthnova_physics_simulator.add_sensor(front_head_rgb_camera_config)

    # Add front head depth camera (RealSense D436)
    front_head_depth_camera_config = DepthCameraConfig(
        name="front_head_depth_camera",
        prim_path=os.path.join(
            robot_path,
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
    front_head_depth_camera_path = synthnova_physics_simulator.add_sensor(front_head_depth_camera_config)

    # Initialize the galbot interface
    galbot_interface_config = GalbotInterfaceConfig()
    # Enable the modules
    galbot_interface_config.modules_manager.enabled_modules.append("front_head_camera")
    # Bind the simulation entity prim path to the interface config
    galbot_interface_config.robot.prim_path = robot_path
    galbot_interface_config.front_head_camera.prim_path_rgb = front_head_rgb_camera_path
    galbot_interface_config.front_head_camera.prim_path_depth = front_head_depth_camera_path
    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config,
        simulator=synthnova_physics_simulator
    )
    galbot_interface.initialize()

    # Add a table
    table_position = np.array([-0.7, 0, 0])
    table_orientation = np.array([0, 0, 0, 1])

    table_config = MeshConfig(
        prim_path="/World/Table",
        name="table",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("table")
            .joinpath("table.xml"),
        position=table_position,
        orientation=table_orientation,
        scale=[0.5747, 0.5747, 0.5747]
    )
    table_path = synthnova_physics_simulator.add_object(table_config)

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
        position=table_position + np.array([0.05, 0.1, 0.53]),
        orientation=[0.5, -0.5, -0.5, 0.5],
        scale=[1.0, 1.0, 1.0]
    )
    power_drill_path = synthnova_physics_simulator.add_object(power_drill_config)

    # Add cube
    cube_config = CuboidConfig(
        name="cube",
        prim_path=os.path.join("/World", "cube"),
        position=table_position + np.array([0.2, 0.2, 0.5]),
        orientation=table_orientation,
        scale=[0.05, 0.05, 0.05],
        color=[1.0, 0.0, 0.0]
    )
    cube_path = synthnova_physics_simulator.add_object(cube_config)

    # Add bucket
    bucket_config = MeshConfig(
        name="bucket_1",
        prim_path=os.path.join("/World", "bucket"),
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects_aigc")
            .joinpath("bucket")
            .joinpath("bucket.xml"),
        position=table_position + np.array([0.15, -0.3, 0.5]),
        orientation=table_orientation,
        scale=[0.249, 0.249, 0.249],
    )
    bucket_path = synthnova_physics_simulator.add_object(bucket_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Export to a json file
    synthnova_physics_simulator.export_scenario(
        Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("scenarios")
        .joinpath("ioai_test_scenario_grasp.json")
    )

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()