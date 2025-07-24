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
# Description: Example of using the galbot interface to control the front head camera
# Author: Chenyu Cao@Galbot
# Date: 2025-05-21
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import (
    PhysicsSimulatorConfig,
    RobotConfig,
    RgbCameraConfig,
    RealsenseD436RgbSensorConfig,
    DepthCameraConfig,
    RealsenseD436DepthSensorConfig
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils import preprocess_depth
import os
import numpy as np
import cv2

from pathlib import Path


def main():
    """Main function to set up and run the front head camera example."""
    # Instantiate the simulator
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default scene
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

    # Start the simulation
    synthnova_physics_simulator.play()
    
    # Initial steps to stabilize the simulation
    synthnova_physics_simulator.step(10)

    while True:
        try:
            if not synthnova_physics_simulator.is_running():
                print("Simulator has been closed, exiting...")
                break

            synthnova_physics_simulator.step(7)
            
            # Get rgb data
            rgb_data = galbot_interface.front_head_camera.get_rgb()
            
            # Get depth data
            depth_data = galbot_interface.front_head_camera.get_depth()
            
            # Preprocess depth data for visualization
            # You can also use this function to preprocess the depth data for other purposes
            depth_data = preprocess_depth(
                depth_data,
                scale=1000,  # m to mm
                min_value=0.0,
                max_value=5 * 1000,  # 5m to mm
                data_type=np.uint16,
            )
            
            # Display images in non-blocking way
            cv2.imshow("RGB Camera", cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
            cv2.imshow("Depth Camera", depth_data)

            # Wait for 1ms and check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
                
        except (AttributeError, RuntimeError, Exception) as e:
            # Handle any errors during shutdown or resource cleanup
            print(f"Simulator resources unavailable during shutdown: {e}")
            break

    # Get camera parameters
    params = galbot_interface.front_head_camera.get_parameters()
    intrinsic_matrix = params["rgb"]["intrinsic_matrix"]
    print(params)
    print("Intrinsic matrix:", intrinsic_matrix)

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()
