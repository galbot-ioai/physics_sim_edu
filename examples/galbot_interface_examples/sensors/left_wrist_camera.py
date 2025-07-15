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
# Description: Example of using the galbot interface to control the left wrist camera
# Author: Chenyu Cao@Galbot
# Date: 2025-05-21
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import (
    PhysicsSimulatorConfig,
    RobotConfig,
    RgbCameraConfig,
    RealsenseD405RgbSensorConfig,
    DepthCameraConfig,
    RealsenseD405DepthSensorConfig,
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils import preprocess_depth
import os
import numpy as np
import cv2

from pathlib import Path
import open3d as o3d

def visualize_point_cloud(point_cloud_data):
    """Simple point cloud visualization with Open3D."""
    if point_cloud_data is None or len(point_cloud_data) == 0:
        print("No point cloud data to visualize")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def main():
    """Main function to set up and run the left wrist camera example."""
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
            .joinpath("galbot_one_foxtrot_description")
            .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)
    
    # Add left wrist RGB camera (RealSense D405)
    left_wrist_rgb_camera_config = RgbCameraConfig(
        name="left_wrist_rgb_camera",
        prim_path=os.path.join(
            robot_path,
            "left_arm_link7",
            "left_arm_end_effector_mount_link",
            "left_wrist_rgb_camera",
        ),
        translation=[
            -0.028237434249527474,
            0.010736333485840343,
            0.06633936044703356,
        ],
        rotation=[
            0.521684269699651,
            -0.5095806530392278,
            0.4864975446445531,
            -0.4811374230413561,
        ],
        camera_axes="ros",
        sensor_config=RealsenseD405RgbSensorConfig(),
        parent_entity_name="galbot_one_foxtrot/left_arm_end_effector_mount_link"
    )
    left_wrist_rgb_camera_path = synthnova_physics_simulator.add_sensor(left_wrist_rgb_camera_config)

    # Add left wrist depth camera (RealSense D405)
    left_wrist_depth_camera_config = DepthCameraConfig(
        name="left_wrist_depth_camera",
        prim_path=os.path.join(
            robot_path,
            "left_arm_link7",
            "left_arm_end_effector_mount_link",
            "left_wrist_depth_camera",
        ),
        translation=[
            -0.028237434249527474,
            0.010736333485840343,
            0.06633936044703356,
        ],
        rotation=[
            0.521684269699651,
            -0.5095806530392278,
            0.4864975446445531,
            -0.4811374230413561,
        ],
        camera_axes="ros",
        sensor_config=RealsenseD405DepthSensorConfig(),
        parent_entity_name="galbot_one_foxtrot/left_arm_end_effector_mount_link"
    )
    left_wrist_depth_camera_path = synthnova_physics_simulator.add_sensor(
        left_wrist_depth_camera_config
    )

    # Initialize the galbot interface
    galbot_interface_config = GalbotInterfaceConfig()
    # Enable the modules
    galbot_interface_config.modules_manager.enabled_modules.append("left_wrist_camera")
    # Bind the simulation entity prim path to the interface config
    galbot_interface_config.robot.prim_path = robot_path
    galbot_interface_config.left_wrist_camera.prim_path_rgb = left_wrist_rgb_camera_path
    galbot_interface_config.left_wrist_camera.prim_path_depth = (
        left_wrist_depth_camera_path
    )
    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config, simulator=synthnova_physics_simulator
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
            rgb_data = galbot_interface.left_wrist_camera.get_rgb()
            
            # Get depth data
            depth_data = galbot_interface.left_wrist_camera.get_depth()
            
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
    params = galbot_interface.left_wrist_camera.get_parameters()
    intrinsic_matrix = params["rgb"]["intrinsic_matrix"]
    print(params)
    print("intrinsic_matrix: ", intrinsic_matrix)

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()
