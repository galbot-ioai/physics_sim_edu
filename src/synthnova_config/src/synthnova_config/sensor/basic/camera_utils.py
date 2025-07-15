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
# Description: Camera intrinsic alignment utilities for sensor configurations
# Author: Chenyu Cao@Galbot  
# Date: 2025-07-09
#
#####################################################################################

from .rgb_camera import RgbSensorConfig, RgbCameraConfig
from .depth_camera import DepthSensorConfig, DepthCameraConfig


def align_depth_camera_to_rgb_camera(rgb_camera_config: RgbCameraConfig) -> DepthCameraConfig:
    """
    Align depth camera configuration to RGB camera configuration.
    
    Creates a depth camera configuration with identical parameters to ensure 
    perfect alignment between RGB and depth cameras in simulation.
    
    Args:
        rgb_camera_config: RGB camera configuration instance
        
    Returns:
        DepthCameraConfig with aligned parameters
        
    Example:
        rgb_camera = RgbCameraConfig(...)
        depth_camera = align_depth_camera_to_rgb_camera(rgb_camera)
    """
    camera_data = rgb_camera_config.model_dump()
    # Convert sensor_config to DepthSensorConfig
    camera_data['sensor_config'] = DepthSensorConfig(**camera_data['sensor_config'])
    return DepthCameraConfig(**camera_data)


def align_rgb_camera_to_depth_camera(depth_camera_config: DepthCameraConfig) -> RgbCameraConfig:
    """
    Align RGB camera configuration to depth camera configuration.
    
    Creates an RGB camera configuration with identical parameters to ensure 
    perfect alignment between RGB and depth cameras in simulation.
    
    Args:
        depth_camera_config: Depth camera configuration instance
        
    Returns:
        RgbCameraConfig with aligned parameters
        
    Example:
        depth_camera = DepthCameraConfig(...)
        rgb_camera = align_rgb_camera_to_depth_camera(depth_camera)
    """
    camera_data = depth_camera_config.model_dump()
    # Convert sensor_config to RgbSensorConfig
    camera_data['sensor_config'] = RgbSensorConfig(**camera_data['sensor_config'])
    return RgbCameraConfig(**camera_data) 