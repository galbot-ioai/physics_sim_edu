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
# Description: Depth camera config of Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-04-02
#
#####################################################################################

from typing import Literal, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .rgb_camera import RgbCameraConfig, RgbSensorConfig


class DepthSensorConfig(RgbSensorConfig):
    """
    Configuration class for depth camera sensor components.

    Extends the base RGB sensor configuration to support depth sensing capabilities.
    This class inherits all parameters from RgbSensorConfig while maintaining
    compatibility with the base RGB sensor configuration. It is designed to be
    extensible for future depth-specific parameters.

    Note:
        This class currently inherits all functionality from RgbSensorConfig.
        Future implementations may add depth-specific parameters and methods.
    """

    pass


class DepthCameraConfig(RgbCameraConfig):
    """
    Configuration class for depth camera systems.

    Defines the complete configuration for depth camera systems, extending the
    base RGB camera configuration with depth-specific settings. This class
    encapsulates both the camera system configuration and the depth sensor
    component configuration.

    Attributes:
        type: Fixed identifier for the camera type, set to "depth_camera"
        sensor_config: Configuration settings for the depth sensor component,
            including all inherited RGB sensor parameters and any depth-specific
            settings
    """

    type: Literal["depth_camera"] = "depth_camera"

    sensor_config: DepthSensorConfig = Field(
        default_factory=DepthSensorConfig,
        description="Configuration settings for the depth sensor component, "
        "including all inherited RGB sensor parameters and any "
        "depth-specific settings",
    )
