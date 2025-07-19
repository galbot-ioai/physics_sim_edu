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
# Description: Sensor config for Realsense D435 RGB and Depth camera
# Author: Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

from .basic import RgbSensorConfig, DepthSensorConfig
from typing import Literal, List, Optional


class RealsenseD435RgbSensorConfig(RgbSensorConfig):
    """
    Configuration class for the RealSense D435 RGB camera.

    Intrinsics of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}:
      - Width:         1280
      - Height:        720
      - PPX:          634.326416015625
      - PPY:          370.207214355469
      - Fx:           910.542541503906
      - Fy:           910.544921875
      - Distortion:   Inverse Brown Conrady (using plumb_bob model)
      - Coeffs:       [0.0, 0.0, 0.0, 0.0, 0.0]
      - FOV (deg):    70.2 x 43.14
    """

    frequency: float | None = 60.0
    width: int | None = 1280
    height: int | None = 720
    fx: float | None = 910.542541503906
    fy: float | None = 910.544921875
    cx: float | None = 634.326416015625
    cy: float | None = 370.207214355469
    pixel_size: float | None = 0.003
    f_stop: float | None = 0.0
    focus_distance: float | None = 0.0
    projection_type: str | None = "pinhole"
    clipping_range: list[float] | None = [0.05, 15.0]
    distortion_model: str | None = "plumb_bob"
    distortion_coefficients: list[float] | None = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float | None = 70.2
    vertical_fov: float | None = 43.14


class RealsenseD435DepthSensorConfig(DepthSensorConfig):
    """
    Configuration class for the RealSense D435 Depth camera.

    Intrinsics aligned with RGB sensor for simulation consistency.
    """

    frequency: int = 60
    width: int = 1280
    height: int = 720
    fx: float = 910.542541503906
    fy: float = 910.544921875
    cx: float = 634.326416015625
    cy: float = 370.207214355469
    pixel_size: float = 0.003
    f_stop: float = 0.0
    focus_distance: float = 0.0
    projection_type: Literal["pinhole"] = "pinhole"
    clipping_range: List[float] = [0.05, 15.0]
    distortion_model: Literal["plumb_bob"] = "plumb_bob"
    distortion_coefficients: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float = 88.59
    vertical_fov: float = 57.52
