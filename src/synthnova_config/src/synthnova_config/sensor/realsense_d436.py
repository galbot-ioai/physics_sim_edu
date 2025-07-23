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
# Description: Sensor config for Realsense D436 RGB and Depth camera
# Author: Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

from .basic import RgbSensorConfig, DepthSensorConfig
from typing import Literal, List, Optional


class RealsenseD436RgbSensorConfig(RgbSensorConfig):
    """
    Configuration class for the RealSense D436 RGB camera.

    Intrinsics of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}:
      - Width:         1280
      - Height:        720
      - PPX:          636.495544433594
      - PPY:          363.409881591797
      - Fx:           638.315795898438
      - Fy:           637.68310546875
      - Distortion:   Inverse Brown Conrady (using plumb_bob model)
      - Coeffs:       [0.0, 0.0, 0.0, 0.0, 0.0]
      - FOV (deg):    90.15 x 58.89
    """

    frequency: float | None = 30.0
    width: int | None = 1280
    height: int | None = 720
    fx: float | None = 638.315795898438
    fy: float | None = 637.68310546875
    cx: float | None = 636.495544433594
    cy: float | None = 363.409881591797
    pixel_size: float | None = 0.003
    f_stop: float | None = 0.0
    focus_distance: float | None = 0.0
    projection_type: str | None = "pinhole"
    clipping_range: list[float] | None = [0.05, 15.0]
    distortion_model: str | None = "plumb_bob"
    distortion_coefficients: list[float] | None = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float | None = 90.15
    vertical_fov: float | None = 58.89


class RealsenseD436DepthSensorConfig(DepthSensorConfig):
    """
    Configuration class for the RealSense D436 Depth camera.

    Intrinsics of "Depth" / 1280x720 / {Z16}:
      - Width:         1280
      - Height:        720
      - PPX:          643.402221679688
      - PPY:          355.124542236328
      - Fx:           641.102600097656
      - Fy:           641.102600097656
      - Distortion:   Brown Conrady (using plumb_bob model)
      - Coeffs:       [0.0, 0.0, 0.0, 0.0, 0.0]
      - FOV (deg):    89.9 x 58.63
    """

    frequency: int = 30
    width: int = 1280
    height: int = 720
    fx: float = 641.102600097656
    fy: float = 641.102600097656
    cx: float = 643.402221679688
    cy: float = 355.124542236328
    pixel_size: float = 0.003
    f_stop: float = 0.0
    focus_distance: float = 0.0
    projection_type: Literal["pinhole"] = "pinhole"
    clipping_range: List[float] = [0.05, 15.0]
    distortion_model: Literal["plumb_bob"] = "plumb_bob"
    distortion_coefficients: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float = 89.9
    vertical_fov: float = 58.63
