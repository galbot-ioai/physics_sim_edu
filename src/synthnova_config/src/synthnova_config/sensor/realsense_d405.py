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
# Description: Sensor config for Realsense D405 RGB and Depth camera
# Author: Herman Ye@Galbot
# Date: 2025-04-16
#
#####################################################################################


from .basic import RgbSensorConfig, DepthSensorConfig
from typing import Literal, List, Optional


class RealsenseD405RgbSensor1280x720Config(RgbSensorConfig):
    """
    Configuration class for the RealSense D405 RGB camera(1280x720).

    Intrinsics of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}:
      - Width:         1280
      - Height:        720
      - PPX:           639.14599609375
      - PPY:           355.1600341796875
      - Fx:            649.24755859375
      - Fy:            648.3587036132812
      - Distortion:    Inverse Brown Conrady (using plumb_bob model)
      - Coeffs:        [-0.05448233708739281, 0.061593979597091675, 0.00027212564600631595, 0.00024423596914857626, -0.020646050572395325]
      - FOV (deg):     69.4 x 42.5
    """

    frequency: float | None = 30.0
    width: int | None = 1280
    height: int | None = 720
    fx: float | None = 649.24755859375
    fy: float | None = 648.3587036132812
    cx: float | None = 639.14599609375
    cy: float | None = 355.1600341796875
    pixel_size: float | None = 0.003
    f_stop: float | None = 0.0
    focus_distance: float | None = 0.0
    projection_type: str | None = "pinhole"
    clipping_range: list[float] | None = [0.05, 15.0]
    distortion_model: str | None = "plumb_bob"
    distortion_coefficients: list[float] | None = [
        -0.05448233708739281,
        0.061593979597091675,
        0.00027212564600631595,
        0.00024423596914857626,
        -0.020646050572395325,
    ]
    horizontal_fov: float | None = 69.4
    vertical_fov: float | None = 42.5


class RealsenseD405DepthSensor1280x720Config(DepthSensorConfig):
    """
    Configuration class for the RealSense D405 Depth camera(1280x720).

    Intrinsics aligned with RGB sensor for simulation consistency.
    """

    frequency: int = 30
    width: int = 1280
    height: int = 720
    fx: float = 649.24755859375
    fy: float = 648.3587036132812
    cx: float = 639.14599609375
    cy: float = 355.1600341796875
    pixel_size: float = 0.003
    f_stop: float = 0.0
    focus_distance: float = 0.0
    projection_type: Literal["pinhole"] = "pinhole"
    clipping_range: List[float] = [0.05, 15.0]
    distortion_model: Literal["plumb_bob"] = "plumb_bob"
    distortion_coefficients: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float = 75
    vertical_fov: float = 47.6


class RealsenseD405RgbSensor640x360Config(RgbSensorConfig):
    """
    Configuration class for the RealSense D405 RGB camera(640x360).

    Intrinsics of "Color" / 640x360 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}:
      - Width:           640
      - Height:          360
      - PPX:             320.10334232983035
      - PPY:             180.82591009857398
      - Fx:              326.60852933950645
      - Fy:              326.51969070689336
      - Distortion:      Fisheye Polynomial (using fisheye_rational_polynomial model)
      - Coeffs:          [2.06237741252592, -1.665695411417451, 7.66816484333735e-05, 0.0015090122930477689, 2.9956500629510607, 2.122488524220362, -1.6845480550202077, 3.0081876186493104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      - horizontal_fov:  88.83
      - vertical_fov:    57.73
      - diagonal_fov:    96.70
    """

    frequency: float | None = 30.0
    width: int | None = 640
    height: int | None = 360
    fx: float | None = 326.60852933950645
    fy: float | None = 326.51969070689336
    cx: float | None = 320.10334232983035
    cy: float | None = 180.82591009857398
    pixel_size: float | None = 0.003
    f_stop: float | None = 0.0
    focus_distance: float | None = 0.0
    projection_type: str | None = "fisheyePolynomial"
    clipping_range: list[float] | None = [0.05, 15.0]
    distortion_model: str | None = "fisheye_rational_polynomial"
    distortion_coefficients: list[float] | None = [
        2.06237741252592,
        -1.665695411417451,
        7.66816484333735e-05,
        0.0015090122930477689,
        2.9956500629510607,
        2.122488524220362,
        -1.6845480550202077,
        3.0081876186493104,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    horizontal_fov: float | None = 88.83
    vertical_fov: float | None = 57.73
    diagonal_fov: float | None = 96.70


class RealsenseD405DepthSensor640x360Config(DepthSensorConfig):
    """
    Configuration class for the RealSense D405 Depth camera(640x360).

    Intrinsics aligned with RGB sensor for simulation consistency.
    """

    frequency: int = 30
    width: int = 640
    height: int = 360
    fx: float = 326.60852933950645
    fy: float = 326.51969070689336
    cx: float = 320.10334232983035
    cy: float = 180.82591009857398
    pixel_size: float = 0.003
    f_stop: float = 0.0
    focus_distance: float = 0.0
    projection_type: Literal["pinhole"] = "pinhole"
    clipping_range: List[float] = [0.05, 15.0]
    distortion_model: Literal["plumb_bob"] = "plumb_bob"
    distortion_coefficients: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    horizontal_fov: float | None = 88.83
    vertical_fov: float | None = 57.73
    diagonal_fov: float | None = 96.70


class RealsenseD405RgbSensorConfig(RealsenseD405RgbSensor640x360Config):
    """
    Configuration class for the RealSense D405 RGB camera(640x360 Default).
    """

    pass


class RealsenseD405DepthSensorConfig(RealsenseD405DepthSensor640x360Config):
    """
    Configuration class for the RealSense D405 Depth camera(640x360 Default).
    """

    pass
