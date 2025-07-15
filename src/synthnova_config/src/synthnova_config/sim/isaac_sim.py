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
# Description: Isaac Sim config of Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Dict


class IsaacSimConfig(BaseModel):
    physics_hz: int = Field(
        default=60,
        gt=0,
        description="Physics simulation frequency in Hz",
        json_schema_extra={"examples": [60, 120, 240]},
    )

    rendering_hz: int = Field(
        default=60,
        gt=0,
        description="Rendering frequency in Hz",
        json_schema_extra={"examples": [60, 120, 240]},
    )

    width: int = Field(
        default=1280,
        ge=0,
        description="Screen width in pixels",
        json_schema_extra={"examples": [1280, 1920, 3840]},
    )

    height: int = Field(
        default=720,
        ge=0,
        description="Screen height in pixels",
        json_schema_extra={"examples": [720, 1080, 2160]},
    )

    headless: bool = Field(
        default=False,
        description="Run the simulator in headless mode (no graphical output)",
        json_schema_extra={"examples": [True, False]},
    )

    sync_loads: bool = Field(
        default=True,
        description="Load resources synchronously",
        json_schema_extra={"examples": [True, False]},
    )

    renderer: Literal["RayTracedLighting", "PathTracing"] = Field(
        default="RayTracedLighting",
        description="Rendering type: 'RayTracedLighting' or 'PathTracing'",
        json_schema_extra={"examples": ["RayTracedLighting", "PathTracing"]},
    )

    active_gpu: int = Field(
        default=0,
        ge=0,
        description="GPU device ID for rendering",
        json_schema_extra={"examples": [0, 1]},
    )

    physics_gpu: int = Field(
        default=0,
        ge=0,
        description="GPU device ID for physics simulation",
        json_schema_extra={"examples": [0, 1]},
    )

    multi_gpu: bool = Field(
        default=True,
        description="Enable multi-GPU support",
        json_schema_extra={"examples": [True, False]},
    )

    max_gpu_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of GPUs to use (None means no limit)",
        json_schema_extra={"examples": [None, 2, 4]},
    )

    hide_ui: bool = Field(
        default=False,
        description="Hide the UI",
        json_schema_extra={"examples": [True, False]},
    )

    gpu_dynamics: bool = Field(
        default=True,
        description="Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU",
        json_schema_extra={"examples": [True, False]},
    )

    gamepad_camera_control: bool = Field(
        default=False,
        description="Enable gamepad for camera control",
        json_schema_extra={"examples": [True, False]},
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Ensure values are validated on assignment
        extra="forbid",  # Disallow extra fields
        frozen=False,  # Allow modification after creation
    )
