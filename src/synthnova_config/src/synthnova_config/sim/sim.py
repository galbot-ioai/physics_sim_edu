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
# Module: Simulator Configuration
# Description: Core configuration classes for Synthnova simulator
# Author: Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

from pydantic import BaseModel, Field, ConfigDict
from .isaac_sim import IsaacSimConfig
from .mujoco import MujocoConfig
from typing import Literal
from ..cloud import CloudStreamingConfig


class SimConfig(BaseModel):
    """
    Central configuration class for Synthnova simulation environment.

    This class serves as the primary configuration container for the simulation
    environment, integrating settings for both physics and rendering subsystems.
    It supports multiple simulation backends through specialized configuration
    classes and provides runtime validation of all parameters.

    The configuration leverages Pydantic for robust type checking and validation,
    ensuring all simulation parameters are properly configured and within valid
    operational ranges.

    Attributes:
        simulator_type: Specifies the active simulation backend ("isaac_sim" or "mujoco")
        isaac_sim_config: Configuration parameters for NVIDIA Isaac Sim integration,
            including graphics, physics, and rendering settings
        mujoco_config: Configuration parameters for MuJoCo physics engine integration
        cloud_streaming_config: Settings for cloud-based streaming and remote
            visualization capabilities

    Example:
        ```python
        # Initialize simulator with Isaac Sim backend
        config = SimConfig(
            simulator_type="isaac_sim",
            isaac_sim_config=IsaacSimConfig(
                physics_dt=0.01,
                rendering_dt=0.016,
                # Additional Isaac Sim specific settings
            )
        )
        ```

    Note:
        - All configuration fields are validated at runtime
        - Undefined fields are prohibited to prevent configuration errors
        - Configuration can be modified after initialization
        - Type hints and validation rules are enforced through Pydantic
    """

    simulator_type: Literal["isaac_sim", "mujoco"] = Field(
        default="isaac_sim",
        description="Active simulation backend type",
        json_schema_extra={"examples": ["isaac_sim", "mujoco"]},
    )

    isaac_sim_config: IsaacSimConfig = Field(
        default_factory=IsaacSimConfig,
        description="Configuration parameters for NVIDIA Isaac Sim integration, "
        "including graphics, physics, and rendering settings",
    )

    mujoco_config: MujocoConfig = Field(
        default_factory=MujocoConfig,
        description="Configuration parameters for MuJoCo physics engine integration, "
        "including simulation and visualization settings",
    )

    cloud_streaming_config: CloudStreamingConfig = Field(
        default_factory=CloudStreamingConfig,
        description="Configuration parameters for cloud-based streaming and "
        "remote visualization capabilities",
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Enable runtime validation of field assignments
        extra="forbid",  # Prevent accidental addition of undefined fields
        frozen=False,  # Allow modification of configuration after initialization
    )
