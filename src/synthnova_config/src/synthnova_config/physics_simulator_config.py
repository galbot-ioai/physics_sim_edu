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
# Module: Synthnova physics simulator configuration
# Description: Top-level configuration management for Synthnova physics simulator
# Author: Herman Ye@Galbot
# Date: 2025-04-16
#
#####################################################################################

from pydantic import BaseModel, Field, ConfigDict
from .logger import LoggerConfig
from .cloud import CloudStreamingConfig
from .sim import MujocoConfig
from .scenario import ScenarioConfig


class PhysicsSimulatorConfig(BaseModel):
    """
    Central configuration class for the Synthnova physics simulator.

    This class serves as the root configuration container that orchestrates all
    components of the simulation system. It integrates and manages configurations
    for logging, simulation, and env subsystems, providing a unified
    interface for system configuration.

    The configuration system is built on Pydantic, providing:
    - Runtime type checking and validation
    - Automatic conversion of input data
    - JSON/YAML serialization support
    - Configuration inheritance and composition

    Attributes:
        logger_config: Settings for system-wide logging, including:
            - Log levels and verbosity
            - Output destinations (file, console, etc.)
            - Log formatting and rotation policies
        sim_config: Core simulation engine settings, including:
            - Physics engine parameters
            - Rendering and visualization options
            - Performance tuning parameters
        world_config: World-specific settings, including:
            - Robot and object configurations
            - Sensor and actuator settings
            - Scene and world parameters

    Example:
        ```python
        # Create a complete simulation configuration
        config = PhysicsSimulatorConfig(
            logger_config=LoggerConfig(
                log_level="INFO",
                log_file="simulation.log",
                rotation_size="10MB"
            ),
            sim_config=SimConfig(
                simulator_type="isaac_sim",
                physics_dt=0.01,
                rendering_dt=0.016
            ),
            world_config=WorldConfig(
                robot_configs=[...],
                object_configs=[...],
                sensor_configs=[...]
            )
        )
        ```

    Note:
        - All configuration fields are validated at runtime
        - Undefined fields are prohibited to prevent configuration errors
        - Configuration can be modified after initialization
        - Type hints and validation rules are enforced through Pydantic
        - Nested configurations are automatically validated
    """

    logger_config: LoggerConfig = Field(
        default_factory=LoggerConfig,
        description="System-wide logging configuration, including log levels, "
        "output destinations, and formatting options",
    )

    cloud_streaming_config: CloudStreamingConfig = Field(
        default_factory=CloudStreamingConfig,
        description="Configuration parameters for cloud-based streaming and "
        "remote visualization capabilities",
    )

    mujoco_config: MujocoConfig = Field(
        default_factory=MujocoConfig,
        description="Configuration parameters for MuJoCo integration, "
        "including graphics, physics, and rendering settings",
    )

    scenario_config: ScenarioConfig = Field(
        default_factory=ScenarioConfig,
        description="Scenario configuration, including scenario settings, "
        "object configurations, and sensor configurations",
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Enable runtime validation of field assignments
        extra="forbid",  # Prevent accidental addition of undefined fields
        frozen=False,  # Allow modification of configuration after initialization
    )
