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
# Description: Synthnova Ground Plane Config
# Author: Herman Ye@Galbot
# Date: 2025-05-10
#
#####################################################################################

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo
from numpydantic import NDArray, Shape
import numpy as np


class GroundPlaneConfig(BaseModel):
    """Configuration for ground plane objects in the simulation environment.

    This class defines the essential properties of ground planes in the Synthnova
    simulation environment. The ground plane is a fundamental component that provides
    a stable surface for objects to interact with.

    Attributes:
        static_friction: Coefficient of static friction (default: 0.5).
            Represents the friction force that must be overcome to start moving an object.
            Must be greater than or equal to dynamic_friction.
        dynamic_friction: Coefficient of dynamic friction (default: 0.5).
            Represents the friction force that opposes motion when an object is moving.
            Must be less than or equal to static_friction.
        restitution: Coefficient of restitution/bounciness (default: 0.8).
            Determines how much energy is conserved in collisions (0 = no bounce, 1 = perfect bounce).
        z_position: Height of the ground plane in meters (default: 0.0).
            Defines the vertical position of the ground plane in the simulation world.
    """

    static_friction: float = Field(
        default=0.5, ge=0.0, description="Coefficient of static friction (must be ≥ 0)"
    )
    dynamic_friction: float = Field(
        default=0.5, ge=0.0, description="Coefficient of dynamic friction (must be ≥ 0)"
    )
    restitution: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Coefficient of restitution (0 = no bounce, 1 = perfect bounce)",
    )
    z_position: float = Field(
        default=0.0, description="Height of the ground plane in meters"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
    )

    @field_validator("dynamic_friction")
    @classmethod
    def validate_friction_coefficients(cls, v: float, info: ValidationInfo) -> float:
        """Validate that dynamic friction is less than or equal to static friction."""
        static_friction = info.data.get("static_friction")
        if static_friction is not None and v > static_friction:
            raise ValueError(
                f"Dynamic friction ({v}) must be less than or equal to static friction "
                f"({static_friction})"
            )
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert GroundPlaneConfig to a dictionary.

        Returns:
            Dictionary representation of GroundPlaneConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundPlaneConfig":
        """Create an GroundPlaneConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            GroundPlaneConfig: Configured ground plane instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
