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
# Description: Scenario configuration module for Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-05-11
#
#####################################################################################
from typing import Optional, Dict, Any, List, Literal, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import toml
import json
from pathlib import Path
import uuid
from typing_extensions import Self

from ..robot import RobotConfig
from ..object import (
    ObjectConfig,
    MeshConfig,
    CuboidConfig,
    ConeConfig,
    CapsuleConfig,
    CylinderConfig,
    SphereConfig,
)
from ..sensor import SensorConfig, RgbCameraConfig, DepthCameraConfig, Lidar3DConfig, ImuConfig
from ..object import GroundPlaneConfig


class ScenarioConfig(BaseModel):
    """Configuration class for managing scene scenarios in TOML format.

    This class provides functionality to load and export scene configurations
    in TOML format. It supports storing and retrieving scene elements like
    robots, objects, sensors, and ground plane configurations.

    Attributes:
        name (str): Name of the scenario
        description (str): Description of the scenario
        uuid (str): Unique identifier for the scenario
        robots (List[RobotConfig]): List of robot configurations
        objects (List[Union[MeshConfig, CuboidConfig, ConeConfig, CapsuleConfig, CylinderConfig, SphereConfig]]): List of object configurations
        sensors (List[Union[RgbCameraConfig, DepthCameraConfig, Lidar3DConfig, ImuConfig]]): List of sensor configurations
        ground_planes (List[GroundPlaneConfig]): List of ground plane configurations
    """

    name: str | None = Field(default=None, description="Name of the scenario")
    description: str | None = Field(
        default=None, description="Description of the scenario"
    )
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_"),
        description="Unique identifier for the scenario",
    )
    robots: List[RobotConfig] = Field(
        default_factory=list, description="List of robot configurations"
    )
    objects: List[
        Union[
            MeshConfig,
            CuboidConfig,
            ConeConfig,
            CapsuleConfig,
            CylinderConfig,
            SphereConfig,
        ]
    ] = Field(default_factory=list, description="List of object configurations")
    sensors: List[Union[RgbCameraConfig, DepthCameraConfig, Lidar3DConfig, ImuConfig]] = Field(
        default_factory=list, description="List of sensor configurations"
    )
    ground_planes: List[GroundPlaneConfig] = Field(
        default_factory=list, description="List of ground plane configurations"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate the scenario name.

        Args:
            v: The name to validate

        Returns:
            str: The validated name

        Raises:
            ValueError: If the name is invalid
        """
        if not v.strip():
            raise ValueError("Scenario name cannot be empty")
        return v.strip()

    @field_validator("uuid", mode="before")
    @classmethod
    def validate_uuid_format(cls, value: str | None) -> str | None:
        """Validate and convert UUID format to use underscores instead of hyphens.

        Args:
            value: UUID string to validate

        Returns:
            str | None: UUID string with underscores, or None if input is None

        Raises:
            ValueError: If UUID format is invalid
        """
        if value is None:
            return None

        # Convert to string if not already
        value = str(value)

        # Replace hyphens with underscores
        value = value.replace("-", "_")

        # Validate UUID format (after replacing hyphens with underscores)
        try:
            # Convert back to standard UUID format for validation
            uuid_obj = uuid.UUID(value.replace("_", "-"))
            # Convert back to underscore format
            return str(uuid_obj).replace("-", "_")
        except ValueError:
            raise ValueError("Invalid UUID format")

    @model_validator(mode="after")
    @classmethod
    def set_default_name(cls, self) -> Self:
        """Set default name using uuid with underscores if name is not provided.

        Args:
            self: The model instance

        Returns:
            Self: The model instance
        """
        if self.name is None:
            # Convert UUID to use underscores instead of hyphens
            uuid_str = self.uuid.replace("-", "_")
            self.name = f"scenario_{uuid_str}"
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert ScenarioConfig to a dictionary.

        Returns:
            Dictionary representation of ScenarioConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create a ScenarioConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            ScenarioConfig: Configured scenario instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "ScenarioConfig":
        """Load scenario configuration from a TOML file.

        Args:
            file_path: Path to the TOML file

        Returns:
            ScenarioConfig: Configured scenario instance

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file content is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {file_path}: {str(e)}")

    def export_to_file(self, file_path: str | Path) -> None:
        """Export scenario configuration to a TOML file.

        Args:
            file_path: Path where the TOML file will be saved

        Raises:
            ValueError: If the export fails
        """
        file_path = Path(file_path)
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Get the JSON string with indentation for better readability
            json_string = self.model_dump_json(indent=4)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_string)
        except Exception as e:
            raise ValueError(f"Failed to export configuration to {file_path}: {str(e)}")
