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
# Description: IMU config of Synthnova
#
# Author: Herman Ye@Galbot, Jinrui He@Galbot
# Date: 2025-03-06
#
#####################################################################################


import numpy as np
from typing import Literal, List, Optional, Union, Tuple, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from numpydantic import NDArray, Shape
from pathlib import PosixPath
from pathlib import Path
from .sensor import SensorConfig
from ...object.object import convert_to_abs_str_path, convert_int_to_float
import uuid
from typing_extensions import Self


class ImuSensorConfig(BaseModel):
    """Configuration class for IMU sensor parameters.

    This class defines the intrinsic parameters of an IMU sensor, including its sampling frequency
    and other sensor-specific configurations. It provides methods to load configurations from JSON
    files and export configurations to files.

    Attributes:
        frequency (float | None): Sampling frequency of the sensor in Hz. If None, the default frequency
            from the simulation environment will be used.

    Methods:
        load_from_file: Load configuration from a JSON file
        export_to_file: Export configuration to a JSON file
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    frequency: float | None = Field(
        default=None, description="Frequency of the sensor in Hz"
    )

    @field_validator(
        "frequency",
        mode="before",
    )
    @classmethod
    def process_imu_sensor_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @classmethod
    def load_from_file(cls, json_path: str) -> "ImuSensorConfig":
        import json
        from pathlib import Path

        json_path = str(Path(json_path).expanduser().resolve())
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)

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


class ImuConfig(SensorConfig):
    """Configuration class for IMU sensor in the simulation environment.

    This class inherits from SensorConfig and defines both the intrinsic parameters and the pose
    of an IMU sensor in the simulation environment. It supports both global (world frame) and local
    (parent frame) coordinate systems for positioning the sensor.

    Attributes:
        type (Literal["imu"]): Type identifier for the sensor, always "imu"
        name (str | None): A human-readable identifier for the entity instance
        uuid (str | None): A unique identifier for the entity instance
        prim_path (str | PosixPath | None): USD primitive path that specifies the location of the IMU sensor
            in the USD scene graph. This path is used to identify and locate the sensor in the simulation
            environment.
        position (NDArray[Shape["3"], np.float64] | None): Global position [x, y, z] in world frame
        orientation (NDArray[Shape["4"], np.float64] | None): Global orientation quaternion [qx, qy, qz, qw] in world frame
        translation (NDArray[Shape["3"], np.float64] | None): Local position [x, y, z] relative to parent frame
        rotation (NDArray[Shape["4"], np.float64] | None): Local orientation quaternion [qx, qy, qz, qw] relative to parent frame
        sensor_config (ImuSensorConfig): Configuration for the IMU sensor parameters

    Validators:
        validate_paths: Converts prim_path to absolute string path
        check_pose: Ensures either global (position/orientation) or local (translation/rotation) pose is used, not both

    Note:
        Either global pose (position + orientation) or local pose (translation + rotation) must be specified,
        but not both. The pose defines the sensor's position and orientation in the simulation environment.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["imu"] = "imu"

    name: str | None = Field(
        ..., description="A human-readable identifier for the entity instance"
    )
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_"),
        description="A unique identifier for the entity instance",
    )
    prim_path: str | PosixPath | None = Field(
        default=None,
        description="USD primitive path that specifies the location of the IMU sensor",
    )
    position: NDArray[Shape["3"], np.float64] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Global position [x, y, z] relative to world frame",
    )
    orientation: NDArray[Shape["4"], np.float64] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Global orientation quaternion [qx, qy, qz, qw] relative to world frame",
    )
    translation: NDArray[Shape["3"], np.float64] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Local position [x, y, z] relative to parent frame",
    )
    rotation: NDArray[Shape["4"], np.float64] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Local orientation quaternion [qx, qy, qz, qw] relative to parent frame",
    )
    sensor_config: ImuSensorConfig = Field(
        default_factory=ImuSensorConfig, description="Sensor configuration"
    )

    @field_validator("prim_path")
    @classmethod
    def validate_paths(cls, value):
        """Convert paths to absolute string paths."""
        return convert_to_abs_str_path(value)

    @field_validator(
        "position", "orientation", "translation", "rotation", mode="before"
    )
    @classmethod
    def process_imu_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @model_validator(mode="after")
    @classmethod
    def check_pose_consistency(cls, self) -> Self:
        """Check that either position/orientation or translation/rotation is used, but not both.

        Args:
            self: The model instance

        Returns:
            Self: The model instance

        Raises:
            ValueError: If both global and local coordinates are used simultaneously
        """
        position = self.position
        orientation = self.orientation
        translation = self.translation
        rotation = self.rotation

        has_global = position is not None or orientation is not None
        has_local = translation is not None or rotation is not None

        if has_global and has_local:
            raise ValueError(
                "Cannot use both global (position/orientation) and local (translation/rotation) coordinates simultaneously"
            )

        return self

    @model_validator(mode="after")
    @classmethod
    def set_imu_config_default_name(cls, self) -> Self:
        """Set default name using type and uuid with underscores if name is not provided.

        Args:
            self: The model instance

        Returns:
            Self: The model instance
        """
        if self.name is None and self.type is not None:
            # Convert UUID to use underscores instead of hyphens
            uuid_str = self.uuid.replace("-", "_")
            self.name = f"{self.type}_{uuid_str}"
        return self

    @field_validator("uuid", mode="before")
    @classmethod
    def validate_imu_config_uuid_format(cls, value: str | None) -> str | None:
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert ImuConfig to a dictionary.

        Returns:
            Dictionary representation of ImuConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImuConfig":
        """Create an ImuConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            ImuConfig: Configured IMU instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
