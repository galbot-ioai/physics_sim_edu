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
# Description: Lidar config of Synthnova
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


class Lidar3DSensorConfig(BaseModel):
    """Configuration class for Lidar sensor parameters.

    This class defines the parameters specific to the Lidar sensor, including its
    scanning type, rotation frequency, valid range, reflectance values, and other
    sensor-specific attributes.

    Attributes:
        scan_type (str): Lidar scan type (e.g., rotary, linear). Rotary indicates mechanical rotation scanning, while linear indicates non-rotating scanning.
        rotation_frequency (float): Lidar rotation frequency in Hz, representing how many complete rotations the sensor makes per second.
        valid_range (NDArray[Shape["2"], np.float64]): Valid range of the lidar in meters [min_range, max_range]. min_range: minimum detectable distance, max_range: maximum detectable distance
        min_reflectance (float): Minimum reflectance value (typically between 0 and 1) that the lidar can detect. Below this value, the sensor might fail to register a return signal.
        min_reflectance_range (float): Maximum detection range in meters for objects with the specified minimum reflectance. This defines how far the sensor can detect objects with minimum reflectivity.
        avg_power (float): Average power consumption of the lidar in watts during operation.
        wave_length (float): Wavelength of the lidar's laser in nanometers (nm). This affects the sensor's performance in different environmental conditions.
        pulse_time (float): Pulse time of the lidar in nanoseconds (ns), representing the duration of each laser pulse.
        firing_frequency (float): Firing frequency of the lidar in Hz, determining how often the sensor emits laser pulses.
        num_of_point_cloud_per_data_frame (int): Number of point clouds generated per data frame, representing the point cloud density of the sensor.
        num_of_emitters (int): Number of laser emitters in the lidar, typically a multiple of 8 for mechanical lidars.
        start_azimuth_deg (float): Start azimuth angle in degrees (typically 0°), defining the beginning of the horizontal scanning range.
        end_azimuth_deg (float): End azimuth angle in degrees (typically 360°), defining the end of the horizontal scanning range.
        up_elevation_deg (float): Up elevation angle in degrees (positive), defining the maximum vertical scanning angle.
        down_elevation_deg (float): Down elevation angle in degrees (negative), defining the minimum vertical scanning angle.
        default_fire_time_ns_dt (float): Default time offset for pulse firing in nanoseconds, used to synchronize multiple emitters.
        azimuth_error_mean (float): Mean azimuth error in degrees, representing systematic error in horizontal angle measurements.
        azimuth_error_std (float): Standard deviation of azimuth error in degrees, representing random error in horizontal angle measurements.
        elevation_error_mean (float): Mean elevation error in degrees, representing systematic error in vertical angle measurements.
        elevation_error_std (float): Standard deviation of elevation error in degrees, representing random error in vertical angle measurements.

    Note:
        The configuration parameters should be set according to the specific lidar sensor being used.
        Different lidar models may have different parameter ranges and requirements.
    """

    scan_type: str | None = Field(
        default=None,
        description="Lidar scan type (e.g., rotary, linear). Rotary indicates mechanical rotation scanning, while linear indicates non-rotating scanning.",
    )
    rotation_frequency: float | None = Field(
        default=None,
        description="Lidar rotation frequency in Hz, representing how many complete rotations the sensor makes per second.",
    )
    valid_range: NDArray[Shape["2"], np.float64] | None = Field(
        default=None,
        description="Valid range of the lidar in meters [min_range, max_range]. min_range: minimum detectable distance, max_range: maximum detectable distance",
    )
    min_reflectance: float | None = Field(
        default=None,
        description="Minimum reflectance value (typically between 0 and 1) that the lidar can detect. Below this value, the sensor might fail to register a return signal.",
    )
    min_reflectance_range: float | None = Field(
        default=None,
        description="Maximum detection range in meters for objects with the specified minimum reflectance. This defines how far the sensor can detect objects with minimum reflectivity.",
    )
    avg_power: float | None = Field(
        default=None,
        description="Average power consumption of the lidar in watts during operation.",
    )
    wave_length: float | None = Field(
        default=None,
        description="Wavelength of the lidar's laser in nanometers (nm). This affects the sensor's performance in different environmental conditions.",
    )
    pulse_time: float | None = Field(
        default=None,
        description="Pulse time of the lidar in nanoseconds (ns), representing the duration of each laser pulse.",
    )
    firing_frequency: float | None = Field(
        default=None,
        description="Firing frequency of the lidar in Hz, determining how often the sensor emits laser pulses.",
    )
    num_of_point_cloud_per_data_frame: int | None = Field(
        default=None,
        description="Number of point clouds generated per data frame, representing the point cloud density of the sensor.",
    )
    num_of_emitters: int | None = Field(
        default=None,
        description="Number of laser emitters in the lidar, typically a multiple of 8 for mechanical lidars.",
    )
    start_azimuth_deg: float | None = Field(
        default=None,
        description="Start azimuth angle in degrees (typically 0°), defining the beginning of the horizontal scanning range.",
    )
    end_azimuth_deg: float | None = Field(
        default=None,
        description="End azimuth angle in degrees (typically 360°), defining the end of the horizontal scanning range.",
    )
    up_elevation_deg: float | None = Field(
        default=None,
        description="Up elevation angle in degrees (positive), defining the maximum vertical scanning angle.",
    )
    down_elevation_deg: float | None = Field(
        default=None,
        description="Down elevation angle in degrees (negative), defining the minimum vertical scanning angle.",
    )
    default_fire_time_ns_dt: float | None = Field(
        default=None,
        description="Default time offset for pulse firing in nanoseconds, used to synchronize multiple emitters.",
    )
    azimuth_error_mean: float | None = Field(
        default=None,
        description="Mean azimuth error in degrees, representing systematic error in horizontal angle measurements.",
    )
    azimuth_error_std: float | None = Field(
        default=None,
        description="Standard deviation of azimuth error in degrees, representing random error in horizontal angle measurements.",
    )
    elevation_error_mean: float | None = Field(
        default=None,
        description="Mean elevation error in degrees, representing systematic error in vertical angle measurements.",
    )
    elevation_error_std: float | None = Field(
        default=None,
        description="Standard deviation of elevation error in degrees, representing random error in vertical angle measurements.",
    )
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator(
        "rotation_frequency",
        "valid_range",
        "min_reflectance",
        "min_reflectance_range",
        "avg_power",
        "wave_length",
        "pulse_time",
        "firing_frequency",
        "num_of_point_cloud_per_data_frame",
        "start_azimuth_deg",
        "end_azimuth_deg",
        "up_elevation_deg",
        "down_elevation_deg",
        "default_fire_time_ns_dt",
        "azimuth_error_mean",
        "azimuth_error_std",
        "elevation_error_mean",
        "elevation_error_std",
        mode="before",
    )
    @classmethod
    def process_lidar_3d_sensor_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("valid_range", mode="before")
    @classmethod
    def validate_valid_range(cls, value):
        """Validates valid range values are in correct format and range.

        Args:
            value: Valid range values as list or numpy array.

        Returns:
            numpy.ndarray: Validated valid range values.

        Raises:
            ValueError: If valid range values are invalid.
        """
        # Convert to numpy array if it's a list or tuple
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 2:
                raise ValueError(
                    "Valid range must have exactly 2 components [min_range, max_range]"
                )
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if value[0] >= value[1]:
                raise ValueError("min_range must be less than max_range")
            if value[0] < 0:
                raise ValueError("min_range must be non-negative")
        return value

    @field_validator("min_reflectance", mode="before")
    @classmethod
    def validate_min_reflectance(cls, value):
        """Validates minimum reflectance value is in correct range.

        Args:
            value: Minimum reflectance value.

        Returns:
            float: Validated minimum reflectance value.

        Raises:
            ValueError: If minimum reflectance value is invalid.
        """
        if value is not None:
            if not 0 <= value <= 1:
                raise ValueError("Minimum reflectance must be between 0 and 1")
        return value

    @field_validator("start_azimuth_deg", "end_azimuth_deg", mode="before")
    @classmethod
    def validate_azimuth_range(cls, value):
        """Validates azimuth angle values are in correct range.

        Args:
            value: Azimuth angle value in degrees.

        Returns:
            float: Validated azimuth angle value.

        Raises:
            ValueError: If azimuth angle value is invalid.
        """
        if value is not None:
            if not 0 <= value <= 360:
                raise ValueError("Azimuth angle must be between 0 and 360 degrees")
        return value

    @field_validator("up_elevation_deg", "down_elevation_deg", mode="before")
    @classmethod
    def validate_elevation_range(cls, value):
        """Validates elevation angle values are in correct range.

        Args:
            value: Elevation angle value in degrees.

        Returns:
            float: Validated elevation angle value.

        Raises:
            ValueError: If elevation angle value is invalid.
        """
        if value is not None:
            if not -90 <= value <= 90:
                raise ValueError("Elevation angle must be between -90 and 90 degrees")
        return value

    @field_validator("rotation_frequency", "firing_frequency", mode="before")
    @classmethod
    def validate_frequency(cls, value):
        """Validates frequency values are positive.

        Args:
            value: Frequency value in Hz.

        Returns:
            float: Validated frequency value.

        Raises:
            ValueError: If frequency value is invalid.
        """
        if value is not None:
            if value <= 0:
                raise ValueError("Frequency must be positive")
        return value

    @field_validator("num_of_emitters", mode="before")
    @classmethod
    def validate_num_of_emitters(cls, value):
        """Validates number of emitters is positive and typically a multiple of 8.

        Args:
            value: Number of emitters.

        Returns:
            int: Validated number of emitters.

        Raises:
            ValueError: If number of emitters is invalid.
        """
        if value is not None:
            if value <= 0:
                raise ValueError("Number of emitters must be positive")
            if value % 8 != 0:
                raise ValueError(
                    "Number of emitters should typically be a multiple of 8"
                )
        return value

    @field_validator("num_of_point_cloud_per_data_frame", mode="before")
    @classmethod
    def validate_num_of_point_cloud(cls, value):
        """Validates number of point clouds per data frame is positive.

        Args:
            value: Number of point clouds per data frame.

        Returns:
            int: Validated number of point clouds.

        Raises:
            ValueError: If number of point clouds is invalid.
        """
        if value is not None:
            if value <= 0:
                raise ValueError(
                    "Number of point clouds per data frame must be positive"
                )
        return value

    @classmethod
    def load_from_file(cls, json_path: str) -> "Lidar3DSensorConfig":
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


class Lidar3DConfig(SensorConfig):
    """Configuration class for Lidar in the simulation.
    This class defines the parameters for the Lidar, including its type,
    name, position, orientation, and other configuration.

    Attributes:
        type (str): Type of the sensor (e.g., "lidar_3d")
        name (str | None): A human-readable identifier for the entity instance
        uuid (str | None): A unique identifier for the entity instance
        prim_path (str | PosixPath | None): USD primitive path for the lidar in the simulation
        position (NDArray[Shape["3"], np.float64] | None): Global position [x, y, z] relative to world frame (right-hand coordinate system)
        orientation (NDArray[Shape["4"], np.float64] | None): Global orientation quaternion [qx, qy, qz, qw] relative to world frame
        translation (NDArray[Shape["3"], np.float64] | None): Local position [x, y, z] relative to parent frame (right-hand coordinate system)
        rotation (NDArray[Shape["4"], np.float64] | None): Local orientation quaternion [qx, qy, qz, qw] relative to parent frame
        sensor_config (LidarSensorConfig): Sensor configuration
        model_config (ConfigDict): Configuration for the model validation.


    Note:
        Either global pose (position + orientation) or local pose (translation + rotation)
        must be specified, but not both. All pose components must be provided as pairs.
        The coordinate system follows the right-hand rule with:
        - X-axis: forward
        - Y-axis: left
        - Z-axis: up

        The lidar configuration should be adjusted based on the specific sensor model being simulated,
        including its scanning pattern, field of view, and performance characteristics.
    """

    type: Literal["lidar_3d"] = "lidar_3d"

    name: str | None = Field(
        ..., description="A human-readable identifier for the entity instance"
    )
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_"),
        description="A unique identifier for the entity instance",
    )
    prim_path: str | PosixPath | None = Field(
        None, description="USD primitive path for the lidar in the simulation"
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
    sensor_config: Lidar3DSensorConfig = Field(
        default_factory=Lidar3DSensorConfig, description="Sensor configuration"
    )
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator(
        "position", "orientation", "translation", "rotation", mode="before"
    )
    @classmethod
    def process_lidar_3d_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("prim_path", mode="after")
    @classmethod
    def process_lidar_3d_config_different_input_path(
        cls, value: str | PosixPath | None
    ) -> str | None:
        """Convert paths to absolute string paths."""
        return convert_to_abs_str_path(value)

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
    def set_lidar_3d_config_default_name(cls, self) -> Self:
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
    def validate_lidar_3d_config_uuid_format(cls, value: str | None) -> str | None:
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
        """Convert Lidar3DConfig to a dictionary.

        Returns:
            Dictionary representation of Lidar3DConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lidar3DConfig":
        """Create a Lidar3DConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            Lidar3DConfig: Configured lidar instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
