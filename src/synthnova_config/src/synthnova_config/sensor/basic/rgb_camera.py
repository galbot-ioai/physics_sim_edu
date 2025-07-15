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
# Description: RGB camera config of Synthnova
# Author: Herman Ye@Galbot
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
from typing_extensions import Self
import uuid


class RgbSensorConfig(BaseModel):
    """Configuration class for RGB camera sensor parameters.

    This class defines the intrinsic and optical parameters of an RGB camera sensor,
    including resolution, camera matrix parameters, distortion model, and field of view settings.

    Attributes:
        frequency (float | None): Sensor frequency in Hz
        width (int | None): Image width in pixels
        height (int | None): Image height in pixels
        fx (float | None): Focal length in x direction
        fy (float | None): Focal length in y direction
        cx (float | None): Principal point x coordinate
        cy (float | None): Principal point y coordinate
        distortion_model (str | None): Camera distortion model
        distortion_coefficients (List[float] | None): Camera distortion coefficients
        pixel_size (float | None): Pixel size in mm
        f_stop (float | None): F-number, ratio of focal length to entrance pupil diameter
        focus_distance (float | None): Distance to perfect sharpness in meters
        projection_type (str | None): Camera projection type
        clipping_range (List[float] | None): Near and far clipping distances [near, far]
        horizontal_fov (float | None): Horizontal field of view in degrees
        vertical_fov (float | None): Vertical field of view in degrees
        diagonal_fov (float | None): Diagonal field of view in degrees
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    frequency: float | None = Field(
        default=None, gt=0, description="Sensor frequency in Hz"
    )
    width: int | None = Field(default=None, gt=0, description="Image width in pixels")
    height: int | None = Field(default=None, gt=0, description="Image height in pixels")
    fx: float | None = Field(default=None, description="Focal length in x direction")
    fy: float | None = Field(default=None, description="Focal length in y direction")
    cx: float | None = Field(default=None, description="Principal point x coordinate")
    cy: float | None = Field(default=None, description="Principal point y coordinate")
    distortion_model: (
        Literal["plumb_bob", "fisheye_rational_polynomial", "fisheye_kannala_brandt"]
        | None
    ) = Field(default=None, description="Camera distortion model")
    distortion_coefficients: List[float] | None = Field(
        default=None, description="Camera distortion coefficients"
    )
    pixel_size: float | None = Field(
        default=None,
        gt=0,
        description="Pixel size in mm (typically 0.003 for high resolution cameras)",
    )
    f_stop: float | None = Field(
        default=None,
        ge=0,
        description="F-number, ratio of focal length to entrance pupil diameter",
    )
    focus_distance: float | None = Field(
        default=None, ge=0, description="Distance to perfect sharpness in meters"
    )
    projection_type: (
        Literal[
            "pinhole",
            "fisheyeOrthographic",
            "fisheyeEquidistant",
            "fisheyeEquisolid",
            "fisheyePolynomial",
            "fisheyeSpherical",
            "fisheyeKannalaBrandtK3",
            "fisheyeRadTanThinPrism",
            "omniDirectionalStereo",
            "pinhole",
            "fisheyeOrthographic",
            "fisheyeEquidistant",
            "fisheyeEquisolid",
            "fisheyePolynomial",
            "fisheyeSpherical",
            "fisheyeKannalaBrandtK3",
            "fisheyeRadTanThinPrism",
            "omniDirectionalStereo",
        ]
        | None
    ) = Field(default=None, description="Camera projection type")
    clipping_range: List[float] | None = Field(
        default_factory=lambda: [0.05, 30.0],
        description="Near and far clipping distances [near, far]",
    )
    horizontal_fov: float | None = Field(
        default=None, description="Horizontal field of view in degrees"
    )
    vertical_fov: float | None = Field(
        default=None, description="Vertical field of view in degrees"
    )
    diagonal_fov: float | None = Field(
        default=None, description="Diagonal field of view in degrees"
    )

    @field_validator(
        "frequency",
        "fx",
        "fy",
        "cx",
        "cy",
        "pixel_size",
        "f_stop",
        "focus_distance",
        "vertical_fov",
        "horizontal_fov",
        "diagonal_fov",
        "clipping_range",
        mode="before",
    )
    @classmethod
    def process_rgb_sensor_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("clipping_range", mode="before")
    @classmethod
    def validate_clipping_range(cls, value):
        """Validates clipping range values are in correct format and range.

        Args:
            value: Clipping range values as list or numpy array.

        Returns:
            numpy.ndarray: Validated clipping range values.

        Raises:
            ValueError: If clipping range values are invalid.
        """
        # Convert to numpy array if it's a list or tuple
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 2:
                raise ValueError(
                    "Clipping range must have exactly 2 components [near, far]"
                )
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value

    @classmethod
    def load_from_file(cls, json_path: str) -> "RgbSensorConfig":
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


class RgbCameraConfig(SensorConfig):
    """Configuration class for RGB cameras in the simulation.

    This class defines the complete configuration for an RGB camera, including its
    position and orientation in the scene, as well as its sensor parameters.

    Attributes:
        type (str): Type of the camera, which is "rgb_camera"
        name (str | None): A human-readable identifier for the entity instance
        uuid (str | None): A unique identifier for the entity instance
        prim_path (str | PosixPath | None): USD primitive path for the camera in the simulation
        parent_entity_name (str | None): Name of the parent entity of the camera in mujoco, temporary use
        position (NDArray[Shape["3"], np.float64] | None): Global position [x, y, z]
        orientation (NDArray[Shape["4"], np.float64] | None): Global orientation quaternion [qx, qy, qz, qw]
        translation (NDArray[Shape["3"], np.float64] | None): Local position [x, y, z]
        rotation (NDArray[Shape["4"], np.float64] | None): Local orientation quaternion [qx, qy, qz, qw]
        viewport_name (str | None): Name identifier for visualization viewport
        camera_axes (str | None): Camera axes specification
        sensor_config (RgbSensorConfig): Sensor configuration

    Note:
        Either global pose (position + orientation) or local pose (translation + rotation)
        must be specified, but not both. All pose components must be provided as pairs.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["rgb_camera"] = "rgb_camera"

    name: str | None = Field(
        ..., description="Unique name for the camera in the scene manager"
    )
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_")
    )
    prim_path: str | PosixPath | None = Field(
        ..., description="USD primitive path for the camera in the simulation"
    )
    parent_entity_name: str | None = Field(
        default=None,
        description="Name of the parent entity of the camera in mujoco, temporary use",
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
    viewport_name: str | None = Field(
        default=None, description="Name identifier for visualization viewport"
    )
    camera_axes: Literal["world", "ros", "usd"] | None = Field(
        default="world",
        description="Camera axes, World means x-forward, y-right, z-down, ROS means x-right, y-down, z-forward, USD means x-right, y-up, z-back",
    )

    sensor_config: RgbSensorConfig = Field(
        default_factory=RgbSensorConfig, description="Sensor configuration"
    )

    @field_validator(
        "position", "orientation", "translation", "rotation", mode="before"
    )
    @classmethod
    def process_rgb_camera_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("prim_path", mode="after")
    @classmethod
    def process_rgb_camera_config_different_input_path(
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
    def set_rgb_camera_config_default_name(cls, self) -> Self:
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
    def validate_rgb_camera_config_uuid_format(cls, value: str | None) -> str | None:
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
        """Convert RgbCameraConfig to a dictionary.

        Returns:
            Dictionary representation of RgbCameraConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RgbCameraConfig":
        """Create an RgbCameraConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            RgbCameraConfig: Configured rgb camera instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
