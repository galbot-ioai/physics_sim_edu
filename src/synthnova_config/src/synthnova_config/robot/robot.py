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
# Description: Robot configuration module for Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-04-02
#
#####################################################################################
from typing import Optional, Dict, Any, List, Literal, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pathlib import PosixPath
import numpy as np
from numpydantic import NDArray, Shape
from typing_extensions import Self
from ..object import InteractionType
from ..object import SemanticType
from ..object import CollisionType
from ..object.object import convert_int_to_float, convert_to_abs_str_path
import uuid


class RobotConfig(BaseModel):
    """Base configuration class for Synthnova robots.

    This class serves as the foundation for configuring robots in the Synthnova simulation environment.

    Attributes:
        prim_path (str | PosixPath): The unique primitive path in the scene graph hierarchy.
        name (str | None): A human-readable identifier for the object instance.
        uuid (str | None): A unique identifier for the object instance.
        usd_path (str | PosixPath | None): Path to the USD (Universal Scene Description) file.
        mjcf_path (str | PosixPath | None): Path to the MuJoCo XML configuration file.
        urdf_path (str | PosixPath | None): Path to the URDF file for robot kinematics.
        srdf_path (str | PosixPath | None): Path to the SRDF file for robot semantic information.
        position (NDArray[Shape["3"], np.float64] | None): Global position vector [x, y, z] in world coordinates.
        orientation (NDArray[Shape["4"], np.float64] | None): Global orientation quaternion [x, y, z, w] in world coordinates.
        translation (NDArray[Shape["3"], np.float64] | None): Local position vector [x, y, z] relative to parent.
        rotation (NDArray[Shape["4"], np.float64] | None): Local orientation quaternion [x, y, z, w] relative to parent.
        scale (NDArray[Shape["3"], np.float64]): Scale factors [x, y, z] for object dimensions.
        interaction_type (InteractionType): How the object interacts with the simulation environment.
        collision_type (CollisionType): Method for generating collision geometry.
        semantic_label (str | None): Label used for semantic segmentation.
        semantic_type (SemanticType | None): Type of semantic segmentation (class or instance).
        attributes (Dict[str, Any] | None): Additional object-specific properties.

    Note:
        Due to the diverse nature of robot configurations, a single configuration file cannot cover all possible use cases.
        Therefore, RobotConfig is designed to be extensible, allowing users to inherit from this base class or add their own
        custom fields as needed. The fields provided here are recommended configurations that cover common use cases.

        Position/orientation and translation/rotation are mutually exclusive.
        Use either global coordinates (position/orientation) or local coordinates (translation/rotation),
        but not both simultaneously.

    Example:
        To create a custom robot configuration:
        ```python
        class CustomRobotConfig(RobotConfig):
            custom_field: str
            additional_parameters: Dict[str, Any]
        ```
    """

    prim_path: str | PosixPath
    name: str | None = None
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_")
    )
    usd_path: str | PosixPath | None = None
    mjcf_path: str | PosixPath | None = None
    urdf_path: str | PosixPath | None = None
    srdf_path: str | PosixPath | None = None
    position: NDArray[Shape["3"], np.float64] | None = None
    orientation: NDArray[Shape["4"], np.float64] | None = None
    translation: NDArray[Shape["3"], np.float64] | None = None
    rotation: NDArray[Shape["4"], np.float64] | None = None
    scale: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0])
    )
    interaction_type: InteractionType = Field(
        default_factory=lambda: InteractionType.DYNAMIC
    )
    collision_type: CollisionType | None = CollisionType.CONVEX_DECOMPOSITION
    semantic_label: str | None = None
    semantic_type: SemanticType | None = SemanticType.CLASS
    attributes: Dict[str, Any] | None = None

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator(
        "position", "orientation", "translation", "rotation", "scale", mode="before"
    )
    @classmethod
    def process_object_config_int_to_float(
        cls, value: List | Tuple | np.ndarray | None
    ) -> List | Tuple | np.ndarray | None:
        """Convert integer values to float in arrays.

        This function converts integer values in arrays to float64 type.
        It handles different input types including lists, tuples, and numpy arrays.

        Args:
            value (List | Tuple | np.ndarray | None): Input array to convert.
                Can be a list, tuple, numpy array, or None.

        Returns:
            List | Tuple | np.ndarray | None: Converted array with float64 type.
                Returns None if input is None.
                Returns numpy array for list/tuple inputs.
                Returns converted numpy array for numpy array inputs.

        Example:
            >>> process_object_config_int_to_float([1, 2, 3])
            array([1., 2., 3.], dtype=float64)
        """
        return convert_int_to_float(value)

    @field_validator(
        "prim_path", "usd_path", "mjcf_path", "urdf_path", "srdf_path", mode="after"
    )
    @classmethod
    def process_object_config_different_input_path(
        cls, value: str | PosixPath | None
    ) -> str | None:
        """Convert paths to absolute string paths.

        This function converts different path types to absolute string paths.
        It handles string paths, PosixPath objects, and None values.

        Args:
            value (str | PosixPath | None): Path to convert.
                Can be a string path, PosixPath object, or None.

        Returns:
            str | None: Absolute path as string, or None if input is None.

        Raises:
            ValueError: If path is empty or invalid type.
        """
        return convert_to_abs_str_path(value)

    @field_validator("uuid", mode="before")
    @classmethod
    def validate_uuid_format(cls, value: str | None) -> str | None:
        """Validate and convert UUID format to use underscores instead of hyphens.

        This function validates UUID format and converts hyphens to underscores.
        It handles string inputs and None values.

        Args:
            value (str | None): UUID string to validate.
                Can be a string or None.

        Returns:
            str | None: UUID string with underscores, or None if input is None.

        Raises:
            ValueError: If UUID format is invalid.
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

        This function sets a default name for the robot using its UUID if no name
        is provided. The name format is "robot_{uuid}" where uuid uses underscores
        instead of hyphens.

        Args:
            self: The model instance.

        Returns:
            Self: The model instance with default name set if needed.
        """
        if self.name is None:
            # Convert UUID to use underscores instead of hyphens
            uuid_str = self.uuid.replace("-", "_")
            self.name = f"robot_{uuid_str}"
        return self

    @model_validator(mode="after")
    @classmethod
    def check_pose_consistency(cls, self) -> Self:
        """Check that either position/orientation or translation/rotation is used, but not both.

        This function validates that the robot's pose is specified using either
        global coordinates (position/orientation) or local coordinates (translation/rotation),
        but not both simultaneously.

        Args:
            self: The model instance.

        Returns:
            Self: The model instance.

        Raises:
            ValueError: If both global and local coordinates are used simultaneously.
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
    def check_robot_paths(cls, self) -> Self:
        """Check that at least one robot path is provided.

        This function validates that at least one of the robot's configuration paths
        (USD, MJCF, URDF, or SRDF) is provided. It ensures that the robot has
        the necessary configuration files for simulation.

        Args:
            self: The model instance.

        Returns:
            Self: The model instance.

        Raises:
            ValueError: If no valid robot path is provided.
        """
        usd_path = self.usd_path
        mjcf_path = self.mjcf_path
        urdf_path = self.urdf_path
        srdf_path = self.srdf_path

        if not any([usd_path, mjcf_path, urdf_path, srdf_path]):
            raise ValueError(
                "Robot object requires at least one of: usd_path, mjcf_path, urdf_path, or srdf_path"
            )
        # NOTE@Herman: Ignore path checking due to relative path support
        # for path in [usd_path, mjcf_path, urdf_path, srdf_path]:
        #     if path and not PosixPath(path).exists():
        #         raise ValueError(f"Specified path does not exist: {path}")

        return self
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert RobotConfig to a dictionary.

        Returns:
            Dictionary representation of RobotConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """Create a RobotConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            RobotConfig: Configured robot instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)
