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
# Description: Synthnova Object Config
# Author: Herman Ye@Galbot
# Date: 2025-05-10
#
#####################################################################################

from typing import Optional, Dict, Any, List, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import numpy as np
from pathlib import PosixPath
from numpydantic import NDArray, Shape
from enum import Enum
import uuid
from typing_extensions import Self


class ObjType(str, Enum):
    """Object type enumeration class.

    Defines all object types supported in Synthnova:
    - MESH: Mesh model for loading custom 3D models
    - MJCF: MuJoCo physics engine XML configuration file
    - CAPSULE: Basic primitive capsule
    - CONE: Basic primitive cone
    - CUBOID: Basic primitive cuboid
    - CYLINDER: Basic primitive cylinder
    - SPHERE: Basic primitive sphere
    """

    MESH = "mesh"
    MJCF = "mjcf"
    CAPSULE = "capsule"
    CONE = "cone"
    CUBOID = "cuboid"
    CYLINDER = "cylinder"
    SPHERE = "sphere"

    def __str__(self) -> str:
        return self.value


class CollisionType(str, Enum):
    """Collision representation type enumeration class.

    Defines different methods for object collision representation:
    - NONE: Use default collision representation
    - MESH: Use original mesh for collision representation
    - CONVEX_DECOMPOSITION: Decompose complex mesh into multiple convex polyhedra for collision representation
    - CONVEX_HULL: Use convex hull for collision representation, suitable for simple shapes
    - SDF: Use Signed Distance Field (SDF) for collision representation
    """

    NONE = "none"
    MESH = "mesh"
    CONVEX_DECOMPOSITION = "convex_decomposition"
    CONVEX_HULL = "convex_hull"
    SDF = "sdf"

    def __str__(self) -> str:
        return self.value


class SemanticType(str, Enum):
    """Semantic type enumeration class.

    Defines object semantic segmentation types:
    - CLASS: Class-level semantic label for identifying object categories (e.g., table, chair)
    - INSTANCE: Instance-level semantic label for identifying specific object instances (e.g., table1, table2)
    """

    CLASS = "class"
    INSTANCE = "instance"

    def __str__(self) -> str:
        return self.value


class InteractionType(str, Enum):
    """Interaction type enumeration class.

    Defines how an object interacts with the simulation environment:
    - DYNAMIC: Visible, participates in collision detection, and affected by physics (e.g., movable objects)
    - KINEMATIC: Visible, participates in collision detection, but controlled by external forces (e.g., robots)
    - STATIC: Visible, participates in collision detection, but fixed in place (e.g., walls, floor)
    - GHOST: Visible but passes through other objects (e.g., visual markers)
    - INVISIBLE: Not rendered and doesn't participate in physics (e.g., trigger volumes)
    """

    DYNAMIC = "dynamic"  # Affected by physics, participates in collision
    KINEMATIC = "kinematic"  # Controlled externally, participates in collision
    STATIC = "static"  # Fixed in place, participates in collision
    GHOST = "ghost"  # Visible but no collision
    INVISIBLE = "invisible"  # Not rendered, no collision

    def __str__(self) -> str:
        return self.value


class ObjectConfig(BaseModel):
    """Base configuration class for Synthnova objects.

    This class serves as the foundation for configuring objects in the Synthnova simulation environment.
    It defines common properties and behaviors shared across different object types.

    Attributes:
        prim_path (str | PosixPath): The unique primitive path in the scene graph hierarchy.
        type (ObjType | None): The type of object (mesh, mjcf, capsule, cone, cuboid, cylinder, or sphere).
        name (str | None): A human-readable identifier for the entity instance.
        uuid (str | None): A unique identifier for the entity instance.
        usd_path (str | PosixPath | None): Path to the USD (Universal Scene Description) file.
        obj_path (str | PosixPath | None): Path to the Wavefront OBJ mesh file.
        mjcf_path (str | PosixPath | None): Path to the MuJoCo XML configuration file.
        urdf_path (str | PosixPath | None): Path to the URDF file for articulated objects.
        position (NDArray[Shape["3"], np.float64] | None): Global position vector [x, y, z] in world coordinates.
        orientation (NDArray[Shape["4"], np.float64] | None): Global orientation quaternion [x, y, z, w] in world coordinates.
        translation (NDArray[Shape["3"], np.float64] | None): Local position vector [x, y, z] relative to parent.
        rotation (NDArray[Shape["4"], np.float64] | None): Local orientation quaternion [x, y, z, w] relative to parent.
        scale (NDArray[Shape["3"], np.float64]): Scale factors [x, y, z] for object dimensions.
        interaction_type (InteractionType): How the object interacts with the simulation environment.
        semantic_label (str | None): Label used for semantic segmentation.
        semantic_type (SemanticType | None): Type of semantic segmentation (class or instance).
        attributes (Dict[str, Any] | None): Additional object-specific properties.

    Note:
        Position/orientation and translation/rotation are mutually exclusive.
        Use either global coordinates (position/orientation) or local coordinates (translation/rotation),
        but not both simultaneously.
    """

    prim_path: str | PosixPath
    type: ObjType | None = None
    name: str | None = None
    uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "_")
    )
    usd_path: str | PosixPath | None = None
    obj_path: str | PosixPath | None = None
    mjcf_path: str | PosixPath | None = None
    urdf_path: str | PosixPath | None = None
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
    semantic_label: str | None = None
    semantic_type: SemanticType | None = SemanticType.CLASS
    attributes: Dict[str, Any] | None = None

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    @field_validator(
        "position", "orientation", "translation", "rotation", "scale", mode="before"
    )
    @classmethod
    def process_object_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator(
        "prim_path", "usd_path", "mjcf_path", "obj_path", "urdf_path", mode="after"
    )
    @classmethod
    def process_object_config_different_input_path(
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
    def set_object_config_default_name(cls, self) -> Self:
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
    def validate_object_config_uuid_format(cls, value: str | None) -> str | None:
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
        """Convert ObjectConfig to a dictionary.

        Returns:
            Dictionary representation of ObjectConfig
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectConfig":
        """Create an ObjectConfig from a dictionary.

        Args:
            data: Dictionary containing specific configuration data

        Returns:
            ObjectConfig: Configured object instance

        Raises:
            ValueError: If configuration data is invalid
        """
        return cls.model_validate(data)


class MeshConfig(ObjectConfig):
    """Configuration class for mesh-based objects in Synthnova.

    This class extends ObjectConfig with mesh-specific properties and validation rules.
    It supports loading geometry from USD or OBJ files with physics simulation options.

    Attributes:
        type (Literal["mesh"]): Always "mesh" for mesh-based objects.
        collision_type (CollisionType): Method for generating collision geometry.


    Raises:
        ValueError: If pose specifications are invalid or required paths are missing.
    """

    type: Literal["mesh"] = "mesh"
    collision_type: CollisionType | None = CollisionType.CONVEX_DECOMPOSITION

    @model_validator(mode="after")
    @classmethod
    def check_mesh_paths(cls, self) -> Self:
        """Check that at least one mesh path (USD, OBJ, or MJCF) is provided and exists.

        Args:
            self: The model instance

        Returns:
            Self: The model instance

        Raises:
            ValueError: If no valid mesh path is provided or if the provided path does not exist
        """
        usd_path = self.usd_path
        obj_path = self.obj_path
        mjcf_path = self.mjcf_path

        if not any([usd_path, obj_path, mjcf_path]):
            raise ValueError(
                "Mesh object requires at least one of: usd_path, obj_path, or mjcf_path"
            )

        # NOTE@Herman: Ignore path checking due to relative path support
        # for path in [usd_path, obj_path, mjcf_path]:
        #     if path and not PosixPath(path).exists():
        #         raise ValueError(f"Specified path does not exist: {path}")

        return self


class CapsuleConfig(ObjectConfig):
    """Configuration class for capsule-shaped primitive objects in Synthnova.

    This class extends ObjectConfig with capsule-specific properties including dimensions
    and visual appearance settings.

    Attributes:
        type (Literal["capsule"]): Always "capsule" for primitive capsules.
        radius (float): Radius of the capsule in meters.
        height (float): Height of the capsule in meters.
        color (NDArray[Shape["3"], np.float64]): RGB color values [r, g, b], each in range [0, 1].
    """

    type: Literal["capsule"] = "capsule"
    radius: float = Field(gt=0, default_factory=lambda: 1)
    height: float = Field(gt=0, default_factory=lambda: 1)
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    @field_validator("color", mode="before")
    @classmethod
    def process_capsule_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("color", mode="before")
    @classmethod
    def validate_color(cls, value):
        """Validates color values are in correct format and range.

        Args:
            value: RGB color values as list or numpy array.

        Returns:
            numpy.ndarray: Validated color values.

        Raises:
            ValueError: If color values are invalid.
        """
        # Convert to numpy array if it's a list or tuple
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 3:
                raise ValueError("Color must have exactly 3 components [r, g, b]")
            if not all(0 <= c <= 1 for c in value):
                raise ValueError("Color components must be in range [0, 1]")
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value


class ConeConfig(ObjectConfig):
    """Configuration class for cone-shaped primitive objects in Synthnova.

    This class extends ObjectConfig with cone-specific properties including dimensions
    and visual appearance settings.

    Attributes:
        type (Literal["cone"]): Always "cone" for primitive cones.
        radius (float): Base radius of the cone in meters.
        height (float): Height of the cone in meters.
        color (NDArray[Shape["3"], np.float64]): RGB color values [r, g, b], each in range [0, 1].
    """

    type: Literal["cone"] = "cone"
    radius: float = Field(gt=0, default_factory=lambda: 1)
    height: float = Field(gt=0, default_factory=lambda: 1)
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    @field_validator("color", mode="before")
    @classmethod
    def process_cone_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("color", mode="before")
    @classmethod
    def validate_color(cls, value):
        """Validates color values are in correct format and range."""
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 3:
                raise ValueError("Color must have exactly 3 components [r, g, b]")
            if not all(0 <= c <= 1 for c in value):
                raise ValueError("Color components must be in range [0, 1]")
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value


class CuboidConfig(ObjectConfig):
    """Configuration class for cuboid-shaped primitive objects in Synthnova.

    This class extends ObjectConfig with cuboid-specific properties including dimensions
    and visual appearance settings.

    Attributes:
        type (Literal["cuboid"]): Always "cuboid" for primitive cuboids.
        size (float): Size of the cuboid in meters (length of each edge).
        color (NDArray[Shape["3"], np.float64]): RGB color values [r, g, b], each in range [0, 1].
    """

    type: Literal["cuboid"] = "cuboid"
    size: float = Field(gt=0, default_factory=lambda: 1)
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    @field_validator("color", mode="before")
    @classmethod
    def process_cuboid_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("color", mode="before")
    @classmethod
    def validate_color(cls, value):
        """Validates color values are in correct format and range."""
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 3:
                raise ValueError("Color must have exactly 3 components [r, g, b]")
            if not all(0 <= c <= 1 for c in value):
                raise ValueError("Color components must be in range [0, 1]")
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value


class CylinderConfig(ObjectConfig):
    """Configuration class for cylinder-shaped primitive objects in Synthnova.

    This class extends ObjectConfig with cylinder-specific properties including dimensions
    and visual appearance settings.

    Attributes:
        type (Literal["cylinder"]): Always "cylinder" for primitive cylinders.
        radius (float): Radius of the cylinder in meters.
        height (float): Height of the cylinder in meters.
        color (NDArray[Shape["3"], np.float64]): RGB color values [r, g, b], each in range [0, 1].
    """

    type: Literal["cylinder"] = "cylinder"
    radius: float = Field(gt=0, default_factory=lambda: 1)
    height: float = Field(gt=0, default_factory=lambda: 1)
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    @field_validator("color", mode="before")
    @classmethod
    def process_cylinder_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("color", mode="before")
    @classmethod
    def validate_color(cls, value):
        """Validates color values are in correct format and range."""
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 3:
                raise ValueError("Color must have exactly 3 components [r, g, b]")
            if not all(0 <= c <= 1 for c in value):
                raise ValueError("Color components must be in range [0, 1]")
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value


class   SphereConfig(ObjectConfig):
    """Configuration class for sphere-shaped primitive objects in Synthnova.

    This class extends ObjectConfig with sphere-specific properties including dimensions
    and visual appearance settings.

    Attributes:
        type (Literal["sphere"]): Always "sphere" for primitive spheres.
        radius (float): Radius of the sphere in meters.
        color (NDArray[Shape["3"], np.float64]): RGB color values [r, g, b], each in range [0, 1].
    """

    type: Literal["sphere"] = "sphere"
    radius: float = Field(gt=0, default_factory=lambda: 1)
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    @field_validator("color", mode="before")
    @classmethod
    def process_sphere_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("color", mode="before")
    @classmethod
    def validate_color(cls, value):
        """Validates color values are in correct format and range."""
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value is not None:
            if len(value) != 3:
                raise ValueError("Color must have exactly 3 components [r, g, b]")
            if not all(0 <= c <= 1 for c in value):
                raise ValueError("Color components must be in range [0, 1]")
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        return value


def convert_to_abs_str_path(value: str | PosixPath | None) -> str | None:
    """Convert paths to absolute string paths.

    Args:
        value: Path to convert, can be string, PosixPath, or None

    Returns:
        str | None: Absolute path as string, or None if input is None

    Raises:
        ValueError: If path is empty or invalid type
    """
    if value is None:
        return None
    elif isinstance(value, str):
        if not value.strip():  # Check for empty or whitespace-only strings
            raise ValueError("Path cannot be empty")
        return value
    elif isinstance(value, PosixPath):
        return str(value.absolute())
    else:
        raise ValueError(f"Invalid path type: {type(value)}")


def convert_int_to_float(
    value: List | Tuple | np.ndarray | None,
) -> List | Tuple | np.ndarray | None:
    """Convert integer values to float in arrays."""
    if value is None:
        return value
    if isinstance(value, (List, Tuple)):
        return np.array(value, dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    return value
