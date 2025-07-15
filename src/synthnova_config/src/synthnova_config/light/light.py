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
# Description: Light definitions for Synthnova
# Author: Herman Ye, Chenyu Cao@Galbot
# Date: 2025-04-29
#
#####################################################################################

from typing import Optional, Union, Literal, List, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator
import numpy as np
from numpydantic import NDArray, Shape
from functools import wraps
from pathlib import PosixPath
from ..object.object import convert_int_to_float


class Light(BaseModel):
    """Base class for all light types.

    This class represents a generic light source with configurable properties such as color,
    intensity, and temperature. It serves as the foundation for specific light implementations.
    The implementation follows the USD Lux LightAPI specification.

    Attributes:
        type: Type identifier for the light.
        active: Whether the light is active.
        translation: Translation of the light in the world in x, y, z (in meters).
        rotation: Rotation of the light in the world in roll, pitch, yaw (in radians).
        color: The color of emitted light, in the rendering color space.
        intensity: Light intensity that scales the brightness of the light linearly.
        color_temperature: Color temperature in degrees Kelvin, representing the white point.
            The default is 6500K (D65 white point). Lower values are warmer and higher values are cooler.
            Valid range is from 1000K to 10000K. Only takes effect when enable_color_temperature is true.
            When active, the computed result multiplies against the color attribute.
            The color is always calculated as an RGB color using a D65 white point, normalized such that
            the default value of 6500K will always result in white.
        enable_color_temperature: Enables using colorTemperature. Default is True.
        exposure: Scales the brightness of the light exponentially as a power of 2 (similar to an F-stop control over exposure).
        diffuse: A multiplier for the effect of this light on the diffuse response of materials.
        specular: A multiplier for the effect of this light on the specular response of materials.
        normalize: Normalizes the emission such that the power of the light remains constant while
            altering the size of the light, by dividing the luminance by the world-space surface area
            of the light. This makes it easier to independently adjust the brightness and size of the light.


    References:
        - https://openusd.org/release/api/class_usd_lux_light_a_p_i.html
    """

    type: str = Field(..., description="Type identifier for the light")
    active: bool = Field(
        default=True,
        description="Whether the light is active",
    )
    translation: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0]),
        description="Translation of the light in the world in x, y, z (in meters)",
    )
    rotation: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0]),
        description="Rotation of the light in the world in roll, pitch, yaw (in radians)",
    )
    color: NDArray[Shape["3"], np.float64] = Field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0]),
        description="The color of emitted light, in the rendering color space",
    )
    intensity: float = Field(
        default=1.0,
        description="Light intensity that scales the brightness of the light linearly",
        ge=0.0,
    )
    color_temperature: float = Field(
        default=6500.0,
        description="Color temperature in degrees Kelvin",
        ge=1000.0,
        le=10000.0,
    )
    enable_color_temperature: bool = Field(
        default=True,
        description="Enables using colorTemperature",
    )
    exposure: float = Field(
        default=0.0,
        description="Scales the brightness of the light exponentially",
        ge=-5.0,
        le=5.0,
    )
    diffuse: float = Field(
        default=1.0,
        description="Multiplier for diffuse response of materials",
        ge=0.0,
    )
    specular: float = Field(
        default=1.0,
        description="Multiplier for specular response of materials",
        ge=0.0,
    )
    normalize: bool = Field(
        default=False,
        description="Normalizes the emission such that the power of the light remains constant "
        "while altering the size of the light",
    )

    @field_validator("translation", "rotation", mode="before")
    @classmethod
    def process_light_config_int_to_float(
        cls, value: List | Tuple | np.ndarray
    ) -> List | Tuple | np.ndarray:
        """Convert integer values to float in arrays."""
        return convert_int_to_float(value)

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v: np.ndarray) -> np.ndarray:
        """Validates that the translation array has the correct shape.

        Args:
            v: The pose matrix to validate.

        Returns:
            The validated pose matrix.

        Raises:
            ValueError: If the pose matrix is not 4x4.
        """
        if v.shape != (3,):
            raise ValueError("Translation array must be 3-dimensional")
        return v

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: np.ndarray) -> np.ndarray:
        """Validates that the rotation array has the correct shape.

        Args:
            v: The rotation array to validate.

        Returns:
            The validated rotation array.

        Raises:
            ValueError: If the rotation array is not 3-dimensional.
        """
        if v.shape != (3,):
            raise ValueError("Rotation array must be 3-dimensional(roll, pitch, yaw)")
        return v

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: np.ndarray) -> np.ndarray:
        """Validates that the color array has the correct shape and values.

        Args:
            v: The color array to validate.

        Returns:
            The validated color array.

        Raises:
            ValueError: If the color array is not 3-dimensional or values are outside [0,2].
        """
        if v.shape != (3,):
            raise ValueError("Color must be a 3-dimensional array")
        if not np.all((v >= 0) & (v <= 2)):
            raise ValueError("Color values must be between 0 and 2")
        return v


class CylinderLight(Light):
    """Cylindrical area light source.

    Light emitted outward from a cylinder. The cylinder is centered at the origin and has
    its major axis on the X axis. The cylinder does not emit light from the flat end-caps.

    Attributes:
        radius: Radius of the cylinder. The default value is 0.5 units.
        length: Length of the cylinder, in the local X axis. The default value is 1.0 units.
        treatasline: A hint that this light can be treated as a 'line' light (effectively,
            a zero-radius cylinder) by renderers that benefit from non-area lighting.
            Renderers that only support area lights can disregard this.

    References:
        - https://openusd.org/release/api/class_usd_lux_cylinder_light.html
    """

    type: str = "cylinder"
    radius: float = Field(
        default=0.5,
        description="Radius of the cylinder. The cylinder is centered at the origin and has "
        "its major axis on the X axis.",
        ge=0.0,
    )
    length: float = Field(
        default=1.0,
        description="Length of the cylinder, in the local X axis.",
        gt=0.0,
    )
    treatasline: bool = Field(
        default=False,
        description="A hint that this light can be treated as a 'line' light (effectively, "
        "a zero-radius cylinder) by renderers that benefit from non-area lighting.",
    )


class DiskLight(Light):
    """Disk area light source.

    Light emitted from one side of a circular disk. The disk is centered in the XY plane
    and emits light along the -Z axis.

    Attributes:
        radius: Radius of the disk. The default value is 0.5 units.

    References:
        - https://openusd.org/release/api/class_usd_lux_disk_light.html
    """

    type: str = "disk"
    radius: float = Field(
        default=0.5,
        description="Radius of the disk. The disk is centered in the XY plane and emits light along the -Z axis.",
        ge=0.0,
    )


class DomeLight(Light):
    """Dome light source that uses an HDR texture for image-based lighting.

    Light emitted inward from a distant external environment, such as a sky or IBL light probe.
    The dome light uses an HDR texture for image-based lighting, with different texture formats
    supported for parameterization of the color map file.

    Attributes:
        texture_file: Path to the HDR texture file for IBL. The texture should be in a format
            suitable for image-based lighting, such as an HDR environment map.
        texture_format: Parameterization of the color map file. Valid values are:
            - automatic: Tries to determine layout from file (e.g., Renderman texture files)
            - latlong: Latitude as X, longitude as Y
            - mirroredBall: Environment reflected in a sphere, using orthogonal projection
            - angular: Similar to mirroredBall but with linear angle mapping
            - cubeMapVerticalCross: Cube map with faces laid out as a vertical cross

    References:
        - https://openusd.org/release/api/class_usd_lux_dome_light.html
    """

    type: str = "dome"
    texture_file: Optional[str] = Field(
        default=None,
        description="Path to the HDR texture file for IBL. The texture should be in a format "
        "suitable for image-based lighting, such as an HDR environment map.",
    )
    texture_format: Optional[
        Literal[
            "automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"
        ]
    ] = Field(
        default="automatic",
        description="Parameterization of the color map file. Valid values are: automatic, latlong, "
        "mirroredBall, angular, or cubeMapVerticalCross.",
    )


class DistantLight(Light):
    """Distant light source that simulates a light source at an infinite distance.

    Light emitted from a distant source along the -Z axis. Also known as a directional light.
    The angle parameter controls the angular diameter of the light, which affects shadow softness.
    As an example, the Sun is approximately 0.53 degrees as seen from Earth.

    Attributes:
        angle: Angular diameter of the light in degrees. Higher values broaden the light
            and therefore soften shadow edges. Range: 0 <= angle < 360. If angle is 0,
            the light represents a perfectly parallel light source. Note that this implies
            we can have a distant light emitting from more than a hemispherical area of
            light if angle > 180. While this is valid, it is possible that for large angles
            a DomeLight may provide better performance.

    References:
        - https://openusd.org/release/api/class_usd_lux_distant_light.html
    """

    type: str = "distant"
    angle: float = Field(
        default=0.53,
        description="Angular diameter of the light in degrees. Higher values broaden the light "
        "and therefore soften shadow edges. Range: 0 <= angle < 360. If angle is 0, "
        "the light represents a perfectly parallel light source.",
        ge=0.0,
        lt=360.0,
    )


class RectLight(Light):
    """Rectangular area light source.

    This light type emits light from a rectangular surface, which can be textured.
    The rectangle is centered in the XY plane and emits light along the -Z axis.
    The rectangle is 1 unit in length in the X and Y axis by default.
    In the default position, a texture file's min coordinates should be at (+X, +Y)
    and max coordinates at (-X, -Y).

    Attributes:
        width: Width of the rectangle, in the local X axis (default: 1.0)
        height: Height of the rectangle, in the local Y axis (default: 1.0)
        texture_file: Optional path to a color texture file to use on the rectangle.
            The texture coordinates are mapped such that min coordinates are at (+X, +Y)
            and max coordinates at (-X, -Y) in the default position.

    References:
        - https://openusd.org/release/api/class_usd_lux_rect_light.html
    """

    type: str = "rect"
    width: float = Field(
        default=1.0,
        description="Width of the rectangle, in the local X axis",
        gt=0.0,
    )
    height: float = Field(
        default=1.0,
        description="Height of the rectangle, in the local Y axis",
        gt=0.0,
    )
    texture_file: Optional[str] = Field(
        default=None,
        description="Path to a color texture file to use on the rectangle. "
        "Texture coordinates are mapped with min at (+X, +Y) and max at (-X, -Y) in default position.",
    )


class SphereLight(Light):
    """Spherical area light source.

    Light emitted outward from a sphere. The sphere is centered at the origin.

    Attributes:
        radius: Radius of the sphere. The default value is 0.5 units.
        treataspoint: A hint that this light can be treated as a 'point' light
            (effectively, a zero-radius sphere) by renderers that benefit from
            non-area lighting. Renderers that only support area lights can
            disregard this.

    References:
        - https://openusd.org/release/api/class_usd_lux_sphere_light.html
    """

    type: str = "sphere"
    radius: float = Field(
        default=0.5,
        description="Radius of the sphere. The sphere is centered at the origin.",
        ge=0.0,
    )
    treataspoint: bool = Field(
        default=False,
        description="A hint that this light can be treated as a 'point' light "
        "(effectively, a zero-radius sphere) by renderers that benefit "
        "from non-area lighting.",
    )
