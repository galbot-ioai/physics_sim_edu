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
# Description: Mujoco config of Synthnova
# Author: Herman Ye@Galbot
# Date: 2025-04-01
#
#####################################################################################

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional


class CompilerConfig(BaseModel):
    """Compiler configuration class for defining simulation parameters.

    strippath (bool): Keep full file paths when loading assets.
        Defaults to False.
    meshdir (str): Directory containing mesh files.
        Defaults to "meshes/".
    angle (str): Controls angle units.
        Valid values: "radian", "degree".
        Defaults to "radian".
    autolimits (bool): Auto-infer joint limits from ranges.
        Defaults to True.
    boundmass (float): Minimum mass for all bodies (except world).
        Must be positive. Defaults to 1e-12.
    boundinertia (float): Minimum inertia for all bodies (except world).
        Must be positive. Defaults to 1e-12.
    discardvisual (bool): Discard visual elements.
        Defaults to False.
    fusestatic (bool): Enable static body fusion optimization.
        Defaults to False.
    eulerseq (str): Euler rotation sequence.
        Valid values: "xyz", "xzy", "yxz", "yzx", "zxy", "zyx".
        Defaults to "zyx".
    inertiagrouprange (str): Range for inertia group.
        Defaults to "0 0".
    """

    strippath: bool = Field(
        default=False,
        description="Keep full file paths when loading assets",
        json_schema_extra={"examples": [True, False]},
    )
    meshdir: str = Field(
        default="meshes/",
        description="Directory containing mesh files",
        json_schema_extra={"examples": ["meshes/", "assets/meshes/"]},
    )
    angle: Literal["radian", "degree"] = Field(
        default="radian",
        description="Controls angle units",
        json_schema_extra={"examples": ["radian", "degree"]},
    )

    autolimits: bool = Field(
        default=True,
        description="Auto-infer joint limits from ranges",
        json_schema_extra={"examples": [True, False]},
    )

    boundmass: float = Field(
        default=1e-12,
        gt=0,
        description="Minimum mass for all bodies (except world)",
        json_schema_extra={"examples": [1e-12, 1e-6]},
    )

    boundinertia: float = Field(
        default=1e-12,
        gt=0,
        description="Minimum inertia for all bodies (except world)",
        json_schema_extra={"examples": [1e-12, 1e-6]},
    )

    discardvisual: bool = Field(
        default=False,
        description="Discard visual elements",
        json_schema_extra={"examples": [True, False]},
    )

    fusestatic: bool = Field(
        default=False,
        description="Enable static body fusion optimization",
        json_schema_extra={"examples": [True, False]},
    )

    eulerseq: Literal["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"] = Field(
        default="zyx",
        description="Euler rotation sequence",
        json_schema_extra={"examples": ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]},
    )

    inertiagrouprange: str = Field(
        default="0 0",
        description="Range for inertia group",
        json_schema_extra={"examples": ["0 0", "1 2"]},
    )
    model_config = ConfigDict(
        validate_assignment=True,  # Validate values on assignment
        extra="forbid",  # Prevent extra fields
        frozen=False,  # Allow field modification after creation
    )


class SizeConfig(BaseModel):
    """Size configuration class for defining simulation parameters.

    memory (str): Size of memory allocated for dynamic arrays in the mjData.arena memory space.
        Defaults to "-1" which lets the compiler guess the size.
        Can be specified with units: K (kilo), M (mega), G (giga), T (tera), P (peta), E (exa).
        Example: "16M" means 16 megabytes.
    """

    memory: str = Field(
        default="-1",
        description="Size of memory allocated for dynamic arrays in the mjData.arena memory space",
        json_schema_extra={"examples": ["-1", "16M", "32M", "1G"]},
    )
    model_config = ConfigDict(
        validate_assignment=True,  # Ensure values are validated on assignment
        extra="forbid",  # Disallow extra fields
        frozen=False,  # Allow modification after creation
    )


class MujocoConfig(BaseModel):
    """Simulator configuration class for defining simulation parameters.

    This class provides configuration options for the simulator.

    Attributes:
        timestep (float): Simulation time step in seconds.
            Must be positive. Defaults to 0.001.
        gravity (str): Gravitational acceleration vector.
            Defaults to "0 0 -9.81".
        density (float): Density of the medium (e.g., air=1.2, water=1000).
            Must be non-negative. Defaults to 0.
        impratio (float): Implicit ratio for solver.
            Must be positive. Defaults to 20.
        viscosity (float): Viscosity of the medium.
            Must be non-negative. Defaults to 0.
        integrator (str): Numerical integration method.
            Valid values: "implicitfast", "implicit", "explicit".
            Defaults to "implicitfast".
        cone (str): Type of friction cone.
            Valid values: "pyramidal", "elliptic".
            Defaults to "elliptic".
        jacobian (str): Constraint Jacobian type.
            Valid values: "dense", "sparse", "auto".
            Defaults to "auto".
        solver (str): Constraint solver algorithm.
            Valid values: "Newton", "CG", "PGS".
            Defaults to "Newton".
        iterations (int): Maximum solver iterations.
            Must be positive. Defaults to 100.
        tolerance (float): Solver termination tolerance.
            Must be positive. Defaults to 1e-8.
        compiler_config (CompilerConfig): Compiler settings.
            Defaults to CompilerConfig().
        size_config (SizeConfig): Size settings.
            Defaults to SizeConfig().
        headless (bool): Run the simulator in headless mode (no graphical output).
            Defaults to False.
        realtime_sync (bool): Run the simulator in realtime mode (sync with real time).
            Defaults to True.
        realtime_factor (float): Real-time factor for the simulator.
            Must be positive. Defaults to 1.0.

    """

    timestep: float = Field(
        default=0.001,
        gt=0,
        description="Simulation time step in seconds",
        json_schema_extra={"examples": [0.001, 0.002]},
    )

    gravity: str = Field(
        default="0 0 -9.81",
        description="Gravitational acceleration vector",
        json_schema_extra={"examples": ["0 0 -9.81", "0 0 0"]},
    )

    density: float = Field(
        default=0.0,
        ge=0,
        description="Density of the medium",
        json_schema_extra={"examples": [0, 1.2, 1000]},
    )

    impratio: float = Field(
        default=20,
        gt=0,
        description="Implicit ratio for solver",
        json_schema_extra={"examples": [20, 30]},
    )

    viscosity: float = Field(
        default=0.00002,
        ge=0,
        description="Viscosity of the medium",
        json_schema_extra={"examples": [0, 0.001]},
    )

    integrator: Literal["implicitfast", "implicit", "explicit"] = Field(
        default="implicitfast",
        description="Numerical integration method",
        json_schema_extra={"examples": ["implicitfast", "implicit", "explicit"]},
    )

    cone: Literal["pyramidal", "elliptic"] = Field(
        default="elliptic",
        description="Type of friction cone",
        json_schema_extra={"examples": ["pyramidal", "elliptic"]},
    )

    jacobian: Literal["dense", "sparse", "auto"] = Field(
        default="auto",
        description="Constraint Jacobian type",
        json_schema_extra={"examples": ["dense", "sparse", "auto"]},
    )

    solver: Literal["Newton", "CG", "PGS"] = Field(
        default="Newton",
        description="Constraint solver algorithm",
        json_schema_extra={"examples": ["Newton", "CG", "PGS"]},
    )

    iterations: int = Field(
        default=100,
        gt=0,
        description="Maximum solver iterations",
        json_schema_extra={"examples": [100, 200]},
    )

    tolerance: float = Field(
        default=1e-8,
        gt=0,
        description="Solver termination tolerance",
        json_schema_extra={"examples": [1e-8, 1e-6]},
    )

    compiler_config: CompilerConfig = Field(
        default_factory=CompilerConfig, description="Compiler settings"
    )

    size_config: SizeConfig = Field(
        default_factory=SizeConfig, description="Size settings"
    )

    model_config = ConfigDict(
        validate_assignment=True,  # Ensure values are validated on assignment
        extra="forbid",  # Disallow extra fields
        frozen=False,  # Allow modification after creation
    )

    headless: bool = Field(
        default=False,
        description="Run the simulator in headless mode (no graphical output)",
        json_schema_extra={"examples": [True, False]},
    )

    realtime_sync: bool = Field(
        default=True,
        description="Run the simulator in realtime mode (sync with real time)",
        json_schema_extra={"examples": [True, False]},
    )

    realtime_factor: float = Field(
        default=1.0,
        gt=0,
        description="Real-time factor for the simulator",
        json_schema_extra={"examples": [1.0, 2.0]},
    )
