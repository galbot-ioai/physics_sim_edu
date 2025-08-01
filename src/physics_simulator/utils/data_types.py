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
#
#####################################################################################
#
# Description: Joint trajectory data type
# Author: Herman Ye@Galbot
# Date: 2025-02-10
#
#####################################################################################
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict
import numpy as np


class JointTrajectory(BaseModel):
    """
    Defines a robotic joint trajectory consisting of joint positions, velocities, accelerations, efforts and names.

    Attributes:
        positions (Optional[np.ndarray]):
            Joint positions relative to their zero configuration.
            Expressed in radians for revolute/continuous joints and meters for prismatic joints.
        velocities (Optional[np.ndarray]):
            Joint velocities in radians per second (rad/s) for revolute joints
            and meters per second (m/s) for prismatic joints.
        accelerations (Optional[np.ndarray]):
            Joint accelerations in radians per second squared (rad/s²) for revolute joints
            and meters per second squared (m/s²) for prismatic joints.
        efforts (Optional[np.ndarray]):
            Joint efforts, expressed in Newton-meters (Nm) for revolute joints
            and Newtons (N) for prismatic joints.
        joint_names (Optional[List[str]]):
            Names of the joints, defining the order of numerical data.
        dt (Optional[float]):
            Time step duration in seconds.
            If specified, it must be mutually exclusive with `frequency`.
        frequency (Optional[float]):
            Execution frequency in Hertz (Hz). Must be positive if specified.
        time_from_start (Optional[np.ndarray]):
            Time durations (in seconds) representing the desired time from the start
            of the trajectory to each trajectory point.

    Notes:
        - Joint data arrays (positions, velocities, accelerations, efforts) are expected to be 2D arrays with
          shape (n_points, n_joints) where the first dimension corresponds to trajectory points and the
          second dimension corresponds to joints.
        - All non-empty joint data arrays must have consistent shapes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    efforts: Optional[np.ndarray] = None
    joint_names: Optional[List[str]] = None
    dt: Optional[float] = None
    frequency: Optional[float] = None
    time_from_start: Optional[np.ndarray] = None

    def model_post_init(self, __context) -> None:
        def process_array(name: str, array: Optional[Any]) -> Optional[np.ndarray]:
            if array is None:
                return None

            arr = np.asarray(array)
            # Shape validfollow_trajectoryueError(f"{name} must be a 2D array, got {arr.ndim}D array")

            # Validate dimensions: first dimension is points, second dimension is joints.
            if self.time_from_start is not None and arr.shape[0] != len(
                self.time_from_start
            ):
                raise ValueError(
                    f"{name} has {arr.shape[0]} points, but {len(self.time_from_start)} time points"
                )
            if self.joint_names is not None and arr.shape[1] != len(self.joint_names):
                raise ValueError(
                    f"{name} has {arr.shape[1]} joints, but {len(self.joint_names)} named joints"
                )
            return arr

        # Validate frequency and dt
        if self.frequency is not None and self.dt is not None:
            if self.frequency != 1 / self.dt:
                raise ValueError(
                    "frequency and dt are mutually exclusive. Only one can be specified."
                )
        if self.frequency is not None and self.frequency <= 0:
            raise ValueError("Frequency must be a positive number.")
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be a positive number.")

        # Validate time_from_start
        if self.time_from_start is not None and self.time_from_start.ndim != 1:
            raise ValueError("time_from_start must be a 1D array.")

        # Validate time sequence
        if self.time_from_start is not None:
            if not np.all(np.diff(self.time_from_start) > 0):
                raise ValueError("time_from_start must be strictly increasing")

        # Validate numpy arrays
        self.positions = process_array("positions", self.positions)
        self.velocities = process_array("velocities", self.velocities)
        self.accelerations = process_array("accelerations", self.accelerations)
        self.efforts = process_array("efforts", self.efforts)

        # Validate consistency of shapes
        shapes = []
        for data in [self.positions, self.velocities, self.accelerations, self.efforts]:
            if data is not None:
                shapes.append(data.shape)

        if len(set(shapes)) > 1:
            raise ValueError(
                f"Inconsistent joint data shapes: {shapes}. "
                "All non-empty arrays must have the same shape"
            )

        # Validate joint names and data: now joint_names should match the second dimension (joints)
        if self.joint_names is not None:
            joint_names_length = len(self.joint_names)
            joint_data_length = shapes[0][1] if shapes else 0
            if joint_names_length != joint_data_length:
                raise ValueError(
                    f"Joint names length ({joint_names_length}) "
                    f"does not match joint data shape ({joint_data_length})"
                )

        # Generate time_from_start if needed based on frequency or dt.
        if self.frequency is not None and self.time_from_start is None:
            # Use the first available array to determine n_points (now: number of points = shape[0])
            for data in [
                self.positions,
                self.velocities,
                self.accelerations,
                self.efforts,
            ]:
                if data is not None:
                    n_points = data.shape[0]
                    self.time_from_start = np.linspace(
                        0, (n_points - 1) / self.frequency, n_points
                    )
                    break
        elif self.dt is not None and self.time_from_start is None:
            # If dt is specified, generate time_from_start based on dt.
            n_points = self.positions.shape[0] if self.positions is not None else 0
            self.time_from_start = np.arange(0, n_points * self.dt, self.dt)

        # Get dt or frequency from either value
        if self.dt is None and self.frequency is not None:
            self.dt = 1 / self.frequency
        elif self.dt is not None and self.frequency is None:
            self.frequency = 1 / self.dt

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        return cls.model_construct(**data)

    def __str__(self) -> str:
        """User-friendly string representation"""
        lines = ["Joint Trajectory Summary:"]

        if self.joint_names:
            lines.append(
                f"- Joints ({len(self.joint_names)}): {', '.join(self.joint_names)}"
            )

        if self.frequency:
            lines.append(f"- Frequency: {self.frequency:.2f} Hz")

        if self.dt:
            lines.append(f"- Time step: {self.dt:.2f} seconds")

        if self.time_from_start is not None:
            duration = self.time_from_start[-1] if len(self.time_from_start) > 0 else 0
            lines.append(
                f"- Duration: {duration:.2f}s ({len(self.time_from_start)} points)"
            )
            lines.append(
                f"- Time range: [{self.time_from_start[0]:.2f} -> {self.time_from_start[-1]:.2f}]s"
            )

        for name in ["positions", "velocities", "accelerations", "efforts"]:
            arr = getattr(self, name)
            if arr is not None:
                # Now shape: (points, joints)
                lines.append(
                    f"- {name.capitalize()}: {arr.shape[0]}x{arr.shape[1]} (points x joints)"
                )

        return "\n".join(lines)

    @property
    def data(self):
        """Return the dictionary representation of the trajectory"""
        return self.to_dict()

    def __len__(self) -> int:
        for data in [self.positions, self.velocities, self.accelerations, self.efforts]:
            if data is not None:
                return len(data)
        return 0

    def __getitem__(self, t: int) -> "JointTrajectory":
        def get_t(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if data is None:
                return None
            return data[t][None]

        return JointTrajectory(
            positions=get_t(self.positions),
            velocities=get_t(self.velocities),
            accelerations=get_t(self.accelerations),
            efforts=get_t(self.efforts),
            joint_names=self.joint_names,
            dt=self.dt,
            frequency=self.frequency,
            time_from_start=self.time_from_start,
        )
